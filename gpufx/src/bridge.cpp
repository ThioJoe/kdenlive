/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * The bridge between MLT's image stack and the GPU.
 *
 * Supported filters keep their own process() function: it pushes the
 * filter and its CPU get_image onto the frame. wrappedProcess() then slides
 * bridgeGetImage() on top of them, so that for every frame the bridge can
 * decide to call the CPU get_image unchanged or to do the work on the GPU.
 *
 * Frames move between the host and the GPU per effect, so any mix of
 * accelerated and CPU-only effects works. When two accelerated effects follow
 * each other in the stack the image stays in device memory between them:
 * the upper one sets kWantDevice on the frame before fetching its input, and
 * the lower one then attaches its device result to the frame (kDeviceImage)
 * instead of downloading it. The host pointer returned along with it is only
 * a tag in that case, its content is stale.
 */

#include "bridge.h"
#include "gpufx_log.h"
#include "mlt_compat.h"

#include <cstring>
#include <memory>

namespace gpufx {

namespace {

const char *const kWrapProperty = "_gpufx.wrap";
const char *const kWantDevice = "_gpufx.want_device";
const char *const kDeviceImage = "_gpufx.device_image";
const char *const kForceCpu = "_gpufx.disable";

struct WrapInfo
{
    mlt_frame (*originalProcess)(mlt_filter, mlt_frame);
    const EffectDescriptor *effect;
};

void destroyWrapInfo(void *data)
{
    delete static_cast<WrapInfo *>(data);
}

struct DeviceImage
{
    DeviceBuffer buffer;
    int width = 0;
    int height = 0;
    const uint8_t *tag = nullptr;
};

struct DeviceImageBox
{
    std::unique_ptr<DeviceImage> image;
};

void destroyDeviceImageBox(void *data)
{
    delete static_cast<DeviceImageBox *>(data);
}

int bridgeGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable);
int bridgeTransitionGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable);

bool isBridge(void *getImage)
{
    return getImage == reinterpret_cast<void *>(&bridgeGetImage) || getImage == reinterpret_cast<void *>(&bridgeTransitionGetImage);
}

/** True if the get_image that will run after the entry at depth (0 = top) is
 *  another bridge, i.e. the next filter down the stack runs through gpufx. */
bool nextIsBridge(mlt_frame frame, int depth)
{
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int count = mlt_deque_count(stack);
    return count > depth && isBridge(mlt_deque_peek(stack, count - 1 - depth));
}

/** The same for the image of a transition's frame, where depth is the first
 *  entry below the transition's own. MLT's tractor puts a hook there that
 *  only sets some frame properties: [transition, get_image_a] on the bottom
 *  frame, [bottom frame, transition, get_image_b] on the top frame. The image
 *  passes through it unchanged, so the bridge below it can be used. */
bool belowTransitionIsBridge(mlt_frame frame, int depth, mlt_transition transition, mlt_frame hookFrame)
{
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int count = mlt_deque_count(stack);
    auto entry = [&](int d) { return count > d ? mlt_deque_peek(stack, count - 1 - d) : nullptr; };
    if (entry(depth + 1) == transition) {
        depth += 2;
        if (hookFrame && entry(depth) == hookFrame) {
            depth += 1;
        }
    }
    return isBridge(entry(depth));
}

bool takeFlag(mlt_properties properties, const char *name)
{
    if (mlt_properties_get_int(properties, name)) {
        mlt_properties_set_int(properties, name, 0);
        return true;
    }
    return false;
}

void attachDeviceImage(mlt_properties properties, std::unique_ptr<DeviceImage> image)
{
    auto *box = new DeviceImageBox{std::move(image)};
    mlt_properties_set_data(properties, kDeviceImage, box, 0, destroyDeviceImageBox, nullptr);
}

std::unique_ptr<DeviceImage> takeDeviceImage(mlt_properties properties, const uint8_t *tag, int width, int height, mlt_image_format format)
{
    auto *box = static_cast<DeviceImageBox *>(mlt_properties_get_data(properties, kDeviceImage, nullptr));
    if (!box) {
        return nullptr;
    }
    std::unique_ptr<DeviceImage> image = std::move(box->image);
    // Deletes the now empty box
    mlt_properties_set_data(properties, kDeviceImage, nullptr, 0, nullptr, nullptr);
    if (!image) {
        return nullptr;
    }
    if (image->tag != tag || image->width != width || image->height != height || format != mlt_image_rgba) {
        logWarning("ignoring a GPU image that does not match the frame image");
        return nullptr;
    }
    return image;
}

/** Runs the original CPU get_image after the input image has already been
 *  fetched, e.g. when a CUDA call failed. The image stack is empty at this
 *  point, so the CPU filter gets the frame's image back from its properties. */
int cpuFallback(mlt_frame frame, mlt_filter filter, mlt_get_image cpuGetImage, uint8_t *input, std::unique_ptr<uint8_t, void (*)(void *)> hostCopy,
                int inputWidth, int inputHeight, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    if (hostCopy) {
        // Pixels downloaded from the GPU, they become the frame image
        const int size = inputWidth * inputHeight * 4;
        mlt_frame_set_image(frame, hostCopy.release(), size, mlt_pool_release);
        mlt_properties_set_int(properties, "format", mlt_image_rgba);
        mlt_properties_set_int(properties, "width", inputWidth);
        mlt_properties_set_int(properties, "height", inputHeight);
    } else if (input && mlt_properties_get_data(properties, "image", nullptr) != input) {
        // Make sure the CPU filter finds this image on the frame
        const int size = inputWidth * inputHeight * 4;
        auto *copy = static_cast<uint8_t *>(mlt_pool_alloc(size));
        if (!copy) {
            return 1;
        }
        memcpy(copy, input, size_t(size));
        mlt_frame_set_image(frame, copy, size, mlt_pool_release);
        mlt_properties_set_int(properties, "format", mlt_image_rgba);
        mlt_properties_set_int(properties, "width", inputWidth);
        mlt_properties_set_int(properties, "height", inputHeight);
    }
    Runtime::instance().countFrame(false);
    mlt_frame_push_service(frame, filter);
    return cpuGetImage(frame, image, format, width, height, writable);
}

/** Downloads an input image that only exists on the GPU, so that the CPU
 *  filter can process it. Returns nullptr if that fails. */
std::unique_ptr<uint8_t, void (*)(void *)> rescueInput(GpuScope &scope, const DeviceBuffer &in, size_t bytes)
{
    std::unique_ptr<uint8_t, void (*)(void *)> copy(static_cast<uint8_t *>(mlt_pool_alloc(int(bytes))), mlt_pool_release);
    CUresult res = copy ? scope.download(copy.get(), in.ptr(), bytes) : CUDA_ERROR_OUT_OF_MEMORY;
    if (res == CUDA_SUCCESS) {
        res = scope.synchronize();
    }
    if (res != CUDA_SUCCESS) {
        copy.reset();
    }
    return copy;
}

int runOnGpu(mlt_frame frame, mlt_filter filter, mlt_get_image cpuGetImage, const EffectDescriptor &effect, EffectPlan &plan, bool callerWantsDevice,
             bool lowerOnGpu, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    Runtime &runtime = Runtime::instance();
    EffectJob &job = *plan.job;

    // Our filter entry, the stack now starts with the image source
    mlt_frame_pop_service(frame);

    uint8_t *input = nullptr;
    mlt_image_format inputFormat = mlt_image_rgba;
    int w = plan.inputWidth > 0 ? plan.inputWidth : *width;
    int h = plan.inputHeight > 0 ? plan.inputHeight : *height;
    job.beforeFetch(frame);
    if (lowerOnGpu) {
        mlt_properties_set_int(properties, kWantDevice, 1);
    }
    const int error = mlt_frame_get_image(frame, &input, &inputFormat, &w, &h, 0);
    mlt_properties_set_int(properties, kWantDevice, 0);
    std::unique_ptr<DeviceImage> deviceInput = takeDeviceImage(properties, input, w, h, inputFormat);
    if (error || !input) {
        *image = input;
        *format = inputFormat;
        *width = w;
        *height = h;
        return error ? error : 1;
    }
    const size_t bytes = size_t(w) * size_t(h) * 4;
    const bool inputOnDevice = deviceInput != nullptr;
    std::unique_ptr<uint8_t, void (*)(void *)> rescued(nullptr, mlt_pool_release);
    if (inputFormat != mlt_image_rgba || w <= 0 || h <= 0 || !job.prepare(w, h)) {
        if (inputOnDevice) {
            GpuScope scope;
            if (scope.ok()) {
                rescued = rescueInput(scope, deviceInput->buffer, bytes);
            }
            if (!rescued) {
                logError("%s: lost the GPU image", effect.service);
                return 1;
            }
        }
        job.restoreFrame(frame);
        return cpuFallback(frame, filter, cpuGetImage, input, std::move(rescued), w, h, image, format, width, height, writable);
    }

    const int outW = plan.outputWidth > 0 ? plan.outputWidth : w;
    const int outH = plan.outputHeight > 0 ? plan.outputHeight : h;
    const size_t outBytes = size_t(outW) * size_t(outH) * 4;
    CUresult res = CUDA_ERROR_NOT_INITIALIZED;
    {
        GpuScope scope;
        if (scope.ok()) {
            DeviceBuffer in;
            DeviceBuffer out;
            if (inputOnDevice) {
                in = std::move(deviceInput->buffer);
                res = CUDA_SUCCESS;
            } else {
                res = in.allocate(bytes);
                if (res == CUDA_SUCCESS) {
                    res = scope.upload(in.ptr(), input, bytes);
                }
            }
            if (res == CUDA_SUCCESS) {
                res = out.allocate(outBytes);
            }
            if (res == CUDA_SUCCESS) {
                res = job.runResized(scope, w, h, in.ptr(), outW, outH, out.ptr());
            }
            if (res == CUDA_SUCCESS) {
                if (callerWantsDevice) {
                    // The caller continues on the GPU: keep the result there
                    res = scope.synchronize();
                    uint8_t *tag = input;
                    if (res == CUDA_SUCCESS && outBytes != bytes) {
                        // The host image stands for the GPU one, give it the right size
                        tag = static_cast<uint8_t *>(mlt_pool_alloc(int(outBytes)));
                        if (tag) {
                            mlt_frame_set_image(frame, tag, int(outBytes), mlt_pool_release);
                        } else {
                            res = CUDA_ERROR_OUT_OF_MEMORY;
                        }
                    }
                    if (res == CUDA_SUCCESS) {
                        auto handoff = std::make_unique<DeviceImage>();
                        handoff->buffer = std::move(out);
                        handoff->width = outW;
                        handoff->height = outH;
                        handoff->tag = tag;
                        attachDeviceImage(properties, std::move(handoff));
                        *image = tag;
                        *format = mlt_image_rgba;
                        *width = outW;
                        *height = outH;
                        runtime.countFrame(true);
                        logDebug("%s: %dx%d on GPU, kept on GPU", effect.service, outW, outH);
                        return 0;
                    }
                } else {
                    auto *result = static_cast<uint8_t *>(mlt_pool_alloc(int(outBytes)));
                    res = result ? scope.download(result, out.ptr(), outBytes) : CUDA_ERROR_OUT_OF_MEMORY;
                    if (res == CUDA_SUCCESS) {
                        res = scope.synchronize();
                    }
                    if (res == CUDA_SUCCESS) {
                        mlt_frame_set_image(frame, result, int(outBytes), mlt_pool_release);
                        *image = result;
                        *format = mlt_image_rgba;
                        *width = outW;
                        *height = outH;
                        runtime.countFrame(true);
                        logDebug("%s: %dx%d on GPU", effect.service, outW, outH);
                        return 0;
                    }
                    if (result) {
                        mlt_pool_release(result);
                    }
                }
            }
            runtime.handleError(res, effect.service);
            if (inputOnDevice && in.valid()) {
                // The host image is stale: get the pixels back for the CPU filter
                rescued = rescueInput(scope, in, bytes);
            }
        } else if (inputOnDevice) {
            // No usable context, the input only exists on the GPU
            res = CUDA_ERROR_NOT_INITIALIZED;
        }
    }
    if (inputOnDevice && !rescued) {
        logError("%s: lost the GPU image after a CUDA error", effect.service);
        return 1;
    }
    logDebug("%s: falling back to the CPU", effect.service);
    job.restoreFrame(frame);
    return cpuFallback(frame, filter, cpuGetImage, input, std::move(rescued), w, h, image, format, width, height, writable);
}

int bridgeGetImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    auto cpuGetImage = reinterpret_cast<mlt_get_image>(mlt_frame_pop_service(frame));
    // Always consume the flag, it is only meant for the get_image right below its setter
    const bool callerWantsDevice = takeFlag(properties, kWantDevice);
    auto filter = static_cast<mlt_filter>(mlt_deque_peek_back(MLT_FRAME_IMAGE_STACK(frame)));
    if (!cpuGetImage || !filter) {
        return 1;
    }
    mlt_properties filterProperties = MLT_FILTER_PROPERTIES(filter);
    auto *info = static_cast<WrapInfo *>(mlt_properties_get_data(filterProperties, kWrapProperty, nullptr));
    if (!info || !moduleEnabled() || !effectEnabled(*info->effect) || mlt_properties_get_int(filterProperties, kForceCpu)) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    const EffectDescriptor &effect = *info->effect;

    EffectContext ctx;
    ctx.filter = filter;
    ctx.frame = frame;
    ctx.properties = filterProperties;
    ctx.format = *format;
    ctx.width = *width;
    ctx.height = *height;
    mlt_service_lock(MLT_FILTER_SERVICE(filter));
    ctx.position = mlt_filter_get_position(filter, frame);
    ctx.length = mlt_filter_get_length2(filter, frame);
    EffectPlan plan = effect.plan(ctx);
    mlt_service_unlock(MLT_FILTER_SERVICE(filter));

    const bool lowerOnGpu = nextIsBridge(frame, 1);
    switch (plan.decision) {
    case Decision::PassThrough:
        if (callerWantsDevice && lowerOnGpu && *format == mlt_image_rgba) {
            // Nothing to do: let the image travel through on the GPU
            mlt_frame_pop_service(frame);
            mlt_properties_set_int(properties, kWantDevice, 1);
            const int error = mlt_frame_get_image(frame, image, format, width, height, writable);
            mlt_properties_set_int(properties, kWantDevice, 0);
            return error;
        }
        return cpuGetImage(frame, image, format, width, height, writable);
    case Decision::Cpu:
        Runtime::instance().countFrame(false);
        return cpuGetImage(frame, image, format, width, height, writable);
    case Decision::Gpu:
        break;
    }
    if (!plan.job) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    // A cheap effect on its own would spend more time copying the frame to
    // the GPU and back than the CPU needs to process it
    if (effect.lightweight && !callerWantsDevice && !(lowerOnGpu && *format == mlt_image_rgba)) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    if (!Runtime::instance().ensureInitialized()) {
        return cpuGetImage(frame, image, format, width, height, writable);
    }
    return runOnGpu(frame, filter, cpuGetImage, effect, plan, callerWantsDevice, lowerOnGpu, image, format, width, height, writable);
}

/* ==========================================================================
 * Transitions
 * ========================================================================== */

struct TransitionWrapInfo
{
    mlt_frame (*originalProcess)(mlt_transition, mlt_frame, mlt_frame);
    const TransitionDescriptor *descriptor;
};

void destroyTransitionWrapInfo(void *data)
{
    delete static_cast<TransitionWrapInfo *>(data);
}

bool isRgba(mlt_image_format format)
{
    return format == mlt_image_rgba;
}

/** An image fetched by the transition: in host memory, or on the GPU with a
 *  stale host buffer standing for it. */
struct FetchedImage
{
    uint8_t *host = nullptr;
    mlt_image_format format = mlt_image_rgba;
    int width = 0;
    int height = 0;
    std::unique_ptr<DeviceImage> device;
};

/** Brings an image that only exists on the GPU back to its frame, for the CPU. */
bool ensureHost(mlt_frame frame, FetchedImage &fetched)
{
    if (!fetched.device) {
        return true;
    }
    const size_t bytes = size_t(fetched.width) * size_t(fetched.height) * 4;
    std::unique_ptr<uint8_t, void (*)(void *)> copy(nullptr, mlt_pool_release);
    {
        GpuScope scope;
        if (scope.ok()) {
            copy = rescueInput(scope, fetched.device->buffer, bytes);
        }
    }
    fetched.device.reset();
    if (!copy) {
        return false;
    }
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    fetched.host = copy.release();
    mlt_frame_set_image(frame, fetched.host, int(bytes), mlt_pool_release);
    mlt_properties_set_int(properties, "format", mlt_image_rgba);
    mlt_properties_set_int(properties, "width", fetched.width);
    mlt_properties_set_int(properties, "height", fetched.height);
    return true;
}

/** Downloads an image of the GPU to its frame (the host buffer was stale). */
bool restoreToFrame(GpuScope &scope, const DeviceBuffer &buffer, mlt_frame frame, FetchedImage &fetched)
{
    const size_t bytes = size_t(fetched.width) * size_t(fetched.height) * 4;
    std::unique_ptr<uint8_t, void (*)(void *)> copy =
        buffer.valid() ? rescueInput(scope, buffer, bytes) : std::unique_ptr<uint8_t, void (*)(void *)>(nullptr, mlt_pool_release);
    if (!copy) {
        return false;
    }
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    fetched.host = copy.release();
    mlt_frame_set_image(frame, fetched.host, int(bytes), mlt_pool_release);
    mlt_properties_set_int(properties, "format", mlt_image_rgba);
    mlt_properties_set_int(properties, "width", fetched.width);
    mlt_properties_set_int(properties, "height", fetched.height);
    return true;
}

/** Hands the frame to the CPU transition after images were fetched: they are
 *  on their frames, where the CPU transition finds them again. */
int transitionCpuFallback(mlt_frame aFrame, mlt_transition transition, mlt_frame bFrame, mlt_get_image cpuGetImage, FetchedImage &top, FetchedImage &bottom,
                          uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    if (!ensureHost(bFrame, top) || !ensureHost(aFrame, bottom)) {
        logError("lost a GPU image after a CUDA error");
        return 1;
    }
    Runtime::instance().countFrame(false);
    mlt_frame_push_service(aFrame, transition);
    mlt_frame_push_frame(aFrame, bFrame);
    return cpuGetImage(aFrame, image, format, width, height, writable);
}

/** mlt_image_is_opaque() of an RGBA image on the GPU. Returns a CUDA error code. */
CUresult deviceImageOpaque(const DeviceImage &image, bool &opaque)
{
    GpuScope scope;
    if (!scope.ok()) {
        return CUDA_ERROR_NOT_INITIALIZED;
    }
    DeviceBuffer flag;
    CUresult res = flag.allocate(sizeof(int));
    if (res == CUDA_SUCCESS) {
        res = scope.fill(flag.ptr(), 0, sizeof(int));
    }
    const int count = image.width * image.height;
    if (res == CUDA_SUCCESS) {
        res = scope.launch1D(KernelId::gpufx_alpha_check, count, image.buffer.ptr(), count, flag.ptr());
    }
    int transparent = 0;
    if (res == CUDA_SUCCESS) {
        res = scope.download(&transparent, flag.ptr(), sizeof(int));
    }
    if (res == CUDA_SUCCESS) {
        res = scope.synchronize();
    }
    opaque = transparent == 0;
    return res;
}

int runTransitionOnGpu(mlt_frame aFrame, mlt_get_image cpuGetImage, const TransitionDescriptor &descriptor, TransitionPlan &plan, const TransitionContext &ctx,
                       bool callerWantsDevice, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    TransitionJob &job = *plan.job;
    Runtime &runtime = Runtime::instance();
    mlt_frame bFrame = mlt_frame_pop_frame(aFrame);
    auto transition = static_cast<mlt_transition>(mlt_frame_pop_service(aFrame));
    mlt_properties aProperties = MLT_FRAME_PROPERTIES(aFrame);
    mlt_properties bProperties = MLT_FRAME_PROPERTIES(bFrame);
    job.beforeFetch(aFrame, bFrame);

    // Top image
    FetchedImage top;
    top.format = job.topFormat;
    top.width = job.topWidth;
    top.height = job.topHeight;
    if (ctx.topOnGpu) {
        mlt_properties_set_int(bProperties, kWantDevice, 1);
    }
    int error = mlt_frame_get_image(bFrame, &top.host, &top.format, &top.width, &top.height, 0);
    mlt_properties_set_int(bProperties, kWantDevice, 0);
    top.device = takeDeviceImage(bProperties, top.host, top.width, top.height, top.format);
    FetchedImage bottom;
    if (error || !top.host) {
        return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
    }

    if (job.passOpaqueTop) {
        // The top image is used as is if it has no transparency
        bool hasAlpha = job.topAlpha;
        if (!hasAlpha && (isRgba(top.format) || mlt_frame_get_alpha(bFrame))) {
            hasAlpha = true;
        }
        if (hasAlpha && isRgba(top.format)) {
            bool opaque = false;
            if (top.device) {
                const CUresult res = deviceImageOpaque(*top.device, opaque);
                if (res != CUDA_SUCCESS) {
                    runtime.handleError(res, descriptor.service);
                    return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
                }
            } else {
                struct mlt_image_s planes;
                mlt_image_set_values(&planes, top.host, top.format, top.width, top.height);
                opaque = mlt_image_is_opaque(&planes);
            }
            hasAlpha = !opaque;
        }
        if (!hasAlpha) {
            if (top.device && callerWantsDevice && top.width == ctx.width && top.height == ctx.height) {
                // Stays on the GPU for the caller
                job.passFrameProperties(aFrame, bFrame);
                *image = top.host;
                *format = mlt_image_rgba;
                *width = top.width;
                *height = top.height;
                attachDeviceImage(aProperties, std::move(top.device));
                logDebug("%s: top image passed through on the GPU", descriptor.service);
                return 0;
            }
            if (!ensureHost(bFrame, top)) {
                logError("%s: lost the GPU image", descriptor.service);
                return 1;
            }
            return job.passThrough(aFrame, bFrame, top.host, top.format, top.width, top.height, image, format, width, height);
        }
        if (!isRgba(top.format) && hasConvertImage(bFrame)) {
            convertImage(bFrame, &top.host, &top.format, mlt_image_rgba);
        }
    }
    if (!isRgba(top.format)) {
        return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
    }

    // Bottom image
    bottom.format = mlt_image_rgba;
    bottom.width = ctx.width;
    bottom.height = ctx.height;
    if (ctx.bottomOnGpu) {
        mlt_properties_set_int(aProperties, kWantDevice, 1);
    }
    error = mlt_frame_get_image(aFrame, &bottom.host, &bottom.format, &bottom.width, &bottom.height, job.bottomWritable);
    mlt_properties_set_int(aProperties, kWantDevice, 0);
    bottom.device = takeDeviceImage(aProperties, bottom.host, bottom.width, bottom.height, bottom.format);
    if (error || !bottom.host) {
        return error ? error : 1;
    }
    if (!isRgba(bottom.format) || !job.prepare(top.width, top.height, bottom.width, bottom.height)) {
        return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
    }

    const size_t topBytes = size_t(top.width) * size_t(top.height) * 4;
    const size_t bytes = size_t(bottom.width) * size_t(bottom.height) * 4;
    const bool topFromDevice = top.device != nullptr;
    const bool bottomFromDevice = bottom.device != nullptr;
    CUresult res = CUDA_ERROR_NOT_INITIALIZED;
    {
        GpuScope scope;
        if (scope.ok()) {
            DeviceBuffer topBuffer;
            DeviceBuffer bottomBuffer;
            DeviceBuffer out;
            res = CUDA_SUCCESS;
            if (top.device) {
                topBuffer = std::move(top.device->buffer);
                top.device.reset();
            } else {
                res = topBuffer.allocate(topBytes);
                if (res == CUDA_SUCCESS) {
                    res = scope.upload(topBuffer.ptr(), top.host, topBytes);
                }
            }
            if (res == CUDA_SUCCESS) {
                if (bottom.device) {
                    bottomBuffer = std::move(bottom.device->buffer);
                    bottom.device.reset();
                } else {
                    res = bottomBuffer.allocate(bytes);
                    if (res == CUDA_SUCCESS) {
                        res = scope.upload(bottomBuffer.ptr(), bottom.host, bytes);
                    }
                }
            }
            if (res == CUDA_SUCCESS) {
                res = out.allocate(bytes);
            }
            if (res == CUDA_SUCCESS) {
                res = job.run(scope, topBuffer.ptr(), bottomBuffer.ptr(), out.ptr());
            }
            if (res == CUDA_SUCCESS && callerWantsDevice) {
                res = scope.synchronize();
                if (res == CUDA_SUCCESS) {
                    auto handoff = std::make_unique<DeviceImage>();
                    handoff->buffer = std::move(out);
                    handoff->width = bottom.width;
                    handoff->height = bottom.height;
                    handoff->tag = bottom.host;
                    attachDeviceImage(aProperties, std::move(handoff));
                    // The CPU transition drops the top image
                    mlt_frame_set_image(bFrame, nullptr, 0, nullptr);
                    *image = bottom.host;
                    *format = mlt_image_rgba;
                    *width = bottom.width;
                    *height = bottom.height;
                    runtime.countFrame(true);
                    logDebug("%s: %dx%d on GPU, kept on GPU", descriptor.service, bottom.width, bottom.height);
                    return 0;
                }
            } else if (res == CUDA_SUCCESS) {
                auto *result = static_cast<uint8_t *>(mlt_pool_alloc(int(bytes)));
                res = result ? scope.download(result, out.ptr(), bytes) : CUDA_ERROR_OUT_OF_MEMORY;
                if (res == CUDA_SUCCESS) {
                    res = scope.synchronize();
                }
                if (res == CUDA_SUCCESS) {
                    mlt_frame_set_image(aFrame, result, int(bytes), mlt_pool_release);
                    mlt_frame_set_image(bFrame, nullptr, 0, nullptr);
                    *image = result;
                    *format = mlt_image_rgba;
                    *width = bottom.width;
                    *height = bottom.height;
                    runtime.countFrame(true);
                    logDebug("%s: %dx%d on GPU", descriptor.service, bottom.width, bottom.height);
                    return 0;
                }
                if (result) {
                    mlt_pool_release(result);
                }
            }
            runtime.handleError(res, descriptor.service);
            // Images that only existed on the GPU go back to their frames for the CPU
            if ((topFromDevice && !restoreToFrame(scope, topBuffer, bFrame, top)) ||
                (bottomFromDevice && !restoreToFrame(scope, bottomBuffer, aFrame, bottom))) {
                logError("%s: lost a GPU image after a CUDA error", descriptor.service);
                return 1;
            }
        }
    }
    logDebug("%s: falling back to the CPU", descriptor.service);
    return transitionCpuFallback(aFrame, transition, bFrame, cpuGetImage, top, bottom, image, format, width, height, writable);
}

int bridgeTransitionGetImage(mlt_frame aFrame, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties aProperties = MLT_FRAME_PROPERTIES(aFrame);
    auto cpuGetImage = reinterpret_cast<mlt_get_image>(mlt_frame_pop_service(aFrame));
    const bool callerWantsDevice = takeFlag(aProperties, kWantDevice);
    // The CPU transition entries: [transition, b_frame]
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(aFrame);
    const int count = mlt_deque_count(stack);
    if (!cpuGetImage || count < 2) {
        return 1;
    }
    auto bFrame = static_cast<mlt_frame>(mlt_deque_peek(stack, count - 1));
    auto transition = static_cast<mlt_transition>(mlt_deque_peek(stack, count - 2));
    mlt_properties properties = MLT_TRANSITION_PROPERTIES(transition);
    auto *info = static_cast<TransitionWrapInfo *>(mlt_properties_get_data(properties, kWrapProperty, nullptr));
    if (!info || !moduleEnabled() || !serviceEnabled(info->descriptor->service) || mlt_properties_get_int(properties, kForceCpu)) {
        return cpuGetImage(aFrame, image, format, width, height, writable);
    }
    const TransitionDescriptor &descriptor = *info->descriptor;

    TransitionContext ctx;
    ctx.transition = transition;
    ctx.aFrame = aFrame;
    ctx.bFrame = bFrame;
    ctx.properties = properties;
    ctx.position = mlt_transition_get_position(transition, aFrame);
    ctx.length = mlt_transition_get_length(transition);
    ctx.format = *format;
    ctx.width = *width;
    ctx.height = *height;
    ctx.topOnGpu = belowTransitionIsBridge(bFrame, 0, transition, aFrame);
    ctx.bottomOnGpu = belowTransitionIsBridge(aFrame, 2, transition, nullptr);
    ctx.callerOnGpu = callerWantsDevice;
    TransitionPlan plan = descriptor.plan(ctx);
    if (plan.decision != Decision::Gpu || !plan.job || !Runtime::instance().ensureInitialized()) {
        if (plan.decision == Decision::Gpu) {
            Runtime::instance().countFrame(false);
        }
        return cpuGetImage(aFrame, image, format, width, height, writable);
    }
    return runTransitionOnGpu(aFrame, cpuGetImage, descriptor, plan, ctx, callerWantsDevice, image, format, width, height, writable);
}

mlt_frame wrappedTransitionProcess(mlt_transition transition, mlt_frame aFrame, mlt_frame bFrame)
{
    auto *info = static_cast<TransitionWrapInfo *>(mlt_properties_get_data(MLT_TRANSITION_PROPERTIES(transition), kWrapProperty, nullptr));
    if (!info) {
        return aFrame;
    }
    mlt_frame result = info->originalProcess(transition, aFrame, bFrame);
    if (result != aFrame || !moduleEnabled()) {
        return result;
    }
    // The CPU transition pushed [transition, b_frame, get_image]: insert the bridge on top
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(aFrame);
    const int count = mlt_deque_count(stack);
    if (count >= 3 && mlt_deque_peek(stack, count - 2) == bFrame && mlt_deque_peek(stack, count - 3) == transition) {
        void *cpuGetImage = mlt_deque_pop_back(stack);
        mlt_frame_push_service(aFrame, cpuGetImage);
        mlt_frame_push_get_image(aFrame, bridgeTransitionGetImage);
    }
    return result;
}

mlt_frame wrappedProcess(mlt_filter filter, mlt_frame frame)
{
    auto *info = static_cast<WrapInfo *>(mlt_properties_get_data(MLT_FILTER_PROPERTIES(filter), kWrapProperty, nullptr));
    if (!info) {
        return frame;
    }
    mlt_frame result = info->originalProcess(filter, frame);
    if (result != frame || !moduleEnabled()) {
        return result;
    }
    // The CPU filter pushed itself and its get_image: insert the bridge on top
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int count = mlt_deque_count(stack);
    if (count >= 2 && mlt_deque_peek(stack, count - 2) == filter) {
        void *cpuGetImage = mlt_deque_pop_back(stack);
        mlt_frame_push_service(frame, cpuGetImage);
        mlt_frame_push_get_image(frame, bridgeGetImage);
    }
    return result;
}

} // namespace

void wrapTransition(mlt_transition transition, const TransitionDescriptor &descriptor)
{
    if (!transition || !transition->process || transition->process == wrappedTransitionProcess) {
        return;
    }
    auto *info = new TransitionWrapInfo{transition->process, &descriptor};
    mlt_properties_set_data(MLT_TRANSITION_PROPERTIES(transition), kWrapProperty, info, 0, destroyTransitionWrapInfo, nullptr);
    transition->process = wrappedTransitionProcess;
}

void wrapFilter(mlt_filter filter, const EffectDescriptor &effect)
{
    if (!filter || !filter->process || filter->process == wrappedProcess) {
        return;
    }
    auto *info = new WrapInfo{filter->process, &effect};
    mlt_properties_set_data(MLT_FILTER_PROPERTIES(filter), kWrapProperty, info, 0, destroyWrapInfo, nullptr);
    filter->process = wrappedProcess;
}

} // namespace gpufx
