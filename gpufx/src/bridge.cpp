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

/** True if the get_image that will run after the entry at depth (0 = top) is
 *  another bridge, i.e. the next filter down the stack runs through gpufx. */
bool nextIsBridge(mlt_frame frame, int depth)
{
    mlt_deque stack = MLT_FRAME_IMAGE_STACK(frame);
    const int count = mlt_deque_count(stack);
    return count > depth && mlt_deque_peek(stack, count - 1 - depth) == reinterpret_cast<void *>(&bridgeGetImage);
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

int runOnGpu(mlt_frame frame, mlt_filter filter, mlt_get_image cpuGetImage, const EffectDescriptor &effect, EffectJob &job, bool callerWantsDevice,
             bool lowerOnGpu, uint8_t **image, mlt_image_format *format, int *width, int *height, int writable)
{
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    Runtime &runtime = Runtime::instance();

    // Our filter entry, the stack now starts with the image source
    mlt_frame_pop_service(frame);

    uint8_t *input = nullptr;
    mlt_image_format inputFormat = mlt_image_rgba;
    int w = *width;
    int h = *height;
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
    if (inputFormat != mlt_image_rgba || w <= 0 || h <= 0) {
        return cpuFallback(frame, filter, cpuGetImage, input, {nullptr, mlt_pool_release}, w, h, image, format, width, height, writable);
    }

    const size_t bytes = size_t(w) * size_t(h) * 4;
    const bool inputOnDevice = deviceInput != nullptr;
    std::unique_ptr<uint8_t, void (*)(void *)> rescued(nullptr, mlt_pool_release);
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
                res = out.allocate(bytes);
            }
            if (res == CUDA_SUCCESS) {
                res = job.run(scope, w, h, in.ptr(), out.ptr());
            }
            if (res == CUDA_SUCCESS) {
                if (callerWantsDevice) {
                    // The caller continues on the GPU: keep the result there
                    res = scope.synchronize();
                    if (res == CUDA_SUCCESS) {
                        auto handoff = std::make_unique<DeviceImage>();
                        handoff->buffer = std::move(out);
                        handoff->width = w;
                        handoff->height = h;
                        handoff->tag = input;
                        attachDeviceImage(properties, std::move(handoff));
                        *image = input;
                        *format = mlt_image_rgba;
                        *width = w;
                        *height = h;
                        runtime.countFrame(true);
                        logDebug("%s: %dx%d on GPU, kept on GPU", effect.service, w, h);
                        return 0;
                    }
                } else {
                    auto *result = static_cast<uint8_t *>(mlt_pool_alloc(int(bytes)));
                    res = result ? scope.download(result, out.ptr(), bytes) : CUDA_ERROR_OUT_OF_MEMORY;
                    if (res == CUDA_SUCCESS) {
                        res = scope.synchronize();
                    }
                    if (res == CUDA_SUCCESS) {
                        mlt_frame_set_image(frame, result, int(bytes), mlt_pool_release);
                        *image = result;
                        *format = mlt_image_rgba;
                        *width = w;
                        *height = h;
                        runtime.countFrame(true);
                        logDebug("%s: %dx%d on GPU", effect.service, w, h);
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
                rescued.reset(static_cast<uint8_t *>(mlt_pool_alloc(int(bytes))));
                CUresult copied = rescued ? scope.download(rescued.get(), in.ptr(), bytes) : CUDA_ERROR_OUT_OF_MEMORY;
                if (copied == CUDA_SUCCESS) {
                    copied = scope.synchronize();
                }
                if (copied != CUDA_SUCCESS) {
                    rescued.reset();
                }
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
    return runOnGpu(frame, filter, cpuGetImage, effect, *plan.job, callerWantsDevice, lowerOnGpu, image, format, width, height, writable);
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
