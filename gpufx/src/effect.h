/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "gpu_runtime.h"

#include <framework/mlt.h>

#include <memory>
#include <vector>

namespace gpufx {

/** What the effect is asked to do for one frame, read before the image is fetched. */
struct EffectContext
{
    mlt_filter filter = nullptr;
    mlt_frame frame = nullptr;
    mlt_properties properties = nullptr; // the filter's properties
    mlt_position position = 0;
    mlt_position length = 0;
    // The format and size requested by the caller of get_image
    mlt_image_format format = mlt_image_none;
    int width = 0;
    int height = 0;
};

/** The GPU work for one frame, with the parameters already decoded. */
class EffectJob
{
public:
    virtual ~EffectJob() = default;
    /** Processes width x height RGBA pixels from in to out (device memory,
     *  4 bytes per pixel, not overlapping) on the scope's stream. */
    virtual CUresult run(GpuScope &scope, int width, int height, CUdeviceptr in, CUdeviceptr out) = 0;
    /** Called right before the input is fetched, for the frame changes the
     *  CPU filter makes at that point (they can matter to the filters below). */
    virtual void beforeFetch(mlt_frame frame) { (void)frame; }
    /** Called once the input is fetched, with its size. Returning false
     *  leaves the frame to the CPU filter. */
    virtual bool prepare(int inputWidth, int inputHeight)
    {
        (void)inputWidth;
        (void)inputHeight;
        return true;
    }
    /** For effects whose output size differs from the input size (see
     *  EffectPlan::outputWidth). */
    virtual CUresult runResized(GpuScope &scope, int inWidth, int inHeight, CUdeviceptr in, int outWidth, int outHeight, CUdeviceptr out)
    {
        (void)outWidth;
        (void)outHeight;
        return run(scope, inWidth, inHeight, in, out);
    }
    /** Undoes the changes of beforeFetch(), before the CPU filter processes
     *  the frame after all. */
    virtual void restoreFrame(mlt_frame frame) { (void)frame; }
};

enum class Decision {
    Cpu,        // let the original CPU filter process the frame
    Gpu,        // run the job on the GPU
    PassThrough // the filter does nothing for these parameters
};

struct EffectPlan
{
    Decision decision = Decision::Cpu;
    std::unique_ptr<EffectJob> job;
    /** For effects that do not keep the image size: the size of the input to
     *  request, and the size of the output. 0 keeps the requested size. */
    int inputWidth = 0;
    int inputHeight = 0;
    int outputWidth = 0;
    int outputHeight = 0;

    static EffectPlan cpu() { return {}; }
    static EffectPlan passThrough()
    {
        EffectPlan plan;
        plan.decision = Decision::PassThrough;
        return plan;
    }
    static EffectPlan gpu(std::unique_ptr<EffectJob> job)
    {
        EffectPlan plan;
        plan.decision = Decision::Gpu;
        plan.job = std::move(job);
        return plan;
    }
};

struct EffectDescriptor
{
    /** MLT service name of the CPU filter that gets accelerated */
    const char *service;
    /** Cheap per pixel effects: the CPU version is already fast, so they only
     *  run on the GPU when the frame is on the GPU anyway, next to another
     *  accelerated effect. */
    bool lightweight;
    /** Reads the parameters for this frame (with the filter locked). Must not
     *  fetch the image nor change the frame (see EffectJob::beforeFetch()).
     *  Only rgba output is supported on the GPU, the plan
     *  must return Decision::Cpu when the CPU filter would produce anything
     *  else for ctx.format. */
    EffectPlan (*plan)(const EffectContext &ctx);
};

/** All the effects with a GPU implementation */
const std::vector<const EffectDescriptor *> &effectRegistry();
const EffectDescriptor *findEffect(const char *service);

// Implemented in the effect_*.cpp files
const EffectDescriptor &dropShadowEffect();
const EffectDescriptor &boxBlurEffect();
const EffectDescriptor &rectBlurEffect();
const EffectDescriptor &brightnessEffect();
const EffectDescriptor &liftGammaGainEffect();
const EffectDescriptor &frei0rSquareBlurEffect();
const EffectDescriptor &frei0rGlowEffect();
const EffectDescriptor &frei0rSoftGlowEffect();
const EffectDescriptor &frei0rSharpnessEffect();
const EffectDescriptor &transformEffect();

} // namespace gpufx
