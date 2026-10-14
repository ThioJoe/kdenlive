<!--
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
-->

# gpufx: GPU (CUDA) acceleration of effects

`gpufx` is an MLT module that runs some existing MLT filters on NVIDIA GPUs
with CUDA. It does not add new effects: projects keep using the regular
effects and parameters, and render the same with or without the module, with
or without an NVIDIA GPU.

The module is used by every process that loads MLT: Kdenlive itself (monitors,
timeline preview) and `melt` when rendering, as long as the module is
installed in MLT's module directory.

## Accelerated effects

| Kdenlive effect                     | MLT service         | Notes |
|-------------------------------------|---------------------|-------|
| Drop Shadow                         | `dropshadow`        | Matches the CPU filter (Qt's `QGraphicsDropShadowEffect`): same alpha, colors within one level, 8-bit only |
| BoxBlur                             | `box_blur`          | Identical output, 8-bit only |
| Box Blur                            | `boxblur`           | Identical output |
| Intensity (Brightness), Fade in/out | `brightness`        | Lightweight, see below |
| Lift/gamma/gain                     | `lift_gamma_gain`   | Lightweight, see below |
| Square Blur                         | `frei0r.squareblur` | Identical output |
| Glow                                | `frei0r.glow`       | Identical output |
| Soft Glow                           | `frei0r.softglow`   | Identical output |
| Sharpen                             | `frei0r.sharpness`  | Identical output |

Any mix of accelerated and CPU-only effects works: each accelerated effect is
moved to the GPU for its own processing, and the others stay on the CPU.

When several accelerated effects follow each other in an effect stack, the
frame stays on the GPU between them: it is uploaded once before the first one
and downloaded once after the last one.

*Lightweight* effects are cheap on the CPU, so on their own copying the frame
to the GPU and back would cost more than it saves. They only run on the GPU
when the frame is there anyway for a neighbouring accelerated effect (for
example a fade applied next to a drop shadow).

Frames in other formats than 8-bit RGBA (for example 16-bit `rgba64`
processing), unusual parameter values, and any CUDA error fall back to the
original CPU filter for that frame. frei0r effects always work in 8-bit RGBA;
they stay on the CPU when their MLT `threads` property splits the frame in
slices. An error that leaves the GPU in an unknown state turns GPU processing
off for the rest of the session.

## Requirements

* An NVIDIA GPU of the Maxwell generation or newer (compute capability 5.0)
* The NVIDIA driver. Nothing else: the CUDA toolkit is not needed to build or
  to run the module. The driver compiles the kernels for the installed GPU
  the first time they are used and caches the result.

Without an NVIDIA driver the module does nothing and every effect runs on the
CPU as usual.

## Settings

In Kdenlive: *Settings > Configure Kdenlive > Playback > GPU effects (CUDA)*.
The status line below the option shows the detected GPU. The setting applies
immediately and is passed to the render processes.

Outside Kdenlive the module is controlled with environment variables:

| Variable              | Effect |
|-----------------------|--------|
| `MLT_GPUFX=0`         | Do not touch any filter |
| `MLT_GPUFX_DISABLE`   | Comma separated MLT services to keep on the CPU, e.g. `dropshadow,box_blur` |
| `MLT_GPUFX_DEBUG=1`   | Log every frame processed on the GPU (MLT info log level) |
| `GPUFX_CUDA_DEVICE=n` | Use the n-th CUDA device (default 0) |
| `GPUFX_POOL_MB=n`     | Size of the GPU memory kept for reuse (default: 1/8 of the GPU memory, at most 1 GiB) |

Applications linking MLT can also set the MLT global property
`gpufx.enabled` to `0` or `1` at any time. Creating the `gpufx` filter
service returns a filter whose properties describe the state of the module
(`available`, `status`, `device`, `effects`, `gpu_frames`, ...).

## Building

The module is built with Kdenlive when the CMake option `BUILD_GPUFX` is on
(the default except on macOS). It only depends on MLT.

It is installed in MLT's module directory (from the `mlt-framework-7`
pkg-config file) when that directory is inside the install prefix, otherwise
in `<prefix>/<libdir>/mlt-7`. Kdenlive loads the module from there if MLT did
not, but `melt` (used for the final render) only loads modules from its own
directory. Set `GPUFX_MLT_MODULE_DIR` to choose the location.

The CUDA kernels in `kernels/gpufx_kernels.cu` are shipped compiled to PTX in
`kernels/gpufx_kernels.ptx`, so that no CUDA toolkit is needed. After changing
the kernels, regenerate the PTX with nvcc:

```
cmake --build <builddir> --target gpufx_update_ptx
```

or configure with `-DGPUFX_BUILD_PTX=ON` to compile the kernels during the
build.

## Tests

`tests/gpufx_test` renders every accelerated effect, alone and in chains, once
with the original CPU filter and once through the GPU path, and compares the
pixels. It also checks the number of transfers between memory and GPU, the
fallbacks on errors, and the behaviour without a GPU.

By default the tests run against `mockcuda`, an emulated CUDA driver built
with the tests: it executes the same kernel source compiled as C++ on the CPU,
so the tests run on machines without an NVIDIA GPU. To run the same
comparisons on a real GPU, point the tests at the driver:

```
GPUFX_CUDA_LIBRARY=libcuda.so.1 ./gpufx_test      # Linux
set GPUFX_CUDA_LIBRARY=nvcuda.dll & gpufx_test.exe  # Windows
```

## How it works

* `src/module.cpp` registers the module. It listens to MLT's
  `filter-create-done` factory event and wraps the filters it knows.
* `src/bridge.cpp` replaces the wrapped filter's `process()`: the original
  function still runs and pushes the CPU `get_image` on the frame, and the
  bridge slides its own `get_image` on top of it. For each frame the bridge
  reads the effect parameters, then either calls the CPU `get_image`
  unchanged or fetches the input image, runs the effect on the GPU and
  returns the result. Between two accelerated effects the image is passed on
  the GPU through frame properties instead of being downloaded.
* `src/gpu_runtime.cpp` loads the CUDA driver at runtime, creates the
  context, loads the PTX and keeps pools of GPU memory and streams.
* `src/effect_*.cpp` read the parameters of each effect, exactly like the CPU
  filter does, and launch the kernels.
* `kernels/gpufx_kernels.cu` contains the kernels. They reproduce the integer
  arithmetic of the CPU code (for the drop shadow, Qt's blur, scaling and
  compositing), which is why the output matches.

## Adding an effect

1. Write the kernels in `kernels/gpufx_kernels.cu` (no shared memory, no
   `pow`/`exp`: precompute on the host) and list them in
   `kernels/gpufx_kernel_list.h`. Regenerate the PTX.
2. Add an `EffectDescriptor` in a `src/effect_*.cpp` file: its `plan()`
   function reads the parameters for the frame the same way the CPU filter
   does and returns a job, or `Decision::Cpu` for anything it does not
   handle identically. Register it in `effectRegistry()` in `src/module.cpp`.
3. Add comparisons with the CPU filter to `tests/test_gpufx.cpp`.
