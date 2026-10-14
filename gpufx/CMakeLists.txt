# SPDX-FileCopyrightText: 2026 Kdenlive contributors
# SPDX-License-Identifier: BSD-2-Clause

# gpufx: MLT module running some MLT filters on NVIDIA GPUs with CUDA.
#
# Building it needs no CUDA toolkit: the kernels are compiled into PTX that is
# shipped in kernels/gpufx_kernels.ptx, and the NVIDIA driver is loaded at
# runtime. Without a driver the module leaves every filter on the CPU.
#
# After changing kernels/gpufx_kernels.cu, regenerate the PTX with nvcc:
#   cmake --build <builddir> --target gpufx_update_ptx

set(GPUFX_SOURCE_PTX "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gpufx_kernels.ptx")
set(GPUFX_KERNEL_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gpufx_kernels.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gpufx_device.h"
)

find_program(GPUFX_NVCC nvcc HINTS ENV CUDA_PATH PATH_SUFFIXES bin)
option(GPUFX_BUILD_PTX "Compile the CUDA kernels with nvcc instead of using the bundled PTX" OFF)

function(gpufx_nvcc_command output)
    set(raw "${CMAKE_CURRENT_BINARY_DIR}/gpufx_kernels.nvcc.ptx")
    set(${output}
        COMMAND "${GPUFX_NVCC}" -ptx -arch=compute_50 -O3 -o "${raw}" "${CMAKE_CURRENT_SOURCE_DIR}/kernels/gpufx_kernels.cu"
        PARENT_SCOPE)
endfunction()

if(GPUFX_NVCC)
    gpufx_nvcc_command(nvcc_command)
    add_custom_target(gpufx_update_ptx
        ${nvcc_command}
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/gpufx_kernels.nvcc.ptx" "-DOUTPUT=${GPUFX_SOURCE_PTX}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpufxFinalizePtx.cmake"
        DEPENDS ${GPUFX_KERNEL_SOURCES}
        COMMENT "Regenerating gpufx/kernels/gpufx_kernels.ptx with nvcc"
        VERBATIM)
endif()

if(GPUFX_BUILD_PTX)
    if(NOT GPUFX_NVCC)
        message(FATAL_ERROR "GPUFX_BUILD_PTX is set but nvcc was not found")
    endif()
    set(GPUFX_PTX "${CMAKE_CURRENT_BINARY_DIR}/gpufx_kernels.ptx")
    gpufx_nvcc_command(nvcc_command)
    add_custom_command(OUTPUT "${GPUFX_PTX}"
        ${nvcc_command}
        COMMAND "${CMAKE_COMMAND}" "-DINPUT=${CMAKE_CURRENT_BINARY_DIR}/gpufx_kernels.nvcc.ptx" "-DOUTPUT=${GPUFX_PTX}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpufxFinalizePtx.cmake"
        DEPENDS ${GPUFX_KERNEL_SOURCES} "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpufxFinalizePtx.cmake"
        COMMENT "Compiling the gpufx CUDA kernels"
        VERBATIM)
else()
    set(GPUFX_PTX "${GPUFX_SOURCE_PTX}")
endif()

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/gpufx_ptx.cpp"
    COMMAND "${CMAKE_COMMAND}" "-DINPUT=${GPUFX_PTX}" "-DOUTPUT=${CMAKE_CURRENT_BINARY_DIR}/gpufx_ptx.cpp"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpufxEmbedPtx.cmake"
    DEPENDS "${GPUFX_PTX}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/GpufxEmbedPtx.cmake"
    COMMENT "Embedding the gpufx kernels"
    VERBATIM)

add_library(mltgpufx MODULE
    src/bridge.cpp
    src/cuda_driver.cpp
    src/effect_blur.cpp
    src/effect_color.cpp
    src/effect_dropshadow.cpp
    src/effect_frei0r.cpp
    src/effect_qtblend.cpp
    src/gpu_runtime.cpp
    src/module.cpp
    src/qt_draw.cpp
    src/qt_raster.cpp
    "${CMAKE_CURRENT_BINARY_DIR}/gpufx_ptx.cpp"
)
target_link_libraries(mltgpufx PRIVATE Mlt7::mlt ${CMAKE_DL_LIBS})
set_target_properties(mltgpufx PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
)
if(NOT MSVC)
    # The kernels' emulation and Qt compatibility rely on exact float math
    target_compile_options(mltgpufx PRIVATE -ffp-contract=off)
endif()

# MLT only loads modules from its own module directory.
set(GPUFX_MLT_MODULE_DIR "" CACHE PATH "Where to install the gpufx MLT module (defaults to MLT's module directory)")
set(gpufx_install_dir "${GPUFX_MLT_MODULE_DIR}")
if(NOT gpufx_install_dir)
    if(WIN32)
        set(gpufx_install_dir "${KDE_INSTALL_LIBDIR}/mlt")
    else()
        find_package(PkgConfig QUIET)
        if(PKG_CONFIG_FOUND)
            pkg_get_variable(gpufx_mlt_moduledir mlt-framework-7 moduledir)
        endif()
        get_filename_component(gpufx_prefix "${CMAKE_INSTALL_PREFIX}" ABSOLUTE)
        if(gpufx_mlt_moduledir AND gpufx_mlt_moduledir MATCHES "^${gpufx_prefix}/")
            set(gpufx_install_dir "${gpufx_mlt_moduledir}")
        else()
            set(gpufx_install_dir "${KDE_INSTALL_LIBDIR}/mlt-7")
            if(gpufx_mlt_moduledir)
                message(STATUS "gpufx: MLT's modules are in ${gpufx_mlt_moduledir}, outside of the install prefix. "
                               "The module goes to ${CMAKE_INSTALL_PREFIX}/${gpufx_install_dir}: Kdenlive loads it from there, "
                               "melt only does if it is copied to ${gpufx_mlt_moduledir} (or set GPUFX_MLT_MODULE_DIR).")
            endif()
        endif()
    endif()
endif()
set(KDENLIVE_GPUFX_MODULE_DIR "${gpufx_install_dir}" PARENT_SCOPE)
install(TARGETS mltgpufx
    LIBRARY DESTINATION "${gpufx_install_dir}"
    RUNTIME DESTINATION "${gpufx_install_dir}")

if(BUILD_TESTING)
    add_subdirectory(tests)
endif()
