/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * Checks the gpufx module against the CPU filters it accelerates.
 *
 * Every test renders the same input twice, once with the module turned off
 * for the filters (the original CPU code) and once through the GPU path, and
 * compares the pixels. By default the GPU is the emulated driver built with
 * the tests (mockcuda), which also lets the tests count transfers between
 * host and device and inject failures. Pointing GPUFX_CUDA_LIBRARY at the real
 * driver (libcuda.so.1 / nvcuda.dll) runs the same comparisons on an actual
 * NVIDIA GPU; the checks that need the emulator are then skipped.
 *
 * Usage: gpufx_test [--no-gpu]
 *   --no-gpu: expect CUDA to be unavailable (run with MOCKCUDA_NO_DEVICE=1)
 */

#include "mockcuda/mockcuda.h"

#include <framework/mlt.h>

#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

namespace {

int g_failures = 0;
int g_checks = 0;

void check(bool condition, const char *format, ...)
{
    ++g_checks;
    if (condition) {
        return;
    }
    ++g_failures;
    va_list args;
    va_start(args, format);
    fprintf(stderr, "FAIL: ");
    vfprintf(stderr, format, args);
    fprintf(stderr, "\n");
    va_end(args);
}

void *openLibrary(const char *path)
{
#if defined(_WIN32)
    return reinterpret_cast<void *>(LoadLibraryA(path));
#else
    return dlopen(path, RTLD_NOW | RTLD_GLOBAL);
#endif
}

void *findSymbol(void *library, const char *name)
{
#if defined(_WIN32)
    return reinterpret_cast<void *>(GetProcAddress(reinterpret_cast<HMODULE>(library), name));
#else
    return dlsym(library, name);
#endif
}

struct Mock
{
    MockCudaGetStats getStats = nullptr;
    MockCudaResetStats resetStats = nullptr;
    MockCudaFailAfter failAfter = nullptr;
    bool active() const { return getStats != nullptr; }
    MockCudaStats stats() const
    {
        MockCudaStats s{};
        if (getStats) {
            getStats(&s);
        }
        return s;
    }
    void reset() const
    {
        if (resetStats) {
            resetStats();
        }
    }
} g_mock;

struct Image
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> pixels;
};

Image makeImage(int width, int height, int pattern)
{
    Image image;
    image.width = width;
    image.height = height;
    image.pixels.resize(size_t(width) * size_t(height) * 4);
    uint32_t seed = 12345u + uint32_t(pattern) * 7919u;
    auto random = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return seed >> 8;
    };
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            uint8_t *p = &image.pixels[(size_t(y) * size_t(width) + size_t(x)) * 4];
            p[0] = uint8_t(x * 7 + y * 3);
            p[1] = uint8_t(x * 2 + y * 5);
            p[2] = uint8_t(x ^ y);
            int alpha = 255;
            switch (pattern) {
            case 0: { // opaque rectangle with a semi transparent border on a transparent background
                const bool inside = x > width / 4 && x < 3 * width / 4 && y > height / 4 && y < 3 * height / 4;
                alpha = inside ? 255 : 0;
                if (x == width / 4 + 1 || y == height / 4 + 1) {
                    alpha = 128;
                }
                break;
            }
            case 1: // random alpha, including many 0 and 255
                alpha = int(random() & 255);
                if ((random() & 3) == 0) {
                    alpha = 0;
                } else if ((random() & 3) == 0) {
                    alpha = 255;
                }
                p[0] = uint8_t(random());
                p[1] = uint8_t(random());
                p[2] = uint8_t(random());
                break;
            case 2: { // soft disc
                const double dx = x - width * 0.4;
                const double dy = y - height * 0.6;
                const double d = std::sqrt(dx * dx + dy * dy);
                alpha = std::clamp(int(255 - (d - height * 0.2) * 10), 0, 255);
                break;
            }
            case 3: // stripes touching the borders
                alpha = (x < 3 || y < 2 || x > width - 4) ? 200 : (((x / 7 + y / 5) % 3 == 0) ? 255 : 0);
                break;
            default: // opaque
                alpha = 255;
                break;
            }
            p[3] = uint8_t(alpha);
        }
    }
    return image;
}

struct FilterSpec
{
    const char *service;
    std::vector<std::pair<const char *, std::string>> properties;
};

bool hasFilter(const char *service)
{
    return mlt_properties_get_data(mlt_repository_filters(mlt_factory_repository()), service, nullptr) != nullptr;
}

bool hasFilters(const std::vector<FilterSpec> &specs)
{
    for (const FilterSpec &spec : specs) {
        if (!hasFilter(spec.service)) {
            printf("SKIP: MLT has no %s filter\n", spec.service);
            return false;
        }
    }
    return true;
}

std::vector<mlt_filter> createFilters(mlt_profile profile, const std::vector<FilterSpec> &specs, bool cpu)
{
    std::vector<mlt_filter> filters;
    for (const FilterSpec &spec : specs) {
        mlt_filter filter = mlt_factory_filter(profile, spec.service, nullptr);
        if (!filter) {
            fprintf(stderr, "cannot create filter %s\n", spec.service);
            exit(2);
        }
        for (const auto &property : spec.properties) {
            mlt_properties_set(MLT_FILTER_PROPERTIES(filter), property.first, property.second.c_str());
        }
        mlt_properties_set_int(MLT_FILTER_PROPERTIES(filter), "_gpufx.disable", cpu ? 1 : 0);
        filters.push_back(filter);
    }
    return filters;
}

typedef std::vector<std::pair<const char *, std::string>> Properties;

/** Runs the filters (in attach order) over a frame holding the image.
 *  frameProperties are set on the frame, and the output is requested at
 *  requestWidth x requestHeight (default: the input size). */
Image render(mlt_profile profile, const std::vector<FilterSpec> &specs, const Image &input, bool cpu, mlt_position position = 0,
             const Properties &frameProperties = {}, int requestWidth = 0, int requestHeight = 0)
{
    std::vector<mlt_filter> filters = createFilters(profile, specs, cpu);
    mlt_frame frame = mlt_frame_init(nullptr);
    mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
    const int size = input.width * input.height * 4;
    auto *buffer = static_cast<uint8_t *>(mlt_pool_alloc(size));
    memcpy(buffer, input.pixels.data(), size_t(size));
    mlt_frame_set_image(frame, buffer, size, mlt_pool_release);
    mlt_properties_set_int(properties, "format", mlt_image_rgba);
    mlt_properties_set_int(properties, "width", input.width);
    mlt_properties_set_int(properties, "height", input.height);
    mlt_properties_set_int(properties, "meta.media.width", input.width);
    mlt_properties_set_int(properties, "meta.media.height", input.height);
    for (const auto &property : frameProperties) {
        mlt_properties_set(properties, property.first, property.second.c_str());
    }
    mlt_frame_set_position(frame, position);
    for (mlt_filter filter : filters) {
        mlt_filter_process(filter, frame);
    }
    uint8_t *image = nullptr;
    mlt_image_format format = mlt_image_rgba;
    int width = requestWidth > 0 ? requestWidth : input.width;
    int height = requestHeight > 0 ? requestHeight : input.height;
    const int error = mlt_frame_get_image(frame, &image, &format, &width, &height, 0);
    Image output;
    if (!error && image && format == mlt_image_rgba) {
        output.width = width;
        output.height = height;
        output.pixels.assign(image, image + size_t(width) * size_t(height) * 4);
    } else {
        fprintf(stderr, "get_image failed: error %d format %d\n", error, int(format));
    }
    mlt_frame_close(frame);
    for (mlt_filter filter : filters) {
        mlt_filter_close(filter);
    }
    return output;
}

struct Difference
{
    int maxChannel = 0; // largest difference on any channel
    int maxAlpha = 0;
    double maxPremultiplied = 0; // largest difference of premultiplied color
    long pixels = 0;             // pixels with any difference
    bool sizeMismatch = false;
};

Difference compare(const Image &a, const Image &b)
{
    Difference d;
    if (a.width != b.width || a.height != b.height || a.pixels.size() != b.pixels.size() || a.pixels.empty()) {
        d.sizeMismatch = true;
        return d;
    }
    for (size_t i = 0; i < a.pixels.size(); i += 4) {
        bool differs = false;
        for (int c = 0; c < 4; ++c) {
            const int diff = std::abs(int(a.pixels[i + c]) - int(b.pixels[i + c]));
            if (diff) {
                differs = true;
            }
            if (c == 3) {
                d.maxAlpha = std::max(d.maxAlpha, diff);
            } else {
                d.maxChannel = std::max(d.maxChannel, diff);
                const double pa = a.pixels[i + c] * a.pixels[i + 3] / 255.0;
                const double pb = b.pixels[i + c] * b.pixels[i + 3] / 255.0;
                d.maxPremultiplied = std::max(d.maxPremultiplied, std::fabs(pa - pb));
            }
        }
        if (differs) {
            ++d.pixels;
        }
    }
    return d;
}

std::string describe(const std::vector<FilterSpec> &specs)
{
    std::string text;
    for (const FilterSpec &spec : specs) {
        if (!text.empty()) {
            text += " -> ";
        }
        text += spec.service;
        for (const auto &property : spec.properties) {
            text += std::string(" ") + property.first + "=" + property.second;
        }
    }
    return text;
}

/** Renders on CPU and GPU and checks the results. exact: identical pixels,
 *  otherwise colors may differ by 1 (Qt un-premultiplies with an approximate
 *  reciprocal) and alpha must be identical. */
void expectSame(mlt_profile profile, const std::vector<FilterSpec> &specs, const Image &input, bool exact, mlt_position position = 0,
                const Properties &frameProperties = {}, int requestWidth = 0, int requestHeight = 0)
{
    if (!hasFilters(specs)) {
        return;
    }
    const Image cpu = render(profile, specs, input, true, position, frameProperties, requestWidth, requestHeight);
    const Image gpu = render(profile, specs, input, false, position, frameProperties, requestWidth, requestHeight);
    const Difference d = compare(cpu, gpu);
    std::string what = describe(specs) + " on " + std::to_string(input.width) + "x" + std::to_string(input.height);
    for (const auto &property : frameProperties) {
        what += std::string(", ") + property.first + "=" + property.second;
    }
    if (d.sizeMismatch) {
        check(false, "%s: output size mismatch", what.c_str());
        return;
    }
    if (exact) {
        check(d.pixels == 0, "%s: %ld pixels differ (max channel %d, alpha %d)", what.c_str(), d.pixels, d.maxChannel, d.maxAlpha);
    } else {
        check(d.maxAlpha == 0 && d.maxPremultiplied <= 1.0, "%s: alpha diff %d, premultiplied diff %.2f, channel diff %d (%ld pixels)", what.c_str(),
              d.maxAlpha, d.maxPremultiplied, d.maxChannel, d.pixels);
    }
}

void testStatus(mlt_profile profile, bool expectGpu)
{
    mlt_filter status = mlt_factory_filter(profile, "gpufx", nullptr);
    check(status != nullptr, "the gpufx status service is missing");
    if (!status) {
        return;
    }
    mlt_properties properties = MLT_FILTER_PROPERTIES(status);
    const int available = mlt_properties_get_int(properties, "available");
    printf("gpufx status: available=%d device='%s' status='%s' effects='%s'\n", available, mlt_properties_get(properties, "device"),
           mlt_properties_get(properties, "status"), mlt_properties_get(properties, "effects"));
    check(available == (expectGpu ? 1 : 0), "status: available is %d", available);
    const char *effects = mlt_properties_get(properties, "effects");
    check(effects && strstr(effects, "dropshadow"), "status: dropshadow is not listed");
    mlt_filter_close(status);
}

void testDropShadow(mlt_profile profile)
{
    struct Case
    {
        int width, height, pattern;
        const char *radius, *x, *y, *color;
    } cases[] = {
        {64, 48, 0, "1", "8", "8", "#b4636363"},        {64, 48, 0, "0", "0", "0", "#b4636363"},         {64, 48, 1, "0", "0", "0", "#b4636363"},
        {64, 48, 0, "3.9", "2", "-3", "#ff000000"},     {64, 48, 0, "4", "8", "8", "#80ff0000"},         {65, 47, 2, "10", "-5.5", "2.5", "#b4636363"},
        {65, 47, 1, "10", "8", "8", "#b4636363"},       {200, 120, 2, "25", "8.4", "-7.6", "#ff123456"}, {200, 120, 3, "7.5", "0", "0", "#c0000000"},
        {31, 17, 3, "100", "100", "-100", "#b4636363"}, {64, 48, 1, "0.5", "0.5", "-0.5", "#b4636363"},  {64, 48, 1, "2", "2.5", "-2.5", "#01ffffff"},
        {64, 48, 4, "6", "5", "5", "#b4636363"},        {1, 1, 0, "5", "1", "1", "#ff00ff00"},           {2, 3, 1, "4", "-1", "1", "#ff00ff00"},
        {48, 64, 3, "0", "3", "0", "#40ffffff"},        {48, 64, 1, "33", "-60", "70", "#ffffffff"},
    };
    for (const Case &c : cases) {
        const Image input = makeImage(c.width, c.height, c.pattern);
        expectSame(profile, {{"dropshadow", {{"radius", c.radius}, {"x", c.x}, {"y", c.y}, {"color", c.color}}}}, input, false);
    }
    // Animated parameters are evaluated at the frame position
    const Image input = makeImage(80, 60, 2);
    for (mlt_position position : {0, 5, 10}) {
        expectSame(profile, {{"dropshadow", {{"radius", "0=1;10=20"}, {"x", "0=-10;10=10"}, {"y", "0=3;10=-3"}, {"out", "10"}}}}, input, false, position);
    }
}

void testBoxBlur(mlt_profile profile)
{
    const Image input = makeImage(160, 90, 1);
    for (const char *h : {"0", "10", "50", "3", "200"}) {
        for (const char *v : {"0", "20", "7"}) {
            for (const char *preserve : {"0", "1"}) {
                expectSame(profile, {{"box_blur", {{"hradius", h}, {"vradius", v}, {"preserve_alpha", preserve}}}}, input, true);
            }
        }
    }
    expectSame(profile, {{"box_blur", {{"hradius", "33"}, {"vradius", "21"}}}}, makeImage(97, 53, 2), true);
}

void testRectBlur(mlt_profile profile)
{
    const Image input = makeImage(160, 90, 1);
    for (const char *blur : {"0", "1", "5", "40"}) {
        for (const char *hori : {"1", "3"}) {
            expectSame(profile, {{"boxblur", {{"blur", blur}, {"hori", hori}, {"vert", "2"}}}}, input, true);
        }
    }
    // Old style start parameter
    expectSame(profile, {{"boxblur", {{"start", "4"}}}}, makeImage(77, 41, 3), true);
}

void testColor(mlt_profile profile)
{
    const Image input = makeImage(96, 54, 1);
    // Lightweight effects next to a heavy one run on the GPU
    for (const char *level : {"0.5", "1", "1.7", "0"}) {
        for (const char *alpha : {"1", "0.3", "-1"}) {
            expectSame(profile, {{"brightness", {{"level", level}, {"alpha", alpha}}}, {"box_blur", {{"hradius", "20"}, {"vradius", "20"}}}}, input, true);
        }
    }
    expectSame(profile, {{"box_blur", {{"hradius", "20"}, {"vradius", "0"}}}, {"brightness", {{"start", "0.2"}, {"end", "1.4"}, {"out", "20"}}}}, input, true,
               7);
    expectSame(profile,
               {{"lift_gamma_gain",
                 {{"lift_r", "0.1"},
                  {"lift_g", "0"},
                  {"lift_b", "0.2"},
                  {"gamma_r", "1.2"},
                  {"gamma_g", "0.8"},
                  {"gamma_b", "1"},
                  {"gain_r", "1.1"},
                  {"gain_g", "0.9"},
                  {"gain_b", "1.3"}}},
                {"boxblur", {{"blur", "3"}}}},
               input, true);
}

void testFrei0r(mlt_profile profile)
{
    const Image input = makeImage(160, 90, 1);
    const Image portrait = makeImage(53, 97, 2);
    // Square Blur: the kernel size is relative to the largest side
    for (const char *kernel : {"0", "0.004", "0.05", "0.3", "1", "3"}) {
        expectSame(profile, {{"frei0r.squareblur", {{"Kernel size", kernel}}}}, input, true);
    }
    expectSame(profile, {{"frei0r.squareblur", {{"0", "0.2"}}}}, portrait, true);
    for (const char *blur : {"0", "0.01", "0.2", "1", "10"}) {
        expectSame(profile, {{"frei0r.glow", {{"Blur", blur}}}}, input, true);
    }
    expectSame(profile, {{"frei0r.glow", {{"Blur", "2"}}}}, portrait, true);
    for (const char *blend : {"0", "0.5", "1"}) {
        for (const char *blur : {"0", "0.1", "0.5"}) {
            expectSame(profile, {{"frei0r.softglow", {{"blurblend", blend}, {"blur", blur}, {"brightness", "0.6"}, {"sharpness", "0.7"}}}}, input, true);
        }
    }
    // Plugin defaults
    expectSame(profile, {{"frei0r.softglow", {}}}, portrait, true);
    for (const char *amount : {"0", "0.3", "0.5", "1"}) {
        for (const char *size : {"0", "0.3", "0.5", "1"}) {
            expectSame(profile, {{"frei0r.sharpness", {{"Amount", amount}, {"Size", size}}}}, input, true);
        }
    }
    expectSame(profile, {{"frei0r.sharpness", {{"0", "0.9"}, {"1", "0.2"}}}}, makeImage(31, 7, 3), true);
    // Animated parameters, mixed with other effects
    expectSame(profile,
               {{"frei0r.glow", {{"Blur", "0=0.1;10=0.9"}, {"out", "10"}}},
                {"dropshadow", {{"radius", "4"}}},
                {"frei0r.sharpness", {{"Amount", "0.7"}, {"Size", "0=0;10=1"}, {"out", "10"}}},
                {"frei0r.softglow", {{"blur", "0.05"}}}},
               input, false, 5);
    if (g_mock.active() && hasFilter("frei0r.squareblur")) {
        // MLT slices the frame when "threads" is set: left to the CPU. The
        // slices share one plugin instance, so the CPU result is not even
        // repeatable for this plugin, nothing to compare.
        g_mock.reset();
        render(profile, {{"frei0r.squareblur", {{"Kernel size", "0.1"}, {"threads", "0"}}}}, input, false);
        check(mlt_slices_count_normal() == 1 || g_mock.stats().launches == 0, "a sliced frei0r filter ran on the GPU");
    }
}

/** Kdenlive's Transform: MLT's qtblend filter, drawn by QPainter on the CPU */
void testTransform(mlt_profile profile)
{
    if (!hasFilter("qtblend")) {
        printf("SKIP: MLT has no qtblend filter\n");
        return;
    }
    const Properties smooth = {{"consumer.rescale", "bilinear"}};
    const Properties nearest = {{"consumer.rescale", "nearest"}};
    struct Case
    {
        int width, height, pattern;
        const char *rect, *rotation;
        bool distort;
    } cases[] = {
        {160, 90, 1, "20 10 80 45 1", nullptr, false},         // picture in picture
        {160, 90, 2, "-30.5 12.25 200 120 1", nullptr, false}, // zoom from a fractional position
        {160, 90, 1, "10 5 160 90 0.6", nullptr, false},       // moved only, with opacity
        {120, 70, 3, "15 8 100 60 1", "25", false},            // rotation
        {160, 90, 0, "0 0 160 90 1", "-90", false},            // quarter turn
        {97, 61, 4, "5.5 3.5 120 70 0.8", "180", false},       // half turn
        {160, 90, 1, "30 20 50 70 1", nullptr, true},          // distort
        {160, 90, 2, "0% 10% 50% 50% 1", nullptr, false},      // percentages
        {200, 100, 1, "-400 -200 960 540 1", "3", false},      // large zoom, slightly rotated
        {40, 30, 3, "60.3 40.7 33.1 21.9 0.93", "-47", false}, // small
    };
    for (const Case &c : cases) {
        Properties properties = {{"rect", c.rect}};
        if (c.rotation) {
            properties.push_back({"rotation", c.rotation});
        }
        if (c.distort) {
            properties.push_back({"distort", "1"});
        }
        const Image input = makeImage(c.width, c.height, c.pattern);
        for (const Properties *frameProperties : {&smooth, &nearest}) {
            // The output has the profile size
            expectSame(profile, {{"qtblend", properties}}, input, false, 0, *frameProperties, profile->width, profile->height);
        }
    }
    const Image input = makeImage(160, 90, 1);
    // Animated, rotation around the center or an anchor point
    expectSame(profile, {{"qtblend", {{"rect", "0=0 0 160 90 1;10=40 20 80 45 0.5"}, {"rotation", "0=0;10=30"}, {"rotate_center", "1"}, {"out", "10"}}}}, input,
               false, 4, smooth, 160, 90);
    expectSame(profile, {{"qtblend", {{"rect", "10 10 100 60 1"}, {"rotation", "-15"}, {"rotate_anchor", "0.2 0.8"}}}}, input, false, 0, smooth, 160, 90);
    // Two transforms: the lower one follows the consumer scaling set by the upper one
    expectSame(profile, {{"qtblend", {{"rect", "10 10 120 70 1"}}}, {"qtblend", {{"rect", "-20 5 140 80 0.9"}, {"rotation", "12"}}}}, input, false, 0, smooth,
               160, 90);
    // Preview at half resolution: consumer scaling
    expectSame(profile, {{"qtblend", {{"rect", "30 15 100 60 1"}, {"rotation", "8"}}}}, makeImage(80, 45, 2), false, 0, smooth, 80, 45);
    // With a background color
    expectSame(profile, {{"qtblend", {{"rect", "30 15 100 60 0.7"}, {"background_color", "0x80402010"}}}}, input, false, 0, smooth, 160, 90);

    if (g_mock.active()) {
        // Zoom with a shadow: one upload, one download (MLT 7.38 downscales on the CPU, hence the zoom)
        g_mock.reset();
        const std::vector<FilterSpec> chain = {{"qtblend", {{"rect", "-20 -10 200 112.5 1"}, {"rotation", "5"}}}, {"dropshadow", {{"radius", "4"}}}};
        expectSame(profile, chain, input, false, 0, smooth, 160, 90);
        g_mock.reset();
        render(profile, chain, input, false, 0, smooth, 160, 90);
        check(g_mock.stats().uploads == 2 && g_mock.stats().downloads == 1,
              "transform and shadow: %llu uploads (the frame and the transform tables) and %llu downloads", (unsigned long long)g_mock.stats().uploads,
              (unsigned long long)g_mock.stats().downloads);

        // Left to the CPU: other composition modes, rotation with distortion
        for (const Properties &properties :
             std::vector<Properties>{{{"rect", "20 10 100 60 1"}, {"compositing", "13"}}, {{"rect", "20 10 100 30 1"}, {"distort", "1"}, {"rotation", "20"}}}) {
            g_mock.reset();
            expectSame(profile, {{"qtblend", properties}}, input, false, 0, smooth, 160, 90);
            check(g_mock.stats().launches == 0, "%s ran on the GPU", describe({{"qtblend", properties}}).c_str());
        }
    }
}

void testChains(mlt_profile profile)
{
    const Image input = makeImage(120, 80, 2);
    const std::vector<FilterSpec> chain = {
        {"brightness", {{"level", "0.8"}}},
        {"box_blur", {{"hradius", "15"}, {"vradius", "15"}}},
        {"dropshadow", {{"radius", "6"}, {"x", "4"}, {"y", "6"}}},
        {"lift_gamma_gain", {{"gain_g", "1.2"}}},
        {"boxblur", {{"blur", "2"}}},
    };
    expectSame(profile, chain, input, false);
    if (g_mock.active() && hasFilters(chain)) {
        // The whole chain runs on the GPU with a single upload and download
        g_mock.reset();
        const Image result = render(profile, chain, input, false);
        const MockCudaStats stats = g_mock.stats();
        // Only small lookup tables are uploaded besides the frame
        const uint64_t frameBytes = uint64_t(input.width) * uint64_t(input.height) * 4;
        check(stats.uploadedBytes < 2 * frameBytes && stats.downloads == 1, "chain: %llu bytes uploaded and %llu downloads, expected one frame each way",
              (unsigned long long)stats.uploadedBytes, (unsigned long long)stats.downloads);
        check(stats.misuse == 0, "chain: %llu invalid CUDA calls", (unsigned long long)stats.misuse);
        check(!result.pixels.empty(), "chain: no output");

        // A lightweight effect on its own stays on the CPU
        g_mock.reset();
        render(profile, {{"brightness", {{"level", "0.5"}}}}, input, false);
        check(g_mock.stats().uploads == 0, "a single brightness filter was uploaded to the GPU");

        // Effects with nothing to do stay transparent between GPU effects
        g_mock.reset();
        expectSame(profile,
                   {{"dropshadow", {{"radius", "5"}}},
                    {"box_blur", {{"hradius", "0"}, {"vradius", "0"}}},
                    {"brightness", {{"level", "1"}}},
                    {"boxblur", {{"blur", "2"}}}},
                   input, false);
        g_mock.reset();
        render(profile,
               {{"dropshadow", {{"radius", "5"}}},
                {"box_blur", {{"hradius", "0"}, {"vradius", "0"}}},
                {"brightness", {{"level", "1"}}},
                {"boxblur", {{"blur", "2"}}}},
               input, false);
        check(g_mock.stats().uploads == 1 && g_mock.stats().downloads == 1, "pass-through filters broke the GPU chain (%llu uploads, %llu downloads)",
              (unsigned long long)g_mock.stats().uploads, (unsigned long long)g_mock.stats().downloads);
    }
    // A CPU-only filter in the middle splits the chain, the result is the same
    expectSame(
        profile,
        {{"dropshadow", {{"radius", "8"}}}, {"gradientmap", {{"stop.1", "#ff000000 0.0"}, {"stop.2", "#ffffffff 1.0"}}}, {"box_blur", {{"hradius", "10"}}}},
        input, false);
}

/** Through a real producer: format conversions, and filters attached to a service. */
void testProducer(mlt_profile profile)
{
    if (!hasFilters({{"qtcrop", {}}, {"dropshadow", {}}, {"brightness", {}}})) {
        return;
    }
    auto renderProducer = [&](bool cpu, mlt_image_format requested) {
        mlt_producer producer = mlt_factory_producer(profile, nullptr, "color:#a0306090");
        if (!producer) {
            return Image();
        }
        std::vector<mlt_filter> filters = createFilters(profile,
                                                        {{"qtcrop", {{"rect", "10 10 100 60"}, {"radius", "0.2"}}},
                                                         {"dropshadow", {{"radius", "9"}, {"x", "6"}, {"y", "-4"}}},
                                                         {"brightness", {{"level", "0.7"}}}},
                                                        cpu);
        for (mlt_filter filter : filters) {
            mlt_producer_attach(producer, filter);
        }
        mlt_frame frame = nullptr;
        mlt_service_get_frame(MLT_PRODUCER_SERVICE(producer), &frame, 0);
        Image output;
        if (frame) {
            uint8_t *image = nullptr;
            mlt_image_format format = requested;
            int width = profile->width;
            int height = profile->height;
            if (!mlt_frame_get_image(frame, &image, &format, &width, &height, 0) && image) {
                output.width = width;
                output.height = height;
                output.pixels.assign(image, image + size_t(width) * size_t(height) * (format == mlt_image_rgba ? 4 : 2));
            }
            mlt_frame_close(frame);
        }
        for (mlt_filter filter : filters) {
            mlt_filter_close(filter);
        }
        mlt_producer_close(producer);
        return output;
    };
    for (mlt_image_format format : {mlt_image_rgba, mlt_image_yuv422}) {
        const Image cpu = renderProducer(true, format);
        const Image gpu = renderProducer(false, format);
        check(!cpu.pixels.empty() && cpu.pixels.size() == gpu.pixels.size(), "producer test: no image for format %d", int(format));
        if (cpu.pixels.size() == gpu.pixels.size()) {
            int maxDiff = 0;
            for (size_t i = 0; i < cpu.pixels.size(); ++i) {
                maxDiff = std::max(maxDiff, std::abs(int(cpu.pixels[i]) - int(gpu.pixels[i])));
            }
            check(maxDiff <= 1, "producer test (format %d): max difference %d", int(format), maxDiff);
        }
    }
}

void testGlobalSwitch(mlt_profile profile)
{
    if (!g_mock.active() || !hasFilter("dropshadow")) {
        return;
    }
    mlt_properties_set(mlt_global_properties(), "gpufx.enabled", "0");
    g_mock.reset();
    expectSame(profile, {{"dropshadow", {{"radius", "5"}}}}, makeImage(40, 30, 0), false);
    check(g_mock.stats().launches == 0, "gpufx.enabled=0 did not turn GPU processing off");
    mlt_properties_set(mlt_global_properties(), "gpufx.enabled", "1");
    g_mock.reset();
    render(profile, {{"dropshadow", {{"radius", "5"}}}}, makeImage(40, 30, 0), false);
    check(g_mock.stats().launches > 0, "gpufx.enabled=1 did not turn GPU processing back on");
}

/** Must run last: a launch failure turns the GPU off for the session. */
void testFailures(mlt_profile profile)
{
    if (!g_mock.active() || !hasFilter("dropshadow")) {
        return;
    }
    const Image input = makeImage(64, 48, 2);
    const std::vector<FilterSpec> chain = {{"box_blur", {{"hradius", "15"}, {"vradius", "15"}}}, {"dropshadow", {{"radius", "6"}}}};

    // Out of memory is recoverable: the frame falls back to the CPU
    g_mock.failAfter(MOCKCUDA_ALLOC, 1, 2);
    expectSame(profile, chain, input, false);
    g_mock.reset();
    render(profile, chain, input, false);
    check(g_mock.stats().launches > 0, "the GPU was not used anymore after an out of memory error");

    // A failing kernel in the middle of the chain: the output is still right
    g_mock.failAfter(MOCKCUDA_LAUNCH, 2, 719);
    expectSame(profile, chain, input, false);
    g_mock.reset();
    expectSame(profile, chain, input, false);
    check(g_mock.stats().launches == 0, "the GPU was still used after a launch failure");
    mlt_filter status = mlt_factory_filter(profile, "gpufx", "noinit");
    check(status && mlt_properties_get_int(MLT_FILTER_PROPERTIES(status), "available") == 0, "status still reports the GPU after a failure");
    if (status) {
        printf("after failure: %s\n", mlt_properties_get(MLT_FILTER_PROPERTIES(status), "status"));
        mlt_filter_close(status);
    }
}

} // namespace

int main(int argc, char **argv)
{
    bool expectGpu = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-gpu")) {
            expectGpu = false;
        }
    }
#if !defined(_WIN32)
    if (!getenv("DISPLAY") && !getenv("WAYLAND_DISPLAY")) {
        setenv("QT_QPA_PLATFORM", "offscreen", 0);
    }
    setenv("GPUFX_CUDA_LIBRARY", GPUFX_TEST_CUDA_LIBRARY, 0);
#else
    if (!getenv("GPUFX_CUDA_LIBRARY")) {
        _putenv_s("GPUFX_CUDA_LIBRARY", GPUFX_TEST_CUDA_LIBRARY);
    }
#endif

    mlt_repository repository = mlt_factory_init(nullptr);
    if (!repository) {
        fprintf(stderr, "cannot initialize MLT\n");
        return 2;
    }
    if (getenv("GPUFX_TEST_VERBOSE")) {
        mlt_log_set_level(MLT_LOG_DEBUG);
    }
    // Register the module from the build tree, unless MLT already loaded it
    if (!mlt_properties_get_data(mlt_repository_filters(repository), "gpufx", nullptr)) {
        void *module = openLibrary(GPUFX_TEST_MODULE);
        auto registerModule = module ? reinterpret_cast<void (*)(mlt_repository)>(findSymbol(module, "mlt_register")) : nullptr;
        if (!registerModule) {
            fprintf(stderr, "cannot load %s\n", GPUFX_TEST_MODULE);
            return 2;
        }
        registerModule(repository);
    }
    if (void *driver = openLibrary(getenv("GPUFX_CUDA_LIBRARY"))) {
        g_mock.getStats = reinterpret_cast<MockCudaGetStats>(findSymbol(driver, "mockcuda_get_stats"));
        g_mock.resetStats = reinterpret_cast<MockCudaResetStats>(findSymbol(driver, "mockcuda_reset_stats"));
        g_mock.failAfter = reinterpret_cast<MockCudaFailAfter>(findSymbol(driver, "mockcuda_fail_after"));
    }
    if (!expectGpu) {
        g_mock = Mock();
    }

    mlt_profile profile = mlt_profile_init(nullptr);
    profile->width = 160;
    profile->height = 90;
    profile->sample_aspect_num = 1;
    profile->sample_aspect_den = 1;
    profile->display_aspect_num = 16;
    profile->display_aspect_den = 9;

    testStatus(profile, expectGpu);
    testDropShadow(profile);
    testBoxBlur(profile);
    testRectBlur(profile);
    testColor(profile);
    testFrei0r(profile);
    testTransform(profile);
    testChains(profile);
    testProducer(profile);
    testGlobalSwitch(profile);
    testFailures(profile);

    if (g_mock.active()) {
        const MockCudaStats stats = g_mock.stats();
        check(stats.misuse == 0, "%llu invalid CUDA calls", (unsigned long long)stats.misuse);
    }
    printf("%d checks, %d failures\n", g_checks, g_failures);
    mlt_profile_close(profile);
    mlt_factory_close();
    return g_failures ? 1 : 0;
}
