/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * CUDA kernels for the gpufx MLT module.
 *
 * Every kernel reproduces the arithmetic of the CPU filter it replaces, so that
 * switching between CPU and GPU processing does not change the rendered result.
 * Images are 8-bit RGBA (mlt_image_rgba), tightly packed, one 32-bit word per
 * pixel.
 *
 * Keep the kernel list in gpufx_kernel_list.h in sync with this file.
 */

#include "gpufx_device.h"

#define GPUFX_GLOBAL_X (int(blockIdx.x * blockDim.x + threadIdx.x))
#define GPUFX_GLOBAL_Y (int(blockIdx.y * blockDim.y + threadIdx.y))

/* ==========================================================================
 * Drop shadow (MLT "dropshadow" filter, which renders with
 * QGraphicsDropShadowEffect). See gpufx/src/effect_dropshadow.cpp for the
 * geometry, computed on the host.
 * ========================================================================== */

GPUFX_DEV gpufx_u32 gpufx_alpha_at(const gpufx_u32 *__restrict__ src, int w, int h, int x, int y)
{
    if (x < 0 || y < 0 || x >= w || y >= h) {
        return 0;
    }
    return src[size_t(y) * size_t(w) + size_t(x)] >> 24;
}

/* Builds the input of the blur: the alpha channel of the source placed in the
 * padded effect rectangle and shifted by the rounded shadow offset, so that
 * padded (x, y) shows the image at (x + ox, y + oy). When half is set the
 * result is downscaled by two like qt_halfScaled() does (average of averages,
 * truncating). */
GPUFX_KERNEL void gpufx_dropshadow_alpha(const gpufx_u32 *__restrict__ src, int w, int h, gpufx_u8 *__restrict__ dst, int dw, int dh, int ox, int oy, int half)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= dw || y >= dh) {
        return;
    }
    gpufx_u32 v;
    if (half) {
        const int sx = 2 * x + ox;
        const int sy = 2 * y + oy;
        const gpufx_u32 t = (gpufx_alpha_at(src, w, h, sx, sy) + gpufx_alpha_at(src, w, h, sx + 1, sy)) >> 1;
        const gpufx_u32 b = (gpufx_alpha_at(src, w, h, sx, sy + 1) + gpufx_alpha_at(src, w, h, sx + 1, sy + 1)) >> 1;
        v = (t + b) >> 1;
    } else {
        v = gpufx_alpha_at(src, w, h, x + ox, y + oy);
    }
    dst[size_t(y) * size_t(dw) + size_t(x)] = gpufx_u8(v);
}

/* Horizontal pass of Qt's expblur(): an exponential IIR filter run forward and
 * then backward over each row, with the filter state carried over. One thread
 * per row. */
GPUFX_KERNEL void gpufx_expblur_rows(gpufx_u8 *__restrict__ buf, int bw, int bh, int alpha)
{
    const int y = GPUFX_GLOBAL_X;
    if (y >= bh) {
        return;
    }
    gpufx_u8 *row = buf + size_t(y) * size_t(bw);
    int z = 0;
    for (int i = 0; i < bw; ++i) {
        row[i] = gpufx_expblur_step(z, row[i], alpha);
    }
    for (int i = bw - 2; i >= 0; --i) {
        row[i] = gpufx_expblur_step(z, row[i], alpha);
    }
}

/* Vertical pass of Qt's expblur(). Qt rotates the image by 270 degrees and
 * blurs the rows of the rotated copy, which makes the forward pass run from
 * the bottom of each column to the top. One thread per column. */
GPUFX_KERNEL void gpufx_expblur_cols(gpufx_u8 *__restrict__ buf, int bw, int bh, int alpha)
{
    const int x = GPUFX_GLOBAL_X;
    if (x >= bw) {
        return;
    }
    gpufx_u8 *col = buf + x;
    const size_t stride = size_t(bw);
    int z = 0;
    for (int i = bh - 1; i >= 0; --i) {
        col[size_t(i) * stride] = gpufx_expblur_step(z, col[size_t(i) * stride], alpha);
    }
    for (int i = 1; i < bh; ++i) {
        col[size_t(i) * stride] = gpufx_expblur_step(z, col[size_t(i) * stride], alpha);
    }
}

/* Smooth 2x upscale of the half resolution blur, as done by QPainter with
 * SmoothPixmapTransform (fetchTransformedBilinearARGB32PM_simple_scale_helper):
 * 16.16 fixed point sampling with the 1/65536 offset of QSpanData::setupMatrix,
 * 8-bit weights, vertical interpolation first, clamped edges. */
GPUFX_DEV gpufx_u32 gpufx_qt_upscale2(const gpufx_u8 *__restrict__ img, int iw, int ih, int x, int y)
{
    const int fy = 32768 * y - 16385;
    int y1 = fy >> 16;
    int y2;
    const gpufx_u32 disty = gpufx_u32(fy & 0xffff) >> 8;
    if (y1 < 0) {
        y1 = y2 = 0;
    } else if (y1 >= ih - 1) {
        y1 = y2 = ih - 1;
    } else {
        y2 = y1 + 1;
    }
    const int fx = 32768 * x - 16385;
    int x1 = fx >> 16;
    int x2;
    const gpufx_u32 distx = gpufx_u32(fx & 0xffff) >> 8;
    if (x1 < 0) {
        x1 = x2 = 0;
    } else if (x1 >= iw - 1) {
        x1 = x2 = iw - 1;
    } else {
        x2 = x1 + 1;
    }
    const gpufx_u8 *r1 = img + size_t(y1) * size_t(iw);
    const gpufx_u8 *r2 = img + size_t(y2) * size_t(iw);
    const gpufx_u32 left = (gpufx_u32(r1[x1]) * (256u - disty) + gpufx_u32(r2[x1]) * disty) >> 8;
    const gpufx_u32 right = (gpufx_u32(r1[x2]) * (256u - disty) + gpufx_u32(r2[x2]) * disty) >> 8;
    return (left * (256u - distx) + right * distx) >> 8;
}

/* Final composition of the drop shadow onto the image: the colored shadow
 * (SourceIn fill of the blurred alpha) is drawn over the image, then the
 * original image is drawn over the result. Both draws go through Qt's 16-bit
 * pipeline because the target QImage is RGBA8888 (straight alpha).
 * (pl, pt) is the top left corner of the padded effect rectangle, which is
 * where the blur buffer starts in image coordinates. */
GPUFX_KERNEL void gpufx_dropshadow_composite(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, const gpufx_u8 *__restrict__ blur,
                                             int bw, int bh, int pl, int pt, int half, gpufx_u32 color_pm)
{
    const int x = GPUFX_GLOBAL_X;
    const int y = GPUFX_GLOBAL_Y;
    if (x >= w || y >= h) {
        return;
    }
    const int px = x - pl;
    const int py = y - pt;
    gpufx_u32 shadow_alpha;
    if (half) {
        shadow_alpha = (px < 2 * bw && py < 2 * bh) ? gpufx_qt_upscale2(blur, bw, bh, px, py) : 0u;
    } else {
        shadow_alpha = blur[size_t(py) * size_t(bw) + size_t(px)];
    }
    const gpufx_u32 shadow = gpufx_byte_mul(color_pm, shadow_alpha);
    const size_t i = size_t(y) * size_t(w) + size_t(x);
    const gpufx_u32 p = src[i];
    const gpufx_u32 with_shadow = gpufx_store64(gpufx_source_over64(gpufx_expand_pm8(shadow), gpufx_fetch64(p)));
    dst[i] = gpufx_store64(gpufx_source_over64(gpufx_expand_pm8(gpufx_premultiply8(p)), gpufx_fetch64(with_shadow)));
}

/* Drop shadow with no blur and no offset: Qt only draws the source item over
 * the image (QGraphicsEffect::drawSource()). */
GPUFX_KERNEL void gpufx_over_self(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int n)
{
    const int i = GPUFX_GLOBAL_X;
    if (i >= n) {
        return;
    }
    const gpufx_u32 p = src[i];
    dst[i] = gpufx_store64(gpufx_source_over64(gpufx_expand_pm8(gpufx_premultiply8(p)), gpufx_fetch64(p)));
}

/* ==========================================================================
 * Box blur (MLT core "box_blur" filter, mlt_image_box_blur()): separable
 * moving average over 2 * radius + 1 pixels with edge pixels repeated,
 * rounded to nearest. Each thread handles a run of GPUFX_BOX_RUN pixels.
 * ========================================================================== */

#define GPUFX_BOX_RUN 32

GPUFX_DEV gpufx_u32 gpufx_box_average(int sum, int diameter)
{
    // lrint(sum / diameter): the diameter is odd so there are no ties
    return gpufx_u32((2 * sum + diameter) / (2 * diameter));
}

GPUFX_DEV void gpufx_add_px(int *acc, gpufx_u32 p, int sign)
{
    acc[0] += sign * int(p & 0xffu);
    acc[1] += sign * int((p >> 8) & 0xffu);
    acc[2] += sign * int((p >> 16) & 0xffu);
    acc[3] += sign * int(p >> 24);
}

GPUFX_DEV gpufx_u32 gpufx_box_output(const int *acc, int diameter, gpufx_u32 original, int preserve_alpha)
{
    const gpufx_u32 a = preserve_alpha ? (original >> 24) : gpufx_box_average(acc[3], diameter);
    return gpufx_pack(gpufx_box_average(acc[0], diameter), gpufx_box_average(acc[1], diameter), gpufx_box_average(acc[2], diameter), a);
}

GPUFX_KERNEL void gpufx_box_blur_h(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, int radius, int preserve_alpha)
{
    const int x0 = GPUFX_GLOBAL_X * GPUFX_BOX_RUN;
    const int y = GPUFX_GLOBAL_Y;
    if (x0 >= w || y >= h) {
        return;
    }
    const gpufx_u32 *row = src + size_t(y) * size_t(w);
    gpufx_u32 *out = dst + size_t(y) * size_t(w);
    const int last = w - 1;
    const int r = gpufx_min(radius, w / 2);
    const int diameter = 2 * r + 1;
    int acc[4] = {0, 0, 0, 0};
    for (int k = -r; k <= r; ++k) {
        gpufx_add_px(acc, row[gpufx_clamp(x0 + k, 0, last)], 1);
    }
    const int x1 = gpufx_min(x0 + GPUFX_BOX_RUN, w);
    for (int x = x0; x < x1; ++x) {
        out[x] = gpufx_box_output(acc, diameter, row[x], preserve_alpha);
        gpufx_add_px(acc, row[gpufx_clamp(x + r + 1, 0, last)], 1);
        gpufx_add_px(acc, row[gpufx_clamp(x - r, 0, last)], -1);
    }
}

GPUFX_KERNEL void gpufx_box_blur_v(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int w, int h, int radius, int preserve_alpha)
{
    const int x = GPUFX_GLOBAL_X;
    const int y0 = GPUFX_GLOBAL_Y * GPUFX_BOX_RUN;
    if (x >= w || y0 >= h) {
        return;
    }
    const size_t stride = size_t(w);
    const gpufx_u32 *col = src + x;
    gpufx_u32 *out = dst + x;
    const int last = h - 1;
    const int r = gpufx_min(radius, h / 2);
    const int diameter = 2 * r + 1;
    int acc[4] = {0, 0, 0, 0};
    for (int k = -r; k <= r; ++k) {
        gpufx_add_px(acc, col[size_t(gpufx_clamp(y0 + k, 0, last)) * stride], 1);
    }
    const int y1 = gpufx_min(y0 + GPUFX_BOX_RUN, h);
    for (int y = y0; y < y1; ++y) {
        out[size_t(y) * stride] = gpufx_box_output(acc, diameter, col[size_t(y) * stride], preserve_alpha);
        gpufx_add_px(acc, col[size_t(gpufx_clamp(y + r + 1, 0, last)) * stride], 1);
        gpufx_add_px(acc, col[size_t(gpufx_clamp(y - r, 0, last)) * stride], -1);
    }
}

/* ==========================================================================
 * "Box Blur" of the MLT kdenlive module ("boxblur" filter). The CPU version
 * reads a summed area table at clamped corners, which amounts to summing the
 * pixels in (clamp(x - bw), clamp(x + bw)] x (clamp(y - bh), clamp(y + bh)]
 * and scaling by 1 / (2bw * 2bh) in float, truncated. Here the same sums are
 * computed in two exact integer passes.
 * ========================================================================== */

/* Horizontal sums, one int per channel (sums is 4 ints per pixel). */
GPUFX_KERNEL void gpufx_rect_blur_h(const gpufx_u32 *__restrict__ src, int *__restrict__ sums, int w, int h, int bw)
{
    const int x0 = GPUFX_GLOBAL_X * GPUFX_BOX_RUN;
    const int y = GPUFX_GLOBAL_Y;
    if (x0 >= w || y >= h) {
        return;
    }
    const gpufx_u32 *row = src + size_t(y) * size_t(w);
    int *out = sums + 4 * size_t(y) * size_t(w);
    const int last = w - 1;
    int lo = gpufx_clamp(x0 - bw, 0, last);
    int hi = gpufx_clamp(x0 + bw, 0, last);
    int acc[4] = {0, 0, 0, 0};
    for (int i = lo + 1; i <= hi; ++i) {
        gpufx_add_px(acc, row[i], 1);
    }
    const int x1 = gpufx_min(x0 + GPUFX_BOX_RUN, w);
    for (int x = x0; x < x1; ++x) {
        int *o = out + 4 * size_t(x);
        o[0] = acc[0];
        o[1] = acc[1];
        o[2] = acc[2];
        o[3] = acc[3];
        const int nlo = gpufx_clamp(x + 1 - bw, 0, last);
        const int nhi = gpufx_clamp(x + 1 + bw, 0, last);
        if (nhi > hi) {
            gpufx_add_px(acc, row[nhi], 1);
        }
        if (nlo > lo) {
            gpufx_add_px(acc, row[nlo], -1);
        }
        lo = nlo;
        hi = nhi;
    }
}

/* Vertical sums of the horizontal sums, then the CPU filter's float scaling. */
GPUFX_KERNEL void gpufx_rect_blur_v(const int *__restrict__ sums, gpufx_u32 *__restrict__ dst, int w, int h, int bh, float mul)
{
    const int x = GPUFX_GLOBAL_X;
    const int y0 = GPUFX_GLOBAL_Y * GPUFX_BOX_RUN;
    if (x >= w || y0 >= h) {
        return;
    }
    const size_t stride = 4 * size_t(w);
    const int *col = sums + 4 * size_t(x);
    const int last = h - 1;
    int lo = gpufx_clamp(y0 - bh, 0, last);
    int hi = gpufx_clamp(y0 + bh, 0, last);
    int acc[4] = {0, 0, 0, 0};
    for (int j = lo + 1; j <= hi; ++j) {
        const int *s = col + size_t(j) * stride;
        acc[0] += s[0];
        acc[1] += s[1];
        acc[2] += s[2];
        acc[3] += s[3];
    }
    const int y1 = gpufx_min(y0 + GPUFX_BOX_RUN, h);
    for (int y = y0; y < y1; ++y) {
        gpufx_u32 c[4];
        for (int k = 0; k < 4; ++k) {
            c[k] = gpufx_f2u_rz(gpufx_fmul_rn(gpufx_i2f_rn(acc[k]), mul)) & 0xffu;
        }
        dst[size_t(y) * size_t(w) + size_t(x)] = gpufx_pack(c[0], c[1], c[2], c[3]);
        const int nlo = gpufx_clamp(y + 1 - bh, 0, last);
        const int nhi = gpufx_clamp(y + 1 + bh, 0, last);
        if (nhi > hi) {
            const int *s = col + size_t(nhi) * stride;
            acc[0] += s[0];
            acc[1] += s[1];
            acc[2] += s[2];
            acc[3] += s[3];
        }
        if (nlo > lo) {
            const int *s = col + size_t(nlo) * stride;
            acc[0] -= s[0];
            acc[1] -= s[1];
            acc[2] -= s[2];
            acc[3] -= s[3];
        }
        lo = nlo;
        hi = nhi;
    }
}

/* ==========================================================================
 * Per channel lookup tables (brightness/fades, lift/gamma/gain). The tables
 * are computed on the host with the CPU filter's own formulas, which keeps
 * the results identical. lut holds 4 x 256 entries: R, G, B, A.
 * ========================================================================== */

GPUFX_KERNEL void gpufx_lut_rgba(const gpufx_u32 *__restrict__ src, gpufx_u32 *__restrict__ dst, int n, const gpufx_u8 *__restrict__ lut)
{
    const int i = GPUFX_GLOBAL_X;
    if (i >= n) {
        return;
    }
    const gpufx_u32 p = src[i];
    dst[i] = gpufx_pack(lut[p & 0xffu], lut[256 + ((p >> 8) & 0xffu)], lut[512 + ((p >> 16) & 0xffu)], lut[768 + (p >> 24)]);
}
