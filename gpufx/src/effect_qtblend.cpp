/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

/*
 * GPU version of MLT's "qtblend" filter, Kdenlive's Transform effect.
 *
 * qtblend draws the frame with QPainter onto a transparent image: a transform
 * built from the effect rectangle and rotation, and, when the consumer scales
 * with interpolation ("consumer.rescale" other than nearest), bilinear
 * sampling and antialiased edges. The geometry below follows
 * filter_qtblend.cpp of MLT 7.38 and of MLT 7.40 and later, which scales the
 * source differently. The drawing follows Qt's raster engine (qt_raster.cpp
 * for the coverage and the span handling, the gpufx_qt_draw_image kernel for
 * the pixels). Only the default composition mode (alpha blend) and 8-bit
 * processing are handled, everything else stays on the CPU.
 */

#include "effect.h"
#include "gpufx_log.h"
#include "qt_raster.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace gpufx {

namespace {

constexpr int kMaxDimension = 16000; // MLT_QTBLEND_MAX_DIMENSION
constexpr double kMipmapStep = 0.85; // MLT_QT_MIPMAP_STEP

// Fixed point of qdrawhelper.cpp
constexpr int kFixedScale = 1 << 16;
constexpr int kHalfPoint = 1 << 15;

// Modes of gpufx_qt_draw_image
enum DrawMode { Untransformed = 0, Nearest = 1, Bilinear = 2 };

int qRound(double d)
{
    return qt::round(d);
}

template <typename T> T qBound(T min, T value, T max)
{
    return std::max(min, std::min(max, value));
}

/** Frame property changed by the plan, with its previous value */
struct SavedProperty
{
    const char *name;
    bool existed;
    double value;
};

class TransformJob final : public EffectJob
{
public:
    // Decided by the plan, before the source image is fetched
    bool scaleWithQPainter = true; // MLT >= 7.40: the producer scales close to the target, QPainter does the rest
    mlt_rect rect{};
    double consumerAr = 1;
    double sourceAr = 1;
    double sourceDar = 1;
    bool distort = false;
    qt::Transform transform; // translation and rotation
    double opacity = 1;
    bool smooth = false;
    uint32_t background = 0;
    bool setAspectRatio = false;
    bool storeScaling = false; // first qtblend of the frame: it records the consumer scaling
    double scaleX = 1;
    double scaleY = 1;
    std::vector<SavedProperty> saved;

    // Computed by prepare()
    int mode = Untransformed;
    int fdx = 0;
    int fdy = 0;
    int intOpacity = 256;
    std::vector<int> table; // rows, then spans, then runs
    int spansOffset = 0;
    int runsOffset = 0;

    bool prepare(int inputWidth, int inputHeight) override;

    CUresult run(GpuScope &, int, int, CUdeviceptr, CUdeviceptr) override { return CUDA_ERROR_INVALID_VALUE; }

    CUresult runResized(GpuScope &scope, int inWidth, int inHeight, CUdeviceptr in, int outWidth, int outHeight, CUdeviceptr out) override
    {
        DeviceBuffer tables;
        const size_t bytes = table.size() * sizeof(int);
        CUresult res = tables.allocate(bytes);
        if (res == CUDA_SUCCESS) {
            res = scope.upload(tables.ptr(), table.data(), bytes);
        }
        if (res == CUDA_SUCCESS) {
            const CUdeviceptr rows = tables.ptr();
            const CUdeviceptr spans = rows + CUdeviceptr(spansOffset) * sizeof(int);
            const CUdeviceptr runs = rows + CUdeviceptr(runsOffset) * sizeof(int);
            const uint32_t bg = background;
            res = scope.launch2D(KernelId::gpufx_qt_draw_image, outWidth, outHeight, 32, 4, in, inWidth, inHeight, out, outWidth, outHeight, rows, spans, runs,
                                 mode, fdx, fdy, intOpacity, bg);
        }
        return res;
    }

    void beforeFetch(mlt_frame frame) override
    {
        // Same frame changes as the CPU filter, before the source is fetched
        mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
        if (setAspectRatio) {
            mlt_frame_set_aspect_ratio(frame, consumerAr);
        }
        if (storeScaling) {
            for (const char *name : {"qtblend_scalingx", "qtblend_scalingy"}) {
                saved.push_back({name, mlt_properties_get(properties, name) != nullptr, mlt_properties_get_double(properties, name)});
            }
            mlt_properties_set_double(properties, "qtblend_scalingx", scaleX);
            mlt_properties_set_double(properties, "qtblend_scalingy", scaleY);
        }
    }

    void restoreFrame(mlt_frame frame) override
    {
        mlt_properties properties = MLT_FRAME_PROPERTIES(frame);
        for (const SavedProperty &property : saved) {
            if (property.existed) {
                mlt_properties_set_double(properties, property.name, property.value);
            } else {
                mlt_properties_set(properties, property.name, nullptr);
            }
        }
        saved.clear();
    }

    int outputWidth = 0;
    int outputHeight = 0;

private:
    bool buildTable(int sourceWidth, int sourceHeight, const qt::Transform &matrix);
};

/** The rest of filter_get_image() once the source is fetched: scaling of the
 *  source into the rectangle. Returns false where MLT 7.38 scales the image
 *  with QImage::scaled() instead of QPainter. */
bool TransformJob::prepare(int bWidth, int bHeight)
{
    qt::Transform matrix = transform;
    if (scaleWithQPainter) {
        if (distort) {
            if (rect.w != bWidth || rect.h != bHeight) {
                matrix.scale(rect.w / bWidth, rect.h / bHeight);
            }
        } else {
            double scale;
            const double resizeDar = rect.w * consumerAr / rect.h;
            if (sourceDar >= resizeDar) {
                scale = rect.w / bWidth;
            } else {
                scale = rect.h / bHeight;
            }
            // Center image in rect
            matrix.translate((rect.w - (bWidth * scale)) / 2.0, (rect.h - (bHeight * scale)) / 2.0);
            matrix.scale(scale, scale);
        }
    } else {
        if (distort) {
            if (rect.w != bWidth || rect.h != bHeight) {
                if (rect.w < bWidth || rect.h < bHeight) {
                    return false;
                }
                matrix.scale(rect.w / bWidth, rect.h / bHeight);
            }
        } else {
            double scale;
            const double resizeDar = rect.w * consumerAr / rect.h;
            if (sourceDar >= resizeDar) {
                scale = rect.w / bWidth;
            } else {
                scale = rect.h / bHeight * sourceAr;
            }
            if (scale < 1.) {
                return false;
            }
            matrix.translate((rect.w - (bWidth * scale)) / 2.0, (rect.h - (bHeight * scale)) / 2.0);
            if (scale > 1.) {
                matrix.scale(scale, scale);
            }
        }
    }
    // QPainter::setOpacity() and QRasterPaintEngine::opacityChanged()
    const double clamped = std::max(0.0, std::min(1.0, opacity));
    intOpacity = int(clamped * 256);
    return buildTable(bWidth, bHeight, matrix);
}

/** QPainter::drawImage(0, 0, source) through QRasterPaintEngine: decides the
 *  code path and computes the spans and runs Qt would process. */
bool TransformJob::buildTable(int sw, int sh, const qt::Transform &matrix)
{
    const int dw = outputWidth;
    const int dh = outputHeight;
    std::vector<qt::Span> spans;
    std::vector<qt::Run> runs;
    const qt::TransformationType type = matrix.type();
    // Source position of the first pixel of each run
    std::vector<int> starts;

    if (type <= qt::TxTranslate) {
        // drawImage(QPointF): untransformed texture at the rounded position
        const double ptX = matrix.dx();
        const double ptY = matrix.dy();
        qt::fillRectSpans(qRound(ptX), qRound(ptY), sw, sh, dw, dh, spans);
        qt::spanRuns(spans, 512, false, runs);
        // blend_untransformed_generic_rgb64(): data->dx is -pt.x()
        const int xoff = -qRound(ptX);
        const int yoff = -qRound(ptY);
        for (const qt::Run &run : runs) {
            starts.push_back(xoff + run.x);
            starts.push_back(yoff + run.y);
        }
        mode = Untransformed;
        fdx = 0;
        fdy = 0;
    } else {
        // drawImage(QRectF(0, 0, sw, sh), image, QRectF(0, 0, sw, sh))
        if (type <= qt::TxScale && !smooth && sw == 1 && sh == 1) {
            return false; // drawn as a solid color
        }
        // QSpanData::setupMatrix()
        qt::Transform delta;
        delta.translate(1.0 / 65536, 1.0 / 65536);
        const qt::Transform inv = (delta * matrix).inverted();
        const double m11 = inv.m11();
        const double m12 = inv.m12();
        const double m21 = inv.m21();
        const double m22 = inv.m22();
        const double dx = inv.dx();
        const double dy = inv.dy();
        const double f1 = m11 * m11 + m21 * m21;
        const double f2 = m12 * m12 + m22 * m22;
        const bool fastMatrix =
            inv.isAffine() && f1 < 1e4 && f2 < 1e4 && f1 > (1.0 / 65536) && f2 > (1.0 / 65536) && std::fabs(dx) < 1e4 && std::fabs(dy) < 1e4;
        if (!fastMatrix) {
            return false;
        }
        const bool noShear = qt::scaleForTransform(matrix);
        if (!smooth && type == qt::TxScale) {
            double rx, ry, rw, rh;
            matrix.mapRect(0, 0, sw, sh, rx, ry, rw, rh);
            const int x1 = qRound(rx);
            const int y1 = qRound(ry);
            const int x2 = qRound(rx + rw);
            const int y2 = qRound(ry + rh);
            qt::fillRectSpans(x1, y1, x2 - x1, y2 - y1, dw, dh, spans);
            qt::spanRuns(spans, 512, true, runs);
        } else if (noShear || type == qt::TxScale) {
            const qt::PointF a = matrix.map({0, (0 + sh) * 0.5});
            const qt::PointF b = matrix.map({(sw + sw) * 0.5, (0 + sh) * 0.5});
            const double width = noShear ? double(sh) / sw : std::fabs((matrix.m22() * sh) / (matrix.m11() * sw));
            qt::rasterizeLine(a, b, width, smooth, {0, 0, dw - 1, dh - 1}, spans);
            qt::spanRuns(spans, 256, true, runs);
        } else {
            return false; // sheared: drawn as a path
        }
        mode = smooth ? Bilinear : Nearest;
        fdx = int(m11 * kFixedScale);
        fdy = int(m12 * kFixedScale);
        for (const qt::Run &run : runs) {
            const double cx = run.x + 0.5;
            const double cy = run.y + 0.5;
            // canUseFastMatrixPath()
            double fx = (m21 * cy + m11 * cx + dx) * kFixedScale;
            double fy = (m22 * cy + m12 * cx + dy) * kFixedScale;
            double minc = std::min(fx, fy);
            double maxc = std::max(fx, fy);
            fx += std::trunc(m11 * kFixedScale) * run.len;
            fy += std::trunc(m12 * kFixedScale) * run.len;
            minc = std::min(minc, std::min(fx, fy));
            maxc = std::max(maxc, std::max(fx, fy));
            if (!(minc >= double(INT_MIN) && maxc <= double(INT_MAX))) {
                return false;
            }
            int startX = int((m21 * cy + m11 * cx + dx) * kFixedScale);
            int startY = int((m22 * cy + m12 * cx + dy) * kFixedScale);
            if (smooth) {
                startX -= kHalfPoint;
                startY -= kHalfPoint;
            }
            starts.push_back(startX);
            starts.push_back(startY);
        }
    }

    // rows: first span, span count, first run, run count
    table.assign(size_t(4) * size_t(dh), 0);
    spansOffset = int(table.size());
    for (size_t i = 0; i < spans.size(); ++i) {
        const qt::Span &span = spans[i];
        int *row = &table[size_t(4) * size_t(span.y)];
        if (row[1] == 0) {
            row[0] = int(i);
        }
        ++row[1];
        table.push_back(span.x);
        table.push_back(span.len);
        table.push_back(span.coverage);
    }
    runsOffset = int(table.size());
    for (size_t i = 0; i < runs.size(); ++i) {
        const qt::Run &run = runs[i];
        int *row = &table[size_t(4) * size_t(run.y)];
        if (row[3] == 0) {
            row[2] = int(i);
        }
        ++row[3];
        table.push_back(run.x);
        table.push_back(run.len);
        table.push_back(starts[2 * i]);
        table.push_back(starts[2 * i + 1]);
    }
    // Spans and runs come row by row: the kernel looks them up per row
    return true;
}

/** filter_get_image() of filter_qtblend.cpp up to the fetch of the source */
EffectPlan planTransform(const EffectContext &ctx)
{
    const int version = mlt_version_get_int();
    // The geometry of MLT 7.38 to 7.44, newer versions are left to the CPU until checked
    static const bool forced = getenv("MLT_GPUFX_QTBLEND") && !strcmp(getenv("MLT_GPUFX_QTBLEND"), "force");
    if (version < 0x072600 || (version >= 0x072d00 && !forced)) {
        return EffectPlan::cpu();
    }
    // choose_image_format(): 16-bit processing paints in RGBA64
    if (ctx.format == mlt_image_rgba64) {
        return EffectPlan::cpu();
    }
    mlt_properties properties = ctx.properties;
    // Other composition modes than alpha blend
    if (mlt_properties_get_int(properties, "compositing") != 0) {
        return EffectPlan::cpu();
    }
    mlt_profile profile = mlt_service_profile(MLT_FILTER_SERVICE(ctx.filter));
    if (!profile || profile->width <= 0 || profile->height <= 0) {
        return EffectPlan::cpu();
    }
    mlt_properties frameProperties = MLT_FRAME_PROPERTIES(ctx.frame);
    int width = ctx.width;
    int height = ctx.height;

    auto job = std::make_unique<TransformJob>();
    job->scaleWithQPainter = version >= 0x072800;

    const int normalizedWidth = profile->width;
    const int normalizedHeight = profile->height;
    const double consumerAr = mlt_profile_sar(profile);

    mlt_rect rect = {0, 0, double(normalizedWidth), double(normalizedHeight), 1.0};
    int bWidth = mlt_properties_get_int(frameProperties, "meta.media.width");
    int bHeight = mlt_properties_get_int(frameProperties, "meta.media.height");
    const bool distort = mlt_properties_get_int(properties, "distort");

    if (bHeight == 0) {
        bWidth = normalizedWidth;
        bHeight = normalizedHeight;
    }
    // Special case - aspect_ratio = 0
    double bAr = mlt_frame_get_aspect_ratio(ctx.frame);
    const bool setAspectRatio = bAr == 0;
    if (setAspectRatio) {
        bAr = consumerAr;
    }
    const double bDar = bAr * bWidth / bHeight;

    // A qtblend filter below this one already set the consumer scaling
    const double qtblendScaleX = std::min(1., mlt_properties_get_double(frameProperties, "qtblend_scalingx"));
    if (mlt_properties_get(properties, "rect")) {
        rect = mlt_properties_anim_get_rect(properties, "rect", ctx.position, ctx.length);
        if (strchr(mlt_properties_get(properties, "rect"), '%')) {
            rect.x *= normalizedWidth;
            rect.y *= normalizedHeight;
            rect.w *= normalizedWidth;
            rect.h *= normalizedHeight;
        }
    }
    if (qtblendScaleX > 0.) {
        const double qtblendScaleY = std::min(1., mlt_properties_get_double(frameProperties, "qtblend_scalingy"));
        int tmpWidth = bWidth;
        int tmpHeight = bHeight;
        const double scaleFactor = std::max(width / rect.w, height / rect.h);
        if (scaleFactor > 1.) {
            tmpWidth *= scaleFactor;
            tmpHeight *= scaleFactor;
        }
        if (consumerAr * normalizedHeight / normalizedWidth < 1.) {
            width = qBound(qRound(normalizedWidth * qtblendScaleX), tmpWidth, kMaxDimension);
            height = qRound(width * consumerAr * normalizedHeight / normalizedWidth);
        } else {
            height = qBound(qRound(normalizedHeight * qtblendScaleY), tmpHeight, kMaxDimension);
            width = qRound(height * normalizedWidth / normalizedHeight / consumerAr);
        }
        double scale = double(width) / normalizedWidth;
        if (scale != 1.0) {
            rect.x *= scale;
            rect.w *= scale;
        }
        scale = double(height) / normalizedHeight;
        if (scale != 1.0) {
            rect.y *= scale;
            rect.h *= scale;
        }
    } else {
        // First instance of a qtblend filter: check if the requested frame size is scaled
        double scalex = mlt_profile_scale_width(profile, width);
        const double scaley = mlt_profile_scale_height(profile, height);
        job->storeScaling = true;
        job->scaleX = scalex;
        job->scaleY = scaley;
        if (scalex != 1.0) {
            rect.x *= scalex;
            rect.w *= scalex;
            if (bWidth < normalizedWidth) {
                // Adjust scale so that we don't request too small images
                scalex = qBound(scalex, normalizedWidth * scalex / bWidth, 1.);
            }
            // Apply consumer scaling to the source image
            if (scalex < 1.) {
                bWidth *= scalex;
                bHeight *= scalex;
            }
        }
        if (scaley != 1.0) {
            rect.y *= scaley;
            rect.h *= scaley;
        }
    }

    if (job->scaleWithQPainter) {
        // normalize_mlt_source_size()
        if (bAr > 0.0 && consumerAr > 0.0 && bAr != consumerAr && bHeight > 0) {
            bWidth = qRound(bWidth * bAr / consumerAr);
        }
        bWidth = std::max(1, bWidth);
        // Let the producer downscale (adjust_mlt_mipmap_size())
        if (rect.w > 0 && rect.h > 0 && bWidth > 0 && bHeight > 0) {
            double scaleTarget;
            if (distort) {
                scaleTarget = std::max(rect.w / bWidth, rect.h / bHeight);
            } else {
                const double resizeDar = rect.w * consumerAr / rect.h;
                if (bDar >= resizeDar) {
                    scaleTarget = rect.w / bWidth;
                } else {
                    scaleTarget = rect.h / bHeight;
                }
            }
            double mipmapScale = 1.0;
            while (mipmapScale * kMipmapStep >= scaleTarget && mipmapScale > 0.001) {
                mipmapScale *= kMipmapStep;
            }
            if (mipmapScale < 1.0) {
                bWidth = std::max(1, qRound(bWidth * mipmapScale));
                bHeight = std::max(1, qRound(bHeight * mipmapScale));
            }
        }
    }

    job->transform.translate(rect.x, rect.y);
    bool hasAlpha = rect.o < 1 || rect.x != 0 || rect.y != 0 || rect.w != width || rect.h != height || rect.w / bDar < height || rect.h * bDar < width ||
                    bWidth != width || bHeight != height;
    if (mlt_properties_get(properties, "rotation")) {
        const double angle = mlt_properties_anim_get_double(properties, "rotation", ctx.position, ctx.length);
        if (angle != 0.0) {
            if (mlt_properties_get(properties, "rotate_anchor")) {
                const mlt_rect anchor = mlt_properties_anim_get_rect(properties, "rotate_anchor", ctx.position, ctx.length);
                const double anchorX = anchor.x * rect.w;
                const double anchorY = anchor.y * rect.h;
                job->transform.translate(anchorX, anchorY);
                job->transform.rotate(angle);
                job->transform.translate(-anchorX, -anchorY);
            } else if (mlt_properties_get_int(properties, "rotate_center")) {
                job->transform.translate(rect.w / 2.0, rect.h / 2.0);
                job->transform.rotate(angle);
                job->transform.translate(-rect.w / 2.0, -rect.h / 2.0);
            } else {
                job->transform.rotate(angle);
            }
            hasAlpha = true;
        }
    }
    // Nothing to paint: the CPU filter passes the image through or paints it as is
    if (!hasAlpha) {
        return EffectPlan::cpu();
    }
    if (!(rect.w > 0 && rect.h > 0) || !std::isfinite(rect.x) || !std::isfinite(rect.y) || !std::isfinite(rect.w) || !std::isfinite(rect.h) || bWidth <= 0 ||
        bHeight <= 0 || width <= 0 || height <= 0) {
        return EffectPlan::cpu();
    }

    const char *interpolation = mlt_properties_get(frameProperties, "consumer.rescale");
    job->smooth = interpolation && strcmp(interpolation, "nearest") && strcmp(interpolation, "neighbor");
    job->rect = rect;
    job->consumerAr = consumerAr;
    job->sourceAr = bAr;
    job->sourceDar = bDar;
    job->distort = distort;
    job->opacity = rect.o;
    job->background = uint32_t(mlt_properties_get_int(properties, "background_color"));
    job->outputWidth = width;
    job->outputHeight = height;

    job->setAspectRatio = setAspectRatio;

    EffectPlan plan = EffectPlan::gpu(std::move(job));
    plan.inputWidth = bWidth;
    plan.inputHeight = bHeight;
    plan.outputWidth = width;
    plan.outputHeight = height;
    return plan;
}

} // namespace

const EffectDescriptor &transformEffect()
{
    static const EffectDescriptor descriptor{"qtblend", false, planTransform};
    return descriptor;
}

} // namespace gpufx
