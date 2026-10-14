/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <framework/mlt.h>

namespace gpufx {

// MLT 7.42 turned the frame's convert_image into a chain of converters

inline bool hasConvertImage(mlt_frame frame)
{
#if LIBMLT_VERSION_INT >= ((7 << 16) + (42 << 8))
    return mlt_frame_has_convert_image(frame);
#else
    return frame->convert_image != nullptr;
#endif
}

inline int convertImage(mlt_frame frame, uint8_t **image, mlt_image_format *format, mlt_image_format output)
{
#if LIBMLT_VERSION_INT >= ((7 << 16) + (42 << 8))
    return mlt_frame_convert_image(frame, image, format, output);
#else
    return frame->convert_image(frame, image, format, output);
#endif
}

} // namespace gpufx
