/*
    SPDX-FileCopyrightText: 2026 Kdenlive contributors
    SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include "effect.h"

#include <framework/mlt.h>

namespace gpufx {

/** Replaces filter->process so that frames of this filter go through the bridge. */
void wrapFilter(mlt_filter filter, const EffectDescriptor &effect);

/** Module wide switches (environment, MLT global properties, per filter). */
bool moduleEnabled();
bool effectEnabled(const EffectDescriptor &effect);

} // namespace gpufx
