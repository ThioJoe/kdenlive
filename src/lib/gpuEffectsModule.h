/*
SPDX-FileCopyrightText: 2026 Kdenlive contributors
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

/** @brief Registers the gpufx MLT module (GPU effects) if MLT did not load it.
 *
 *  MLT only loads modules from its own directory. When Kdenlive is installed to
 *  another prefix than MLT, the module is next to Kdenlive instead. Call this
 *  right after initializing the MLT factory. */
void loadGpuEffectsModule();
