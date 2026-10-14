/*
SPDX-FileCopyrightText: 2014 Till Theato <root@ttill.de>
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#pragma once

#include <QString>
#include <QStringList>
#include <memory>

namespace Mlt {
class Repository;
}

/** @class MltConnection
    @brief Initializes MLT and provides access to its API.
    This is where the Mlt Factory is initialized, as well as the producers
 */
class MltConnection
{

public:
    /** @brief Open connection to the MLT framework
     */
    static void construct(const QString &mltPath);

    /** @brief Returns a pointer to the MLT Repository*/
    std::unique_ptr<Mlt::Repository> &getMltRepository();

    /** @brief Returns a pointer to the instance of the singleton */
    static std::unique_ptr<MltConnection> &self();

    /** @brief Updates the list of available Lumas
     */
    static void refreshLumas();

    /** @brief Turns GPU processing of effects (gpufx MLT module) on or off according to the settings */
    static void applyGpuEffectsSetting();

    /** @brief Describes the state of GPU effect processing.
     *  This initializes CUDA if it was not done yet, which can take a moment: do not call it from the GUI thread.
     *  @param available set to true if effects can run on the GPU
     *  @param effects set to the MLT services of the effects that can run on the GPU, transitions as transition:<service> */
    static QString gpuEffectsStatus(bool *available = nullptr, QStringList *effects = nullptr);

protected:
    /** @brief Open connection to the MLT framework
        This constructor should be called only once
    */
    MltConnection(const QString &mltPath);

    /** @brief Locates the MLT environment.
     * @param mltPath (optional) path to MLT environment
     *
     * It tries to set the paths of the MLT profiles and renderer, using
     * mltPath, MLT_PREFIX, searching for the binary `melt`, or asking to the
     * user. It doesn't fill any list of profiles, while its name suggests so. */
    void locateMeltAndProfilesPath(const QString &mltPath = QString());

    static std::unique_ptr<MltConnection> m_self;

    /** @brief The MLT repository, useful for filter/producer requests */
    std::unique_ptr<Mlt::Repository> m_repository;

private:
    QString locateMltProfilePath(const QString &libName, const QString &profilePath = QString());

    /** @brief Locate the melt executable.
     *  @returns the path of the melt executable (including the executable name)
     */
    QString locateMeltExe(const QString &profilePath);
};
