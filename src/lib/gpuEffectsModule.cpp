/*
SPDX-FileCopyrightText: 2026 Kdenlive contributors
SPDX-License-Identifier: GPL-3.0-only OR LicenseRef-KDE-Accepted-GPL
*/

#include "gpuEffectsModule.h"
#include "config-kdenlive.h"

#include <QCoreApplication>
#include <QDebug>
#include <QDir>
#include <QFile>
#include <QLibrary>
#include <QStringList>

#include <framework/mlt.h>

void loadGpuEffectsModule()
{
    mlt_repository repository = mlt_factory_repository();
    const QString moduleDir = QStringLiteral(KDENLIVE_GPUFX_MODULE_DIR);
    if (!repository || moduleDir.isEmpty() || mlt_properties_get_data(mlt_repository_filters(repository), "gpufx", nullptr)) {
        return;
    }
    QStringList dirs;
    if (QDir::isAbsolutePath(moduleDir)) {
        dirs << moduleDir;
    } else {
        dirs << QDir::cleanPath(QCoreApplication::applicationDirPath() + QStringLiteral("/../") + moduleDir);
        dirs << QDir::cleanPath(QStringLiteral(KDENLIVE_INSTALL_PREFIX) + QLatin1Char('/') + moduleDir);
    }
    const QStringList names = {QStringLiteral("libmltgpufx.so"), QStringLiteral("libmltgpufx.dll"), QStringLiteral("mltgpufx.dll")};
    for (const QString &dir : std::as_const(dirs)) {
        for (const QString &name : names) {
            const QString path = QDir(dir).absoluteFilePath(name);
            if (!QFile::exists(path)) {
                continue;
            }
            QLibrary module(path);
            auto registerModule = reinterpret_cast<void (*)(mlt_repository)>(module.resolve("mlt_register"));
            if (registerModule) {
                registerModule(repository);
                qDebug() << "Loaded the GPU effects module" << path;
                return;
            }
            qWarning() << "Cannot load the GPU effects module" << path << module.errorString();
        }
    }
}
