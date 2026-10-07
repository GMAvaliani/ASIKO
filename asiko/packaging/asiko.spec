# -*- mode: python ; coding: utf-8 -*-
# Сборка: python packaging/build.py  (вызывает PyInstaller с этим файлом)
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

ROOT = Path(SPECPATH).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from asiko import __version__  # noqa: E402

datas = [(str(ROOT / "src" / "asiko" / "ui" / "icon.png"), "asiko/ui")]
datas += collect_data_files("certifi")
datas += collect_data_files("librosa", excludes=["**/__pycache__"])
for pkg in ("librosa", "numba", "llvmlite", "anthropic", "keyring", "soundfile", "sounddevice", "soxr", "scipy",
            "numpy", "pyloudnorm", "platformdirs", "PySide6"):
    try:
        datas += copy_metadata(pkg)
    except Exception:
        pass

hiddenimports = collect_submodules("asiko") + collect_submodules("keyring.backends") + [
    "PySide6.QtMultimedia", "PySide6.QtMultimediaWidgets", "pyloudnorm", "soxr",
]
excludes = [
    "tkinter", "matplotlib", "IPython", "pytest", "sklearn", "scikit-learn", "pandas",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtWebEngineQuick", "PySide6.QtWebView",
    "PySide6.Qt3DCore", "PySide6.Qt3DRender", "PySide6.Qt3DExtras", "PySide6.Qt3DInput", "PySide6.Qt3DLogic",
    "PySide6.Qt3DAnimation", "PySide6.QtCharts", "PySide6.QtDataVisualization", "PySide6.QtGraphs",
    "PySide6.QtBluetooth", "PySide6.QtNfc", "PySide6.QtSensors", "PySide6.QtSerialPort", "PySide6.QtSerialBus",
    "PySide6.QtPositioning", "PySide6.QtLocation", "PySide6.QtQuick3D", "PySide6.QtRemoteObjects",
    "PySide6.QtScxml", "PySide6.QtSql", "PySide6.QtTest", "PySide6.QtTextToSpeech", "PySide6.QtPdf",
    "PySide6.QtPdfWidgets", "PySide6.QtDesigner", "PySide6.QtHelp", "PySide6.QtHttpServer",
    "PySide6.QtSpatialAudio", "PySide6.QtWebSockets", "PySide6.QtWebChannel",
]

binaries = []
if sys.platform.startswith("linux"):
    for cand in ("/lib/x86_64-linux-gnu/libportaudio.so.2", "/usr/lib/x86_64-linux-gnu/libportaudio.so.2",
                 "/usr/lib64/libportaudio.so.2"):
        if Path(cand).exists():
            binaries.append((cand, "."))
            break

a = Analysis(
    [str(ROOT / "packaging" / "entry.py")],
    pathex=[str(ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure)
icon = str(ROOT / "packaging" / {"win32": "icon.ico", "darwin": "icon.icns"}.get(sys.platform, "icon.png"))
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ASIKO",
    debug=False,
    strip=False,
    upx=False,
    console=False,
    icon=icon,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ASIKO")
if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="ASIKO.app",
        icon=icon,
        bundle_identifier="app.asiko.desktop",
        version=__version__,
        info_plist={
            "CFBundleName": "ASIKO",
            "CFBundleDisplayName": "ASIKO",
            "CFBundleShortVersionString": __version__,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "12.0",
            "CFBundleDocumentTypes": [{"CFBundleTypeName": "Проект ASIKO", "CFBundleTypeExtensions": ["asiko"],
                                       "CFBundleTypeRole": "Editor"}],
        },
    )
