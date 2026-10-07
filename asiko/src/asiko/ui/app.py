"""Запуск приложения."""
from __future__ import annotations

import os
import sys

from .. import APP_NAME, __version__, app_paths


def _setup_streams() -> None:
    # В оконной сборке (PyInstaller --windowed) stdout/stderr отсутствуют — пишем в журнал.
    if sys.stdout is None or sys.stderr is None:
        log = open(app_paths.log_dir() / "asiko.log", "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = log
        if sys.stderr is None:
            sys.stderr = log


def run_gui(argv: list[str]) -> int:
    _setup_streams()
    app_paths.setup_numba_cache()
    from PySide6.QtCore import QTimer
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication

    from . import style
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(argv[:1])
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    app.setOrganizationName(APP_NAME)
    style.apply(app)
    from pathlib import Path

    icon = Path(__file__).with_name("icon.png")
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))
    win = MainWindow()
    win.show()
    args = [a for a in argv[1:] if not a.startswith("-")]
    if args and os.path.exists(args[0]):
        QTimer.singleShot(0, lambda: win.open_path(args[0]))
    elif "--demo" in argv:
        QTimer.singleShot(0, lambda: win.open_demo("space"))
    else:
        QTimer.singleShot(200, win.check_recovery)
    return app.exec()
