"""Фоновые задачи с прогрессом и отменой (интерфейс не блокируется)."""
from __future__ import annotations

import threading
import traceback

from PySide6.QtCore import QObject, QThread, Qt, QTimer, Signal
from PySide6.QtWidgets import QApplication, QProgressDialog

from ..audio.io import Cancelled


class CancelToken:
    def __init__(self):
        self._e = threading.Event()

    def cancel(self) -> None:
        self._e.set()

    def __call__(self) -> bool:
        return self._e.is_set()


class _Signals(QObject):
    progress = Signal(float, str)
    done = Signal(object)
    failed = Signal(str, str)
    cancelled = Signal()


class Job(QThread):
    """fn(progress, cancel) → результат. progress(fraction, message)."""

    def __init__(self, fn, parent=None):
        super().__init__(parent)
        self.fn = fn
        self.token = CancelToken()
        self.sig = _Signals()

    def run(self):
        def progress(f, msg=""):
            self.sig.progress.emit(float(f), str(msg))

        try:
            res = self.fn(progress, self.token)
        except Cancelled:
            self.sig.cancelled.emit()
            return
        except Exception as e:  # сообщение пользователю + трассировка в журнал
            tb = traceback.format_exc()
            print(tb)
            self.sig.failed.emit(str(e) or type(e).__name__, tb)
            return
        if self.token():
            self.sig.cancelled.emit()
        else:
            self.sig.done.emit(res)


_running: set[Job] = set()


def run_job(parent, title: str, fn, on_done=None, on_error=None, on_cancel=None, modal: bool = True,
            cancellable: bool = True, show_after_ms: int = 300) -> Job:
    """Запускает задачу в отдельном потоке с диалогом прогресса и кнопкой «Отмена»."""
    job = Job(fn)
    dlg = None
    if modal:
        dlg = QProgressDialog(title, "Отмена" if cancellable else None, 0, 1000, parent)
        dlg.setWindowTitle("ASIKO")
        dlg.setWindowModality(Qt.WindowModality.WindowModal)
        dlg.setMinimumDuration(show_after_ms)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.setValue(0)
        if cancellable:
            dlg.canceled.connect(job.token.cancel)
        job.dialog = dlg

    def prog(f, msg):
        if dlg is not None:
            dlg.setValue(int(max(0.0, min(1.0, f)) * 1000))
            if msg:
                dlg.setLabelText(f"{title}\n{msg}")

    def finish():
        _running.discard(job)
        if dlg is not None:
            dlg.reset()
            dlg.hide()
            dlg.deleteLater()

    def done(res):
        finish()
        if on_done:
            on_done(res)

    def failed(msg, tb):
        finish()
        if on_error:
            on_error(msg)

    def cancelled():
        finish()
        if on_cancel:
            on_cancel()

    job.sig.progress.connect(prog)
    job.sig.done.connect(done)
    job.sig.failed.connect(failed)
    job.sig.cancelled.connect(cancelled)
    _running.add(job)
    job.start()
    return job


def wait_all(timeout_ms: int = 60000) -> None:
    """Для тестов: ждать завершения фоновых задач, обрабатывая события."""
    import time

    t0 = time.time()
    while _running and (time.time() - t0) * 1000 < timeout_ms:
        QApplication.processEvents()
        time.sleep(0.01)
    QApplication.processEvents()
