"""Необязательный синхронный просмотр видео (QtMultimedia). Звук видео не используется."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import QFileDialog, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..audio.io import VIDEO_EXTENSIONS
from ..commands.schema import op
from .inspector import make_spin

SEEK_TOLERANCE_S = 0.06


class VideoPanel(QWidget):
    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.player = None
        self.widget = None
        self.error = ""
        self.loaded_path = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(4, 4, 4, 4)
        self.holder = QVBoxLayout()
        lay.addLayout(self.holder, 1)
        self.status = QLabel("Видео не подключено. Работа с аудио не требует видео.")
        self.status.setWordWrap(True)
        self.status.setObjectName("hint")
        lay.addWidget(self.status)
        row = QHBoxLayout()
        self.btn_open = QPushButton("Открыть видео…")
        self.btn_open.clicked.connect(self.open_dialog)
        self.btn_clear = QPushButton("Отключить")
        self.btn_clear.clicked.connect(lambda: self.ctrl.ui_ops([op("video.clear")]))
        self.offset = make_spin(-3600, 3600, 0.04, 3, " с", "Время шкалы, на котором начинается видео")
        self.offset.valueChanged.connect(self._offset)
        row.addWidget(self.btn_open)
        row.addWidget(self.btn_clear)
        row.addWidget(QLabel("Начало на шкале"))
        row.addWidget(self.offset)
        lay.addLayout(row)

    def _ensure_player(self) -> bool:
        if self.player is not None:
            return True
        try:
            from PySide6.QtMultimedia import QMediaPlayer
            from PySide6.QtMultimediaWidgets import QVideoWidget
        except ImportError as e:
            self.error = f"Модуль видео недоступен: {e}"
            self.status.setText(self.error)
            return False
        self.player = QMediaPlayer(self)
        self.widget = QVideoWidget()
        self.widget.setMinimumSize(320, 180)
        self.player.setVideoOutput(self.widget)
        self.holder.addWidget(self.widget)
        self.player.errorOccurred.connect(self._on_error)
        self.player.mediaStatusChanged.connect(self._on_status)
        return True

    def open_dialog(self):
        exts = " ".join("*" + e for e in VIDEO_EXTENSIONS)
        p, _ = QFileDialog.getOpenFileName(self, "Видео для просмотра", "", f"Видео ({exts});;Все файлы (*)")
        if p:
            self.ctrl.ui_ops([op("video.set", params={"path": p, "offset": 0.0, "name": Path(p).name})])

    def refresh(self):
        v = self.ctrl.project.video
        self.btn_clear.setEnabled(v is not None)
        self.offset.setEnabled(v is not None)
        if v is None:
            if self.player is not None and self.loaded_path is not None:
                self.player.stop()
                self.player.setSource(QUrl())
            self.loaded_path = None
            self.status.setText("Видео не подключено. Работа с аудио не требует видео.")
            return
        self.offset.blockSignals(True)
        self.offset.setValue(v.offset)
        self.offset.blockSignals(False)
        if v.path != self.loaded_path:
            if not Path(v.path).exists():
                self.status.setText(f"Видео не найдено: {v.path}")
                return
            if not self._ensure_player():
                return
            self.error = ""
            self.loaded_path = v.path
            self.player.setSource(QUrl.fromLocalFile(v.path))
            self.status.setText(f"Загрузка «{v.name}»…")

    def _offset(self, val):
        if self.ctrl.project.video is not None:
            self.ctrl.ui_ops([op("video.offset", params={"offset": float(val)})])

    def _on_error(self, err, msg=""):
        self.error = msg or self.player.errorString()
        self.status.setText(f"Не удалось воспроизвести видео: {self.error}. Набор поддерживаемых кодеков зависит "
                            f"от ОС; работа с аудио продолжается.")

    def _on_status(self, st):
        from PySide6.QtMultimedia import QMediaMetaData, QMediaPlayer

        if st == QMediaPlayer.MediaStatus.LoadedMedia:
            md = self.player.metaData()
            fps = md.value(QMediaMetaData.Key.VideoFrameRate)
            dur = self.player.duration() / 1000.0
            v = self.ctrl.project.video
            self.status.setText(f"«{v.name if v else ''}»: {dur:.2f} с" + (f", {fps:.3f} кадр/с" if fps else "")
                                + ". Видео следует за курсором воспроизведения (без звука видео).")
            self.sync(self.ctrl.playback.position, False, force=True)
        elif st == QMediaPlayer.MediaStatus.InvalidMedia:
            self._on_error(None, self.player.errorString() or "неподдерживаемый формат")

    def video_time(self, timeline_t: float) -> float:
        v = self.ctrl.project.video
        return timeline_t - (v.offset if v else 0.0)

    def sync(self, t: float, playing: bool, force: bool = False):
        if self.player is None or self.loaded_path is None or self.error:
            return
        from PySide6.QtMultimedia import QMediaPlayer

        vt = self.video_time(t)
        dur = self.player.duration() / 1000.0
        is_playing = self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        if vt < 0 or (dur > 0 and vt > dur):
            if is_playing:
                self.player.pause()
            return
        pos = self.player.position() / 1000.0
        if playing:
            if not is_playing:
                self.player.setPosition(int(vt * 1000))
                self.player.play()
            elif abs(pos - vt) > SEEK_TOLERANCE_S:
                self.player.setPosition(int(vt * 1000))
        else:
            if self.player.playbackState() != QMediaPlayer.PlaybackState.PausedState:
                # из состояния «стоп» Qt не показывает кадр после перемотки — переводим в паузу
                self.player.pause()
            if force or abs(pos - vt) > 0.02:
                self.player.setPosition(int(vt * 1000))
