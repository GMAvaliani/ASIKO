"""Главное окно ASIKO."""
from __future__ import annotations

import math
import time
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QTimer
from PySide6.QtGui import QAction, QKeySequence
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolBar,
    QWidget,
)

from .. import APP_NAME, __version__
from ..audio.io import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from ..commands.schema import op
from ..model.automation import seconds
from ..storage.project_io import collect_project
from . import style as S
from .controller import Controller
from .dialogs import (
    ExportDialog,
    HelpDialog,
    MusicWizard,
    RelinkDialog,
    SettingsDialog,
    SpaceWizard,
    check_sources_job,
    export_report_text,
    open_folder,
)
from .inspector import Inspector, make_spin
from .panels import AUDIO_FILTER, CommandPanel, HistoryPanel, LibraryPanel, MeterWidget
from .timeline import TimelinePanel, fmt_time
from .video import VideoPanel
from .workers import run_job

PROJECT_FILTER = "Проект ASIKO (*.asiko)"


class MainWindow(QMainWindow):
    def __init__(self, ctrl: Controller | None = None):
        super().__init__()
        self.ctrl = ctrl or Controller(self)
        self.setWindowTitle(APP_NAME)
        self.resize(1500, 920)
        self.setAcceptDrops(True)
        self.messages: list[tuple[str, str]] = []
        self.last_export = None
        self._build_ui()
        self._build_actions()
        self.ctrl.projectChanged.connect(self._on_project)
        self.ctrl.selectionChanged.connect(self._on_selection)
        self.ctrl.previewProgress.connect(self._on_preview_progress)
        self.ctrl.previewDone.connect(self._on_preview_done)
        self.ctrl.message.connect(self.show_message)
        self.ctrl.pathChanged.connect(self._update_title)
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self._tick)
        self.timer.start()
        self._refresh_all()

    # ---------------------------------------------------------------- layout
    def _build_ui(self):
        ctrl = self.ctrl
        self.timeline = TimelinePanel(ctrl)
        self.timeline.canvas.seekRequested.connect(self.seek)
        self.timeline.canvas.contextRequested.connect(self._context)
        self.setCentralWidget(self.timeline)
        # сценарии
        tb = QToolBar("Сценарии")
        tb.setObjectName("scenarios")
        tb.setMovable(False)
        tb.setIconSize(QSize(16, 16))
        self.btn_space = QPushButton("Звук в сцене → закадровый")
        self.btn_space.setObjectName("scenario")
        self.btn_space.setToolTip("Сценарий: источник в сцене (радио, телефон…) плавно становится закадровым")
        self.btn_space.clicked.connect(self.scenario_space)
        self.btn_music = QPushButton("Переход A → B")
        self.btn_music.setObjectName("scenario")
        self.btn_music.setToolTip("Сценарий: переход между двумя музыкальными треками с вариантами")
        self.btn_music.clicked.connect(self.scenario_music)
        tb.addWidget(self.btn_space)
        tb.addWidget(self.btn_music)
        tb.addSeparator()
        self.btn_import = QPushButton("Импорт аудио…")
        self.btn_import.clicked.connect(self.import_audio)
        self.btn_export = QPushButton("Экспорт WAV…")
        self.btn_export.setObjectName("primary")
        self.btn_export.clicked.connect(self.export_dialog)
        tb.addWidget(self.btn_import)
        tb.addWidget(self.btn_export)
        self._tb_scen = tb
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tb)
        self.addToolBarBreak()
        # транспорт
        tr = QToolBar("Транспорт")
        tr.setObjectName("transport")
        tr.setMovable(False)
        self.b_home = QPushButton("⏮")
        self.b_home.setToolTip("В начало (Home)")
        self.b_home.clicked.connect(lambda: self.seek(0.0))
        self.b_play = QPushButton("▶ Воспр.")
        self.b_play.setToolTip("Воспроизведение / пауза (Пробел)")
        self.b_play.clicked.connect(self.toggle_play)
        self.b_stop = QPushButton("■ Стоп")
        self.b_stop.clicked.connect(self.stop)
        self.b_loop = QPushButton("⟲ Петля")
        self.b_loop.setCheckable(True)
        self.b_loop.setToolTip("Зациклить выделение (L)")
        self.b_loop.toggled.connect(self._loop)
        self.time_label = QLabel("0:00.00")
        self.time_label.setMinimumWidth(90)
        self.time_label.setStyleSheet("font-family: monospace; font-size: 15px; padding: 0 8px;")
        self.b_hear = QPushButton("Прослушать переход")
        self.b_hear.setToolTip("Воспроизвести выбранный переход с контекстом до и после")
        self.b_hear.clicked.connect(self.play_selected_transition)
        self.ctx = make_spin(0, 30, 0.5, 1, " с", "Контекст до и после перехода")
        self.ctx.setValue(3.0)
        self.compare = QComboBox()
        self.compare.setToolTip("Что слушать: результат, исходник без обработки переходов или вариант перехода A → B")
        self.compare.setMinimumWidth(230)
        self.compare.activated.connect(self._compare_changed)
        self.lmatch = QCheckBox("Выровнять громкость")
        self.lmatch.setToolTip("Только для мониторинга при сравнении (BS.1770). В экспорт не попадает.")
        self.lmatch.toggled.connect(self._lmatch)
        self.volume = QSlider(Qt.Orientation.Horizontal)
        self.volume.setRange(0, 100)
        self.volume.setValue(80)
        self.volume.setFixedWidth(90)
        self.volume.setToolTip("Громкость прослушивания (не влияет на экспорт)")
        self.volume.valueChanged.connect(lambda v: setattr(self.ctrl.playback, "volume", (v / 100) ** 2))
        self.ctrl.playback.volume = 0.64
        self.meters = MeterWidget(ctrl)
        for w in (self.b_home, self.b_play, self.b_stop, self.b_loop, self.time_label):
            tr.addWidget(w)
        tr.addSeparator()
        tr.addWidget(self.b_hear)
        tr.addWidget(QLabel(" ±"))
        tr.addWidget(self.ctx)
        tr.addSeparator()
        tr.addWidget(QLabel(" Сравнение: "))
        tr.addWidget(self.compare)
        tr.addWidget(self.lmatch)
        self.addToolBar(Qt.ToolBarArea.TopToolBarArea, tr)
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._tb_scen.addWidget(spacer)
        self._tb_scen.addWidget(QLabel("Прослушивание "))
        self._tb_scen.addWidget(self.volume)
        self._tb_scen.addWidget(QLabel("  Пики L/R "))
        self._tb_scen.addWidget(self.meters)
        # доки
        self.library = LibraryPanel(ctrl)
        self.d_lib = self._dock("Библиотека", self.library, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.inspector = Inspector(self)
        self.d_insp = self._dock("Свойства", self.inspector, Qt.DockWidgetArea.RightDockWidgetArea)
        self.command = CommandPanel(ctrl)
        self.d_cmd = self._dock("Текстовое задание", self.command, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.history = HistoryPanel(ctrl)
        self.d_hist = self._dock("История изменений", self.history, Qt.DockWidgetArea.BottomDockWidgetArea)
        self.splitDockWidget(self.d_cmd, self.d_hist, Qt.Orientation.Horizontal)
        self.video = VideoPanel(ctrl)
        self.d_video = self._dock("Видео", self.video, Qt.DockWidgetArea.LeftDockWidgetArea)
        self.d_video.hide()
        self.resizeDocks([self.d_cmd, self.d_hist], [900, 450], Qt.Orientation.Horizontal)
        self.resizeDocks([self.d_cmd], [230], Qt.Orientation.Vertical)
        self.resizeDocks([self.d_lib, self.d_insp], [220, 500], Qt.Orientation.Horizontal)
        # статус
        self.st_project = QLabel()
        self.st_render = QLabel()
        self.st_audio = QLabel()
        self.statusBar().addWidget(self.st_project, 1)
        self.statusBar().addPermanentWidget(self.st_render)
        self.statusBar().addPermanentWidget(self.st_audio)

    def _dock(self, title, widget, area):
        d = QDockWidget(title, self)
        d.setObjectName(title)
        d.setWidget(widget)
        d.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetClosable
                      | QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(area, d)
        return d

    def _act(self, menu, text, slot, shortcut=None, tip=None):
        a = QAction(text, self)
        if shortcut:
            a.setShortcut(QKeySequence(shortcut))
            a.setShortcutContext(Qt.ShortcutContext.WindowShortcut)
        if tip:
            a.setStatusTip(tip)
        a.triggered.connect(slot)
        menu.addAction(a)
        self.addAction(a)
        return a

    def _build_actions(self):
        mb = self.menuBar()
        m = mb.addMenu("Файл")
        self._act(m, "Новый проект", self.new_project, "Ctrl+N")
        self._act(m, "Открыть…", self.open_dialog, "Ctrl+O")
        self.m_recent = m.addMenu("Недавние проекты")
        self.m_recent.aboutToShow.connect(self._fill_recent)
        demo = m.addMenu("Демо-проекты")
        self._act(demo, "Звук в сцене → закадровый", lambda: self.open_demo("space"))
        self._act(demo, "Переход A → B", lambda: self.open_demo("music"))
        m.addSeparator()
        self._act(m, "Сохранить", self.save, "Ctrl+S")
        self._act(m, "Сохранить как…", self.save_as, "Ctrl+Shift+S")
        self._act(m, "Собрать переносимый проект…", self.collect, None,
                  "Скопировать проект и все используемые материалы в одну папку")
        self._act(m, "Найти исходники…", self.relink)
        m.addSeparator()
        self._act(m, "Импорт аудио…", self.import_audio, "Ctrl+I")
        self._act(m, "Подключить видео…", self.import_video)
        self._act(m, "Экспорт WAV…", self.export_dialog, "Ctrl+E")
        m.addSeparator()
        self._act(m, "Выход", self.close, "Ctrl+Q")
        e = mb.addMenu("Правка")
        self.a_undo = self._act(e, "Отменить", self.ctrl.undo, "Ctrl+Z")
        self.a_redo = self._act(e, "Повторить", self.ctrl.redo, "Ctrl+Shift+Z")
        a = QAction(self)
        a.setShortcut(QKeySequence("Ctrl+Y"))
        a.triggered.connect(self.ctrl.redo)
        self.addAction(a)
        e.addSeparator()
        self._act(e, "Удалить выбранный переход", self.delete_selected, "Delete")
        self._act(e, "Снять выделение участка", lambda: self.ctrl.select(time_range=None), "Escape")
        self._act(e, "Защитить выделенный участок (настройки)", self.protect_selection)
        self._act(e, "Зафиксировать выделенный участок в PCM (сэмплы)", lambda: self.ctrl.freeze_selection(self))
        self._act(e, "Снять все защиты участков", self.clear_protections)
        e.addSeparator()
        self._act(e, "Ограничение: не менять темп и высоту тона",
                  lambda: self.ctrl.ui_ops([op("constraint.set", params={"keep_tempo": True, "keep_pitch": True})]))
        sc = mb.addMenu("Сценарии")
        self._act(sc, "Звук в сцене → закадровый…", self.scenario_space, "Ctrl+1")
        self._act(sc, "Переход A → B…", self.scenario_music, "Ctrl+2")
        p = mb.addMenu("Воспроизведение")
        self._act(p, "Воспроизведение / пауза", self.toggle_play, "Space")
        self._act(p, "Стоп", self.stop)
        self._act(p, "В начало", lambda: self.seek(0.0), "Home")
        self._act(p, "Петля по выделению", lambda: self.b_loop.toggle(), "L")
        self._act(p, "Прослушать выбранный переход", self.play_selected_transition, "Ctrl+P")
        self._act(p, "Назад на 5 с", lambda: self.seek(max(0.0, self.ctrl.playback.position - 5)), "Left")
        self._act(p, "Вперёд на 5 с", lambda: self.seek(self.ctrl.playback.position + 5), "Right")
        v = mb.addMenu("Вид")
        self._act(v, "Увеличить масштаб", lambda: self.timeline.canvas.set_zoom(self.timeline.canvas.px_per_s * 1.5), "Ctrl+=")
        self._act(v, "Уменьшить масштаб", lambda: self.timeline.canvas.set_zoom(self.timeline.canvas.px_per_s / 1.5), "Ctrl+-")
        self._act(v, "Весь проект", lambda: self.timeline.canvas.zoom_fit(), "Ctrl+0")
        self._act(v, "Показать выбранный переход", self.zoom_selected, "Ctrl+F")
        v.addSeparator()
        for d in (self.d_lib, self.d_insp, self.d_cmd, self.d_hist, self.d_video):
            v.addAction(d.toggleViewAction())
        t = mb.addMenu("Дорожка")
        self._act(t, "Добавить дорожку", lambda: self.ctrl.ui_ops([op("track.add")]))
        self._act(t, "Mute выбранной дорожки", lambda: self._track_toggle("mute"), "M")
        self._act(t, "Solo выбранной дорожки", lambda: self._track_toggle("solo"), "S")
        s = mb.addMenu("Настройки")
        self._act(s, "Внешняя модель и аудио…", self.settings_dialog, "Ctrl+,")
        h = mb.addMenu("Справка")
        self._act(h, "Краткая инструкция", lambda: HelpDialog(self).exec(), "F1")
        self._act(h, "О программе", self.about)

    # ---------------------------------------------------------------- refresh
    def _refresh_all(self):
        self.library.refresh()
        self.timeline.refresh()
        self.inspector.refresh()
        self.history.refresh()
        self.video.refresh()
        self._fill_compare()
        self._update_title()
        if self.ctrl.project.video is not None and not self.d_video.isVisible():
            self.d_video.show()

    def _on_project(self, result):
        self._refresh_all()
        if result.applied and result.kind not in ("reset",) and result.summary:
            self.statusBar().showMessage(" ".join(result.summary)[:300], 8000)

    def _on_selection(self):
        self.timeline.canvas.update()
        self.timeline.refresh()
        self.inspector.refresh()
        self._fill_compare()

    def _fill_compare(self):
        modes = self.ctrl.compare_modes()
        cur = self.ctrl.compare_mode
        self.compare.blockSignals(True)
        self.compare.clear()
        for k, lab in modes:
            self.compare.addItem(lab, k)
        i = self.compare.findData(cur)
        if i < 0:
            i = 0
            if cur != "result":
                self.ctrl.set_compare("result")
        self.compare.setCurrentIndex(i)
        self.compare.blockSignals(False)

    def _update_title(self):
        c = self.ctrl
        name = c.path.name if c.path else "без имени"
        self.setWindowTitle(f"{APP_NAME} — {name}{' *' if c.engine.dirty else ''}")
        self.st_project.setText(c.status_text())

    def _tick(self):
        pb = self.ctrl.playback
        pos = pb.position
        self.time_label.setText(fmt_time(pos, 0.01))
        self.b_play.setText("⏸ Пауза" if pb.playing else "▶ Воспр.")
        self.meters.tick()
        if pb.playing:
            if self.timeline.canvas.follow:
                cv = self.timeline.canvas
                if pos > cv.offset + cv.visible_seconds() * 0.95 or pos < cv.offset:
                    cv.offset = max(0.0, pos - cv.visible_seconds() * 0.05)
                    self.timeline._sync_bar()
            self.timeline.canvas.update()
        self.video.sync(pos, pb.playing)
        if pb.backend == "device":
            self.st_audio.setText(f"Аудио: {pb.device_name[:28]} · {pb.sr} Гц")
        else:
            self.st_audio.setText("Аудио: нет устройства (без звука)")
        if self.ctrl.preview_current("result"):
            pk = self.ctrl.master_peak
            pk_db = 20 * math.log10(pk) if pk > 0 else -120
            warn = " · ПЕРЕГРУЗКА" if pk >= 1.0 else ""
            self.st_render.setText(f"Предпрослушивание готово · пик {pk_db:+.1f} дБFS{warn}")
        self._update_title()

    def _on_preview_progress(self, name, frac):
        if name == "result":
            self.st_render.setText(f"Рендер предпрослушивания: {frac * 100:.0f}%")

    def _on_preview_done(self, name, info):
        self.timeline.canvas.update()
        if self.ctrl.boundary_report:
            worst = max(self.ctrl.boundary_report, key=lambda b: b.mismatch_db)
            if worst.mismatch_db > -60:
                self.show_message("warn", f"Граница зафиксированного PCM-участка ({seconds(worst.at)}): живой рендер "
                                          f"отличается на {worst.mismatch_db:.1f} дБ — на стыке применено 20 мс сшивки.")

    # ---------------------------------------------------------------- messages
    def show_message(self, level: str, text: str):
        self.messages.append((level, text))
        if level == "error":
            QMessageBox.warning(self, APP_NAME, text) if not getattr(self, "_quiet", False) else None
        self.statusBar().showMessage(text[:300], 10000)

    # ---------------------------------------------------------------- transport
    def toggle_play(self):
        pb = self.ctrl.playback
        if pb.playing:
            pb.pause()
        else:
            if pb.loop and not (pb.loop[0] <= pb.pos < pb.loop[1]):
                pb.pos = pb.loop[0]
            pb.play()

    def stop(self):
        self.ctrl.playback.stop()
        self.timeline.canvas.update()

    def seek(self, t: float):
        self.ctrl.playback.seek(t)
        self.timeline.canvas.update()
        self.video.sync(t, self.ctrl.playback.playing, force=True)

    def _loop(self, on):
        self.ctrl.toggle_loop(on)
        if on and self.ctrl.playback.loop is None:
            self.b_loop.blockSignals(True)
            self.b_loop.setChecked(False)
            self.b_loop.blockSignals(False)
        self.timeline.canvas.update()

    def play_selected_transition(self, wait_render: bool = False):
        ok = self.ctrl.play_transition(context=self.ctx.value())
        if ok:
            x = self.ctrl.project.any_transition(self.ctrl.selection.transition_id)
            a, b = self.ctrl.transition_bounds(x)
            cv = self.timeline.canvas
            if not (cv.offset <= a and b <= cv.offset + cv.visible_seconds()):
                cv.show_range(max(0.0, a - self.ctx.value()), b + self.ctx.value())
                self.timeline._sync_bar()

    def _compare_changed(self, _i):
        mode = self.compare.currentData()
        self.ctrl.set_compare(mode)

    def _lmatch(self, on):
        self.ctrl.playback.loudness_match = bool(on)
        if on:
            self.ctrl.update_loudness_match()
            g = self.ctrl.playback.monitor_gain_db
            if g:
                self.statusBar().showMessage("Выравнивание громкости (мониторинг): " + ", ".join(
                    f"{k.split(':')[0]} {v:+.1f} дБ" for k, v in g.items()), 8000)

    def start_compare(self):
        """Сравнение вариантов: петля по зоне перехода, включение переключателя."""
        if not self.b_loop.isChecked():
            self.b_loop.setChecked(True)
        else:
            self.ctrl.toggle_loop(True)
        self.ctrl.prepare_compare()
        self._fill_compare()
        self.compare.setFocus()
        if not self.ctrl.playback.playing:
            self.toggle_play()
        self.show_message("info", "Сравнение: переключайте «Результат / Исходник / Вариант N» в панели транспорта — "
                                  "позиция сохраняется. «Выровнять громкость» влияет только на прослушивание.")

    # ---------------------------------------------------------------- scenarios
    def scenario_space(self):
        if not self.ctrl.project.all_clips():
            self.show_message("info", "Сначала импортируйте музыку: перетащите файл на шкалу или «Импорт аудио…».")
            self.import_audio()
            return
        SpaceWizard(self).exec()

    def scenario_music(self):
        MusicWizard(self).exec()

    # ---------------------------------------------------------------- files
    def _maybe_save(self) -> bool:
        if not self.ctrl.engine.dirty:
            return True
        ans = QMessageBox.question(self, APP_NAME, "Сохранить изменения в проекте?",
                                   QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard
                                   | QMessageBox.StandardButton.Cancel)
        if ans == QMessageBox.StandardButton.Save:
            return self.save()
        if ans == QMessageBox.StandardButton.Discard:
            self.ctrl.autosaver.clear(self.ctrl.project.id)
            return True
        return False

    def new_project(self):
        if self._maybe_save():
            self.ctrl.new_project()

    def open_dialog(self):
        if not self._maybe_save():
            return
        p, _ = QFileDialog.getOpenFileName(self, "Открыть проект", str(self.ctrl.path.parent) if self.ctrl.path else "",
                                           PROJECT_FILTER)
        if p:
            self.open_path(p)

    def open_path(self, p: str, recovered_from: str | None = None, check: bool = True):
        try:
            self.ctrl.open_project(p, recovered_from=recovered_from)
        except Exception as e:
            self.show_message("error", f"Не удалось открыть проект: {e}")
            return False
        self.timeline.canvas.zoom_fit()
        if check:
            check_sources_job(self, self._after_check)
        return True

    def _after_check(self, statuses):
        bad = [s for s in statuses if s.status != "ok"]
        for s in bad:
            if s.status == "missing":
                self.ctrl.store.missing.add(s.source_id)
        self.library.refresh()
        self.timeline.canvas.update()
        if bad:
            RelinkDialog(self, statuses).exec()

    def relink(self):
        check_sources_job(self, lambda st: RelinkDialog(self, st).exec() if any(s.status != "ok" for s in st)
                          else self.show_message("info", "Все исходники на месте, контрольные суммы совпадают."))

    def save(self) -> bool:
        if self.ctrl.path is None:
            return self.save_as()
        try:
            self.ctrl.save()
        except Exception as e:
            self.show_message("error", f"Не удалось сохранить: {e}")
            return False
        self.statusBar().showMessage(f"Сохранено: {self.ctrl.path}", 5000)
        self._update_title()
        return True

    def save_as(self) -> bool:
        p, _ = QFileDialog.getSaveFileName(self, "Сохранить проект", str(self.ctrl.path or (Path.home() / f"{self.ctrl.project.name}.asiko")),
                                           PROJECT_FILTER)
        if not p:
            return False
        try:
            self.ctrl.save(p)
        except Exception as e:
            self.show_message("error", f"Не удалось сохранить: {e}")
            return False
        self._update_title()
        return True

    def collect(self):
        d = QFileDialog.getExistingDirectory(self, "Папка для переносимого проекта (лучше пустая)")
        if not d:
            return
        project = self.ctrl.project.clone()
        path = self.ctrl.path

        def done(out):
            QMessageBox.information(self, APP_NAME, f"Переносимый проект собран:\n{out}\nМатериалы скопированы в "
                                                    f"папку media и проверены по контрольным суммам.")

        run_job(self, "Сборка переносимого проекта…", lambda pr, c: collect_project(project, path, d, cancel=c, progress=pr),
                on_done=done, on_error=lambda m: self.show_message("error", f"Сборка не выполнена: {m}"))

    def _fill_recent(self):
        self.m_recent.clear()
        rec = self.ctrl.recent()
        if not rec:
            a = self.m_recent.addAction("(нет)")
            a.setEnabled(False)
        for r in rec:
            self.m_recent.addAction(r, lambda r=r: self._maybe_save() and self.open_path(r))

    def open_demo(self, kind: str):
        if not self._maybe_save():
            return
        from ..demo import ensure_demo

        def done(paths):
            self.open_path(str(paths[kind]))

        run_job(self, "Подготовка демонстрационных материалов…", lambda p, c: ensure_demo(progress=p), on_done=done,
                on_error=lambda m: self.show_message("error", m))

    def import_audio(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Импорт аудио", "", AUDIO_FILTER)
        if paths:
            self.ctrl.import_files(self, paths, track_id=None, at=0.0)

    def import_video(self):
        self.video.open_dialog()
        self.d_video.show()

    def handle_dropped_other(self, paths):
        vids = [p for p in paths if Path(p).suffix.lower() in VIDEO_EXTENSIONS]
        projs = [p for p in paths if p.lower().endswith(".asiko")]
        if projs and self._maybe_save():
            self.open_path(projs[0])
        if vids:
            self.ctrl.ui_ops([op("video.set", params={"path": vids[0], "offset": 0.0, "name": Path(vids[0]).name})])
            self.d_video.show()
        rest = [p for p in paths if p not in vids and p not in projs]
        if rest:
            self.show_message("warn", "Неподдерживаемые файлы: " + ", ".join(Path(p).name for p in rest))

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        audio = [p for p in paths if Path(p).suffix.lower() in AUDIO_EXTENSIONS]
        if audio:
            self.ctrl.import_files(self, audio)
        other = [p for p in paths if p not in audio]
        if other:
            self.handle_dropped_other(other)

    def export_dialog(self):
        if not self.ctrl.project.all_clips():
            self.show_message("info", "Нечего экспортировать: на шкале нет клипов.")
            return
        ExportDialog(self).exec()

    def show_export_report(self, rep):
        self.last_export = rep
        box = QMessageBox(self)
        box.setWindowTitle("Экспорт завершён")
        box.setText("Файл записан и проверен.")
        box.setDetailedText(export_report_text(rep))
        box.setInformativeText(export_report_text(rep).split("\nSHA")[0])
        b = box.addButton("Открыть папку", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        if not getattr(self, "_quiet", False):
            box.exec()
            if box.clickedButton() is b:
                open_folder(rep.path)

    def settings_dialog(self):
        SettingsDialog(self).exec()

    def about(self):
        QMessageBox.about(self, f"О программе {APP_NAME}",
                          f"<b>{APP_NAME} {__version__}</b><br>Подготовка аудио и переходов для видеоигр по "
                          f"текстовому заданию.<br><br>Обработка локальная: PySide6 (Qt, LGPL), numpy/scipy, "
                          f"libsndfile, libsoxr, librosa, PortAudio. Лицензии — в папке licenses дистрибутива.")

    # ---------------------------------------------------------------- edit
    def delete_selected(self):
        sel = self.ctrl.selection
        p = self.ctrl.project
        if sel.transition_id and p.space_transition(sel.transition_id):
            self.ctrl.ui_ops([op("space.delete", target={"transition_id": sel.transition_id})])
        elif sel.transition_id and p.music_transition(sel.transition_id):
            self.ctrl.ui_ops([op("music.delete", target={"transition_id": sel.transition_id})])
        elif sel.clip_ids:
            self.ctrl.ui_ops([op("clip.delete", target={"clip_id": c}) for c in sel.clip_ids])

    def protect_selection(self):
        r = self.ctrl.selection.time_range
        if not r:
            self.show_message("info", "Выделите участок на линейке.")
            return
        self.ctrl.ui_ops([op("protect.add", params={"start": r[0], "end": r[1]})], text="Защита участка")

    def clear_protections(self):
        ops = [op("protect.remove", target={"protection_id": p.id}) for p in self.ctrl.project.protections]
        if ops:
            self.ctrl.ui_ops(ops, text="Снятие защиты участков")

    def zoom_selected(self):
        x = self.ctrl.project.any_transition(self.ctrl.selection.transition_id) if self.ctrl.selection.transition_id else None
        if x is not None:
            a, b = self.ctrl.transition_bounds(x)
            self.timeline.canvas.show_range(max(0.0, a - 3), b + 3)
            self.timeline._sync_bar()

    def _track_toggle(self, what):
        t = self.ctrl.project.track(self.ctrl.selection.track_id) if self.ctrl.selection.track_id else None
        if t is not None:
            self.ctrl.ui_ops([op("track.set", target={"track_id": t.id}, params={what: not getattr(t, what)})])

    def _context(self, kind, oid, gpos, t):
        if kind.startswith("open:"):
            self.inspector.refresh()
            self.d_insp.show()
            self.d_insp.raise_()
            return
        menu = QMenu(self)
        p = self.ctrl.project
        if kind == "clip":
            c = p.clip(oid)
            self.ctrl.select(clip_ids=[oid] if oid not in self.ctrl.selection.clip_ids else self.ctrl.selection.clip_ids,
                             track_id=p.clip_track(oid).id)
            menu.addAction("Звук в сцене → закадровый…", self.scenario_space)
            menu.addAction("Переход A → B…", self.scenario_music)
            menu.addAction("Анализ темпа и долей", lambda: self.ctrl.analyze_sources(self, [c.source_id], force=True))
            menu.addAction("Снять фиксацию" if c.locked else "Зафиксировать клип",
                           lambda: self.ctrl.ui_ops([op("clip.set", target={"clip_id": oid}, params={"locked": not c.locked})]))
            menu.addAction("Удалить клип", lambda: self.ctrl.ui_ops([op("clip.delete", target={"clip_id": oid})]))
        elif kind in ("x_body", "x_start", "x_end", "music", "switch"):
            x = p.any_transition(oid)
            self.ctrl.select(transition_id=oid)
            menu.addAction("Прослушать с контекстом", self.play_selected_transition)
            menu.addAction("Снять утверждение" if x.approved else "Утвердить",
                           lambda: self.ctrl.ui_ops([op("transition.approve", target={"transition_id": oid},
                                                        params={"approved": not x.approved})]))
            menu.addAction("Удалить", self.delete_selected)
        elif kind in ("lane", "empty", "ruler"):
            menu.addAction("Поставить курсор сюда", lambda: self.seek(t))
            menu.addAction("Импорт аудио…", self.import_audio)
            if self.ctrl.selection.time_range:
                menu.addAction("Защитить выделенный участок", self.protect_selection)
                menu.addAction("Зафиксировать выделенный участок в PCM", lambda: self.ctrl.freeze_selection(self))
        if menu.actions():
            menu.exec(gpos)

    # ---------------------------------------------------------------- lifecycle
    def closeEvent(self, ev):
        if not self._maybe_save():
            ev.ignore()
            return
        self.ctrl.shutdown()
        ev.accept()

    def check_recovery(self):
        ents = self.ctrl.autosaver.entries()
        if not ents:
            return
        e = ents[0]
        when = time.strftime("%d.%m %H:%M", time.localtime(e.saved_at))
        ans = QMessageBox.question(self, "Восстановление", f"Найдено автосохранение проекта «{e.name}» от {when}"
                                   + (f"\n(файл: {e.original})" if e.original else " (проект не был сохранён)")
                                   + ".\nВосстановить?")
        if ans == QMessageBox.StandardButton.Yes:
            self.open_path(str(e.path), recovered_from=e.original)
        else:
            self.ctrl.autosaver.clear(e.project_id)
