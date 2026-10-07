"""Диалоги: сценарии, экспорт, настройки, поиск исходников, справка."""
from __future__ import annotations

import html
import os
from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTabWidget,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..audio.export import SAMPLE_RATES, SUBTYPES, ExportOptions
from ..audio.io import sha256_file
from ..commands.schema import op
from ..model.automation import OVERLAP_FILTERS, seconds
from ..model.presets import BUILTIN_PRESETS, ROOMS, SOURCE_LIMITATION_NOTE, get_preset, preset_names
from ..nlp.llm import DEFAULTS, PROVIDERS, LlmConfig
from ..nlp.local_parser import SUPPORTED_TEMPLATES
from ..storage.project_io import check_sources, find_moved_sources
from . import style as S
from .inspector import make_spin, note_box
from .panels import AUDIO_FILTER
from .workers import run_job


class SpaceWizard(QDialog):
    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.ctrl = win.ctrl
        self.setWindowTitle("Звук в сцене → закадровый")
        self.setMinimumWidth(520)
        lay = QVBoxLayout(self)
        lab = QLabel("Звук звучит из источника в сцене, затем плавно становится закадровым (или наоборот). "
                     "Параметры можно изменить после создания.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        form = QFormLayout()
        self.track = QComboBox()
        proj = self.ctrl.project
        for t in proj.tracks:
            if t.clips:
                self.track.addItem(t.name, t.id)
        sel = self.ctrl.selection.track_id
        if sel:
            i = self.track.findData(sel)
            if i >= 0:
                self.track.setCurrentIndex(i)
        self.direction = QComboBox()
        self.direction.addItem("В сцене → закадровый", "in_out")
        self.direction.addItem("Закадровый → в сцене", "out_in")
        self.preset = QComboBox()
        for k, v in preset_names(self.ctrl.user_presets).items():
            if k != "offscreen":
                self.preset.addItem(v, k)
        self.side = QComboBox()
        for lab_, v in (("Справа", 0.6), ("По центру", 0.0), ("Слева", -0.6), ("Как в пресете", None)):
            self.side.addItem(lab_, v)
        self.room = QComboBox()
        self.room.addItem("Как в пресете", None)
        for k, v in ROOMS.items():
            self.room.addItem(v, k)
        a, b = self._default_range()
        self.start = make_spin(0, 36000, 0.1, 3, " с")
        self.start.setValue(a)
        self.end = make_spin(0, 36000, 0.1, 3, " с")
        self.end.setValue(b)
        self.protect = QCheckBox("После конца перехода не менять (защитить настройки)")
        form.addRow("Дорожка", self.track)
        form.addRow("Направление", self.direction)
        form.addRow("Источник в сцене", self.preset)
        form.addRow("Положение", self.side)
        form.addRow("Комната", self.room)
        form.addRow("Начало перехода", self.start)
        form.addRow("Конец перехода", self.end)
        lay.addLayout(form)
        lay.addWidget(self.protect)
        lay.addWidget(note_box(SOURCE_LIMITATION_NOTE))
        self.err = QLabel()
        self.err.setObjectName("error")
        self.err.setWordWrap(True)
        lay.addWidget(self.err)
        bb = QDialogButtonBox()
        self.btn_play = bb.addButton("Создать и прослушать", QDialogButtonBox.ButtonRole.AcceptRole)
        self.btn_ok = bb.addButton("Создать", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton("Отмена", QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_ok.clicked.connect(lambda: self._create(False))
        self.btn_play.clicked.connect(lambda: self._create(True))
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if self.track.count() == 0:
            self.err.setText("Сначала импортируйте аудио и поместите его на шкалу.")
            self.btn_ok.setEnabled(False)
            self.btn_play.setEnabled(False)

    def _default_range(self):
        sel = self.ctrl.selection.time_range
        if sel and sel[1] - sel[0] >= 0.1:
            return sel
        end = self.ctrl.project.content_end()
        if end >= 20:
            return 12.0, 16.0
        return round(end * 0.35, 2), round(end * 0.55, 2)

    def build_ops(self) -> list[dict]:
        tid = self.track.currentData()
        key = self.preset.currentData()
        vals = {}
        if self.side.currentData() is not None:
            vals["pan"] = self.side.currentData()
        params: dict = {}
        if self.direction.currentData() == "in_out":
            params["from_preset"], params["to_preset"] = key, "offscreen"
            if vals:
                params["from_values"] = vals
            if self.room.currentData():
                params["from_room"] = self.room.currentData()
        else:
            params["from_preset"], params["to_preset"] = "offscreen", key
            if vals:
                params["to_values"] = vals
            if self.room.currentData():
                params["to_room"] = self.room.currentData()
        cons = [{"kind": "protect_after", "time": self.end.value()}] if self.protect.isChecked() else None
        return [op("space.create", target={"track_id": tid},
                   time={"coord": "timeline", "unit": "s", "start": self.start.value(), "end": self.end.value()},
                   params=params, constraints=cons)]

    def _create(self, play: bool):
        r = self.ctrl.ui_ops(self.build_ops(), text="Сценарий «Звук в сцене → закадровый»")
        if not r.ok:
            self.err.setText("\n".join(r.errors))
            return
        self.accept()
        if play:
            self.win.play_selected_transition(wait_render=True)


class MusicWizard(QDialog):
    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.ctrl = win.ctrl
        self.setWindowTitle("Переход A → B")
        self.setMinimumWidth(540)
        lay = QVBoxLayout(self)
        lab = QLabel("Переход между двумя музыкальными клипами. Сначала оцениваются темп и доли, затем "
                     "предлагаются до трёх различающихся вариантов. Темп и высота тона не изменяются.")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        form = QFormLayout()
        self.a = QComboBox()
        self.b = QComboBox()
        clips = sorted(self.ctrl.project.all_clips(), key=lambda c: c.start)
        for c in clips:
            self.a.addItem(f"{c.name} ({seconds(c.start)}–{seconds(c.end)})", c.id)
            self.b.addItem(f"{c.name} ({seconds(c.start)}–{seconds(c.end)})", c.id)
        sel = [c for c in self.ctrl.selection.clip_ids if self.ctrl.project.clip(c)]
        if len(sel) == 2:
            sa, sb = sorted(sel, key=lambda c: self.ctrl.project.clip(c).start)
        elif len(clips) >= 2:
            sa, sb = clips[0].id, clips[1].id
        else:
            sa = sb = None
        if sa:
            self.a.setCurrentIndex(self.a.findData(sa))
            self.b.setCurrentIndex(self.b.findData(sb))
        self.at = make_spin(0, 36000, 0.1, 2, " с")
        self.at.setValue(self._default_time(clips))
        self.downbeat = QCheckBox("B входит на сильную долю")
        self.downbeat.setChecked(True)
        self.tail = QCheckBox("Сохранить хвост A (исходный материал после точки перехода)")
        self.tail.setChecked(True)
        self.exact = QCheckBox("Ровно в заданное время (иначе — ближайшая сильная доля A)")
        self.keep = QCheckBox("Не менять темп и высоту тона")
        self.keep.setChecked(True)
        self.keep.setToolTip("Темп и высота тона в ASIKO никогда не изменяются; при несовпадении темпов "
                             "конфликт будет показан в вариантах.")
        self.filt = QComboBox()
        for k, v in OVERLAP_FILTERS.items():
            self.filt.addItem(v, k)
        form.addRow("Клип A (уходит)", self.a)
        form.addRow("Клип B (входит)", self.b)
        form.addRow("Примерное время перехода", self.at)
        lay.addLayout(form)
        for w in (self.downbeat, self.tail, self.exact, self.keep):
            lay.addWidget(w)
        f2 = QFormLayout()
        f2.addRow("Частоты в зоне наложения", self.filt)
        lay.addLayout(f2)
        self.err = QLabel()
        self.err.setObjectName("error")
        self.err.setWordWrap(True)
        lay.addWidget(self.err)
        imp = QPushButton("Импортировать второй трек…")
        imp.clicked.connect(self._import_b)
        lay.addWidget(imp)
        bb = QDialogButtonBox()
        self.btn_ok = bb.addButton("Анализировать и предложить варианты", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton("Отмена", QDialogButtonBox.ButtonRole.RejectRole)
        self.btn_ok.clicked.connect(self._go)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        if len(clips) < 2:
            self.err.setText("Нужны два клипа на шкале: импортируйте второй трек.")
            self.btn_ok.setEnabled(False)

    def _default_time(self, clips):
        sel = self.ctrl.selection.time_range
        if sel:
            return sel[0]
        if clips:
            a = clips[0]
            return min(40.0, max(1.0, a.end - 8.0)) if a.end < 48 else 40.0
        return 40.0

    def _import_b(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Второй трек", "", AUDIO_FILTER)
        if paths:
            self.reject()
            self.ctrl.import_files(self.win, paths, at=None, on_done=lambda r: self.win.scenario_music())

    def build_ops(self) -> list[dict]:
        cons = [{"kind": "keep_tempo"}, {"kind": "keep_pitch"}] if self.keep.isChecked() else None
        return [op("music.create", target={"clip_a_id": self.a.currentData(), "clip_b_id": self.b.currentData()},
                   time={"coord": "timeline", "unit": "s", "at": self.at.value()},
                   params={"b_on_downbeat": self.downbeat.isChecked(), "keep_a_tail": self.tail.isChecked(),
                           "exact": self.exact.isChecked(), "overlap_filter": self.filt.currentData()},
                   constraints=cons)]

    def _go(self):
        a, b = self.a.currentData(), self.b.currentData()
        if a == b:
            self.err.setText("A и B должны быть разными клипами.")
            return
        ops = self.build_ops()
        p = self.ctrl.project
        sids = list({p.clip(a).source_id, p.clip(b).source_id})
        self.accept()

        def after(ok):
            if ok is None:      # анализ отменён пользователем — переход не создаём
                return
            if ok is False:
                self.win.show_message("warn", "Анализ долей не выполнен — переход создаётся по времени, "
                                              "без разметки долей.")
            r = self.ctrl.ui_ops(ops, text="Сценарий «Переход A → B»")
            if r.ok:
                self.win.show_message("info", " ".join(r.summary))

        self.ctrl.analyze_sources(self.win, sids, on_done=after)


class ExportDialog(QDialog):
    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.ctrl = win.ctrl
        self.setWindowTitle("Экспорт WAV")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        g = QGroupBox("Что экспортировать")
        gl = QVBoxLayout(g)
        self.r_all = QRadioButton("Весь проект")
        self.r_sel = QRadioButton("Выделенный участок")
        self.r_x = QRadioButton("Выбранный переход с контекстом")
        self.r_all.setChecked(True)
        sel = self.ctrl.selection.time_range
        self.r_sel.setEnabled(bool(sel))
        if sel:
            self.r_sel.setText(f"Выделенный участок ({seconds(sel[0])}–{seconds(sel[1])})")
        x = self.ctrl.project.any_transition(self.ctrl.selection.transition_id) if self.ctrl.selection.transition_id else None
        self.r_x.setEnabled(x is not None)
        self.ctx = make_spin(0, 60, 0.5, 1, " с")
        self.ctx.setValue(3.0)
        row = QHBoxLayout()
        row.addWidget(self.r_x)
        row.addWidget(QLabel("±"))
        row.addWidget(self.ctx)
        row.addStretch(1)
        gl.addWidget(self.r_all)
        gl.addWidget(self.r_sel)
        gl.addLayout(row)
        self.tails = QCheckBox("Включить хвосты эффектов (реверберация, фильтры)")
        self.tails.setChecked(True)
        self.tails.setToolTip("Весь проект — до естественного затухания. Участок — источники останавливаются на "
                              "конце участка, эффекты затухают после него.")
        self.edges = QCheckBox("Короткие затухания на краях участка (5 мс, против щелчков)")
        self.edges.setChecked(True)
        gl.addWidget(self.tails)
        gl.addWidget(self.edges)
        lay.addWidget(g)
        g2 = QGroupBox("Формат")
        f = QFormLayout(g2)
        self.fmt = QComboBox()
        for k, v in SUBTYPES.items():
            self.fmt.addItem(v, k)
        self.sr = QComboBox()
        for r in SAMPLE_RATES:
            self.sr.addItem(f"{r} Гц", r)
        self.sr.setCurrentIndex(self.sr.findData(48000))
        self.dither = QCheckBox("TPDF-дизеринг при уменьшении разрядности")
        self.norm = QCheckBox("Нормализовать пик до")
        self.norm_db = make_spin(-24, 0, 0.1, 1, " дБFS")
        self.norm_db.setValue(-1.0)
        nr = QHBoxLayout()
        nr.addWidget(self.norm)
        nr.addWidget(self.norm_db)
        nr.addStretch(1)
        f.addRow("Тип", self.fmt)
        f.addRow("Частота", self.sr)
        f.addRow("", self.dither)
        f.addRow("", nr)
        self.variant = QComboBox()
        self.variant.addItem("Текущие параметры переходов A → B", None)
        for m in self.ctrl.project.music_transitions:
            for i, v in enumerate(m.variants):
                self.variant.addItem(f"{m.name}: вариант {i + 1} — {v.name}", (m.id, v.id))
        if self.variant.count() > 1:
            f.addRow("Вариант A → B", self.variant)
        lay.addWidget(g2)
        pr = QHBoxLayout()
        self.path = QLineEdit(self._default_path())
        b = QPushButton("Обзор…")
        b.clicked.connect(self._browse)
        pr.addWidget(QLabel("Файл:"))
        pr.addWidget(self.path, 1)
        pr.addWidget(b)
        lay.addLayout(pr)
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setObjectName("hint")
        lay.addWidget(self.notes)
        bb = QDialogButtonBox()
        ok = bb.addButton("Экспортировать", QDialogButtonBox.ButtonRole.AcceptRole)
        bb.addButton("Отмена", QDialogButtonBox.ButtonRole.RejectRole)
        ok.clicked.connect(self._go)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        for w in (self.fmt, self.sr):
            w.currentIndexChanged.connect(self._update)
        for w in (self.r_all, self.r_sel, self.r_x, self.norm):
            w.toggled.connect(self._update)
        self._update()

    def _default_path(self):
        base = self.ctrl.path.parent if self.ctrl.path else Path.home()
        name = (self.ctrl.path.stem if self.ctrl.path else self.ctrl.project.name) or "экспорт"
        return str(base / f"{name}.wav")

    def _browse(self):
        p, _ = QFileDialog.getSaveFileName(self, "Сохранить WAV", self.path.text(), "WAV (*.wav)")
        if p:
            self.path.setText(p if p.lower().endswith(".wav") else p + ".wav")

    def _update(self):
        notes = []
        region = not self.r_all.isChecked()
        self.edges.setEnabled(region)
        self.norm_db.setEnabled(self.norm.isChecked())
        self.dither.setEnabled(self.fmt.currentData() != "FLOAT")
        if self.sr.currentData() != self.ctrl.project.sample_rate:
            notes.append(f"Будет выполнен ресемплинг {self.ctrl.project.sample_rate} → {self.sr.currentData()} Гц "
                         f"(soxr VHQ) — зафиксированные PCM-участки не будут побитово совпадать.")
        if any(t.solo for t in self.ctrl.project.tracks):
            notes.append("Включено соло: экспорт будет содержать только дорожки соло.")
        if self.ctrl.store.missing:
            notes.append("Некоторые исходники не найдены — их клипы будут молчать.")
        notes.append("Нормализация и лимитер не применяются без явного выбора. Выравнивание громкости "
                     "для сравнения в экспорт не попадает.")
        self.notes.setText(" ".join(notes))

    def options(self) -> ExportOptions:
        start = end = None
        if self.r_sel.isChecked() and self.ctrl.selection.time_range:
            start, end = self.ctrl.selection.time_range
        elif self.r_x.isChecked():
            x = self.ctrl.project.any_transition(self.ctrl.selection.transition_id)
            a, b = self.ctrl.transition_bounds(x)
            start, end = max(0.0, a - self.ctx.value()), b + self.ctx.value()
        override = None
        vd = self.variant.currentData()
        if vd:
            mid, vid = vd
            mt = self.ctrl.project.music_transition(mid)
            v = mt.variant(vid) if mt else None
            if v is not None:
                override = {mid: v.params}
        return ExportOptions(path=self.path.text().strip(), start=start, end=end, tails=self.tails.isChecked(),
                             sample_rate=int(self.sr.currentData()), subtype=self.fmt.currentData(),
                             dither=self.dither.isChecked() and self.dither.isEnabled(),
                             normalize_dbfs=self.norm_db.value() if self.norm.isChecked() else None,
                             edge_fade_ms=5.0 if (self.edges.isChecked() and start is not None) else 0.0,
                             music_override=override)

    def _go(self):
        opts = self.options()
        if not opts.path:
            return
        if Path(opts.path).exists():
            ans = QMessageBox.question(self, "Экспорт", f"Файл уже существует:\n{opts.path}\nЗаменить?")
            if ans != QMessageBox.StandardButton.Yes:
                return
        self.accept()
        self.ctrl.export(self.win, opts, on_done=self.win.show_export_report)


def export_report_text(rep) -> str:
    def db(v):
        return "—" if v != v else ("−∞" if v == float("-inf") else f"{v:+.2f}")

    lines = [f"Файл: {rep.path}", f"Длительность: {rep.duration:.3f} с ({rep.frames} кадров, {rep.sample_rate} Гц, "
             f"{SUBTYPES[rep.subtype]})", f"Хвосты эффектов: {rep.tail_seconds:.2f} с",
             f"Пик: {db(rep.peak_dbfs)} дБFS · true peak ≈ {db(rep.true_peak_dbfs)} дБTP · громкость "
             f"{db(rep.lufs)} LUFS", f"Ограничено сэмплов: {rep.clipped_samples}"]
    if rep.normalize_gain_db:
        lines.append(f"Нормализация: {rep.normalize_gain_db:+.2f} дБ")
    lines.append(f"SHA-256: {rep.sha256}")
    lines += rep.warnings
    return "\n".join(lines)


class SettingsDialog(QDialog):
    def __init__(self, win, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.ctrl = win.ctrl
        self.setWindowTitle("Настройки")
        self.setMinimumWidth(560)
        lay = QVBoxLayout(self)
        tabs = QTabWidget()
        lay.addWidget(tabs)
        # --- ИИ
        w = QWidget()
        f = QFormLayout(w)
        cfg = self.ctrl.llm_config
        self.provider = QComboBox()
        for k, v in PROVIDERS.items():
            self.provider.addItem(v, k)
        self.provider.setCurrentIndex(max(0, self.provider.findData(cfg.provider)))
        self.endpoint = QLineEdit(cfg.endpoint)
        self.model = QLineEdit(cfg.model)
        self.key = QLineEdit()
        self.key.setEchoMode(QLineEdit.EchoMode.Password)
        self.names = QCheckBox("Отправлять имена дорожек и клипов")
        self.names.setChecked(cfg.include_names)
        f.addRow("Провайдер", self.provider)
        f.addRow("Адрес (endpoint)", self.endpoint)
        f.addRow("Модель", self.model)
        f.addRow("API-ключ", self.key)
        kr = QHBoxLayout()
        self.btn_delkey = QPushButton("Удалить сохранённый ключ")
        self.btn_delkey.clicked.connect(self._del_key)
        self.btn_check = QPushButton("Проверить подключение")
        self.btn_check.clicked.connect(self._check)
        kr.addWidget(self.btn_delkey)
        kr.addWidget(self.btn_check)
        f.addRow("", kr)
        f.addRow("", self.names)
        self.key_info = QLabel()
        self.key_info.setWordWrap(True)
        self.key_info.setObjectName("hint")
        f.addRow("", self.key_info)
        self.check_info = QLabel()
        self.check_info.setWordWrap(True)
        f.addRow("", self.check_info)
        priv = QLabel("Модели отправляется только текст задания и минимальное описание проекта (идентификаторы, "
                      "времена, параметры). Аудио, видео, пути к файлам и контрольные суммы не отправляются. "
                      "Всё редактирование, сохранение и экспорт работают без модели и без сети.")
        priv.setWordWrap(True)
        priv.setObjectName("hint")
        f.addRow(priv)
        tabs.addTab(w, "Внешняя модель")
        self.provider.currentIndexChanged.connect(self._provider_changed)
        # --- Аудио
        w2 = QWidget()
        f2 = QFormLayout(w2)
        self.device = QComboBox()
        self.device.addItem("Системное по умолчанию", "default")
        try:
            from ..audio.playback import import_sounddevice

            sd = import_sounddevice()

            for i, d in enumerate(sd.query_devices()):
                if d.get("max_output_channels", 0) >= 2:
                    self.device.addItem(f"{d['name']}", str(i))
        except Exception as e:
            self.device.addItem(f"(PortAudio недоступен: {e})", "default")
        cur = str(self.ctrl.settings.value("audio_device", "default"))
        self.device.setCurrentIndex(max(0, self.device.findData(cur)))
        f2.addRow("Аудиовыход", self.device)
        pb = self.ctrl.playback
        st = (f"Работает: {pb.device_name}, {pb.sr} Гц" if pb.backend == "device"
              else (pb.error or "Аудиовыход не открыт"))
        self.dev_info = QLabel(st)
        self.dev_info.setWordWrap(True)
        f2.addRow("", self.dev_info)
        self.autosave = QSpinBox()
        self.autosave.setRange(10, 3600)
        self.autosave.setSuffix(" с")
        self.autosave.setValue(int(self.ctrl.settings.value("autosave_s", 60)))
        f2.addRow("Автосохранение каждые", self.autosave)
        lat = QLabel(self.ctrl.latency_note())
        lat.setWordWrap(True)
        lat.setObjectName("hint")
        f2.addRow(lat)
        tabs.addTab(w2, "Аудио и проект")
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        bb.button(QDialogButtonBox.StandardButton.Save).setText("Сохранить")
        bb.button(QDialogButtonBox.StandardButton.Cancel).setText("Отмена")
        bb.accepted.connect(self._save)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)
        self._update_key_info()

    def _provider_changed(self):
        prov = self.provider.currentData()
        d = DEFAULTS.get(prov)
        if d:
            self.endpoint.setText(d["endpoint"])
            self.model.setText(d["model"])
        self._update_key_info()

    def _config(self) -> LlmConfig:
        return LlmConfig(provider=self.provider.currentData(), endpoint=self.endpoint.text().strip(),
                         model=self.model.text().strip(), include_names=self.names.isChecked())

    def _update_key_info(self):
        cfg = self._config()
        sec = self.ctrl.secrets
        has = cfg.provider != "none" and bool(sec.get(sec.account(cfg.provider, cfg.endpoint)))
        where = (f"в системном хранилище ({sec.backend_name})" if sec.secure
                 else "только в памяти до закрытия программы — системное хранилище секретов недоступно")
        self.key_info.setText(("Ключ сохранён " if has else "Ключ не задан. Ключ будет храниться ") + where +
                              ". В файлы проекта и журналы ключ не записывается.")
        self.key.setPlaceholderText("•••••• (сохранён)" if has else "")
        self.btn_delkey.setEnabled(has)

    def _del_key(self):
        cfg = self._config()
        self.ctrl.secrets.delete(self.ctrl.secrets.account(cfg.provider, cfg.endpoint))
        self._update_key_info()

    def _check(self):
        cfg = self._config()
        if self.key.text().strip():
            self.ctrl.secrets.set(self.ctrl.secrets.account(cfg.provider, cfg.endpoint), self.key.text().strip())
        from ..nlp.llm import LlmAdapter

        ad = LlmAdapter(cfg, self.ctrl.secrets)
        self.check_info.setText("Проверка…")

        def done(rep):
            color = S.OK if rep.ok else S.DANGER
            self.check_info.setText(f"<span style='color:{color}'>{'Подключение работает' if rep.ok else 'Ошибка'}"
                                    f"</span> ({rep.seconds:.1f} с): " + html.escape(" ".join(rep.messages)))
            self._update_key_info()

        run_job(self, "Проверка подключения…", lambda p, c: ad.check(), on_done=done,
                on_error=lambda m: self.check_info.setText(html.escape(m)))

    def _save(self):
        cfg = self._config()
        stored = self.ctrl.save_llm_config(cfg, self.key.text().strip() or None)
        if stored is False:
            self.win.show_message("warn", "Системное хранилище секретов недоступно: ключ будет храниться только "
                                          "до закрытия программы.")
        s = self.ctrl.settings
        dev = self.device.currentData()
        changed = str(s.value("audio_device", "default")) != dev
        s.setValue("audio_device", dev)
        s.setValue("autosave_s", self.autosave.value())
        self.ctrl.autosave_timer.setInterval(self.autosave.value() * 1000)
        s.sync()
        if changed:
            self.ctrl.playback.close()
            self.ctrl.open_audio()
        self.win.command.update_mode()
        self.accept()


class RelinkDialog(QDialog):
    """Поиск перемещённых исходников."""

    def __init__(self, win, statuses, parent=None):
        super().__init__(parent or win)
        self.win = win
        self.ctrl = win.ctrl
        self.statuses = statuses
        self.setWindowTitle("Исходники не найдены или изменены")
        self.setMinimumWidth(600)
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Некоторые исходные файлы проекта недоступны. Исходники никогда не перезаписываются; "
                             "найдите их, чтобы клипы снова звучали."))
        self.list = QListWidget()
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        b1 = QPushButton("Искать в папке…")
        b1.clicked.connect(self._search)
        b2 = QPushButton("Указать файл…")
        b2.clicked.connect(self._pick)
        b3 = QPushButton("Продолжить без них")
        b3.clicked.connect(self.accept)
        row.addWidget(b1)
        row.addWidget(b2)
        row.addStretch(1)
        row.addWidget(b3)
        lay.addLayout(row)
        self._fill()

    def _fill(self):
        self.list.clear()
        for st in self.statuses:
            if st.status == "ok":
                continue
            it = QListWidgetItem(f"{st.name} — {st.detail}\n{st.path}")
            it.setData(Qt.ItemDataRole.UserRole, st.source_id)
            self.list.addItem(it)
        if self.list.count() == 0:
            self.accept()

    def _apply(self, mapping: dict[str, str]):
        ops = [op("source.relink", target={"source_id": sid}, params={"path": p}) for sid, p in mapping.items()]
        if ops:
            self.ctrl.ui_ops(ops, text="Поиск перемещённых исходников")
            self.ctrl.store.missing -= set(mapping)
            self.ctrl.schedule_preview()
        self.statuses = [s for s in self.statuses if s.source_id not in mapping]
        self._fill()
        self.win.library.refresh()

    def _search(self):
        folder = QFileDialog.getExistingDirectory(self, "Папка для поиска")
        if not folder:
            return
        ids = [s.source_id for s in self.statuses if s.status != "ok"]
        project = self.ctrl.project.clone()

        def done(found):
            if not found:
                QMessageBox.information(self, "Поиск", "Файлы с совпадающей контрольной суммой не найдены.")
            self._apply(found)

        run_job(self, "Поиск исходников (по размеру и контрольной сумме)…",
                lambda p, c: find_moved_sources(project, folder, ids, cancel=c, progress=p), on_done=done,
                on_error=lambda m: QMessageBox.warning(self, "Поиск", m))

    def _pick(self):
        it = self.list.currentItem()
        if it is None:
            return
        sid = it.data(Qt.ItemDataRole.UserRole)
        src = self.ctrl.project.source(sid)
        path, _ = QFileDialog.getOpenFileName(self, f"Где файл «{src.name}»?", "", AUDIO_FILTER)
        if not path:
            return
        if sha256_file(path) != src.sha256:
            QMessageBox.warning(self, "Другой файл", "Содержимое файла отличается от исходника (контрольная сумма "
                                                      "не совпадает). Импортируйте его как новый исходник.")
            return
        self._apply({sid: path})


def check_sources_job(win, on_done):
    project = win.ctrl.project.clone()
    run_job(win, "Проверка исходников…", lambda p, c: check_sources(project, True, cancel=c, progress=p),
            on_done=on_done, on_error=lambda m: win.show_message("error", m), show_after_ms=800)


HELP_HTML = f"""
<h2>ASIKO {__version__} — краткая инструкция</h2>
<p>ASIKO переводит текстовое задание на русском в проверяемые операции над дорожками и выполняет их локально.
Исходные файлы не изменяются: проект хранит только ссылки, контрольные суммы и настройки.</p>
<h3>Сценарий «Звук в сцене → закадровый»</h3>
<ol><li>Импортируйте музыку (перетащите файл на шкалу).</li>
<li>Нажмите <b>«Звук в сцене → закадровый»</b>, выберите источник (радио, телефон, удалённый, за препятствием),
положение и время перехода, нажмите «Создать и прослушать».</li>
<li>В свойствах перехода задайте отдельно, когда раскрываются частоты, меняется ширина и исчезает комната.</li>
<li>Правки текстом: «Продли выбранный переход до 18 секунды», «Перенеси раскрытие частот в конец».</li></ol>
<h3>Сценарий «Переход A → B»</h3>
<ol><li>Импортируйте два трека (второй — на отдельную дорожку).</li>
<li>Нажмите <b>«Переход A → B»</b>, укажите примерное время. ASIKO оценит темп и доли (с уверенностью)
и предложит до трёх вариантов: короткое сведение, длинное сведение, завершение фразы.</li>
<li>Сравните варианты (панель транспорта → «Сравнение»), поправьте точку входа B кнопками ±доля/±10 мс.</li>
<li>Темп и высота тона не изменяются; если темпы не совпадают, конфликт показывается в варианте.</li></ol>
<h3>Прослушивание и экспорт</h3>
<p>Прослушивание и экспорт используют один рендер. Пробел — воспроизведение, L — петля по выделению,
выделение — перетаскивание по линейке. Экспорт (Ctrl+E): WAV 24 бит / 48 кГц по умолчанию, участок или весь
проект, хвосты эффектов по выбору. Нормализация и лимитер — только по явному выбору.</p>
<h3>Защита</h3>
<p>Можно фиксировать клипы, отдельные параметры, утверждать переходы и защищать участки времени (защита настроек).
Хвосты реверберации соседних участков могут звучать в защищённом участке; для побитовой неизменности
используйте «Правка → Зафиксировать выделенный участок в PCM».</p>
<h3>Текстовые команды</h3>
<p>Без сети работает локальный обработчик: он понимает только типовые шаблоны (не произвольный язык):</p>
<ul>{''.join(f'<li>{html.escape(t)}</li>' for t in SUPPORTED_TEMPLATES)}</ul>
<p>Внешняя модель подключается в настройках (адрес, модель, ключ). Её ответы проверяются так же строго.</p>
<h3>Горячие клавиши</h3>
<p>Пробел — воспроизведение/пауза · Home — в начало · L — петля · Ctrl+Z / Ctrl+Shift+Z — отмена/повтор ·
Ctrl+S — сохранить · Ctrl+O — открыть · Ctrl+I — импорт · Ctrl+E — экспорт · Ctrl+колесо — масштаб ·
Ctrl+0 — весь проект · Del — удалить выбранный переход · M/S — mute/solo дорожки · Ctrl+Enter — выполнить задание.</p>
<h3>Ограничения</h3>
<p>{html.escape(SOURCE_LIMITATION_NOTE)} Разделение источников (например, удаление ударных из стереомикса)
не поддерживается. Интеграция с игровыми движками не выполняется — ASIKO готовит аудиофайлы.</p>
"""


class HelpDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Справка ASIKO")
        self.resize(760, 640)
        lay = QVBoxLayout(self)
        tb = QTextBrowser()
        tb.setHtml(HELP_HTML)
        lay.addWidget(tb)
        bb = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        bb.button(QDialogButtonBox.StandardButton.Close).setText("Закрыть")
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)


def open_folder(path: str) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).parent)))
