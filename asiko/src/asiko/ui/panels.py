"""Панели: библиотека, индикаторы уровня, история, поле текстового задания."""
from __future__ import annotations

import html
import math
import time
from pathlib import Path

from PySide6.QtCore import QMimeData, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QDrag, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..audio.io import AUDIO_EXTENSIONS
from ..audio.sources import RESAMPLER
from ..commands.schema import op
from ..model.automation import seconds
from ..nlp.local_parser import SUPPORTED_TEMPLATES
from . import style as S
from .timeline import SOURCE_MIME
from .workers import run_job

AUDIO_FILTER = "Аудио (" + " ".join("*" + e for e in AUDIO_EXTENSIONS) + ");;Все файлы (*)"


class LibraryList(QListWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QListWidget.DragDropMode.DragDrop)
        self.files_dropped = None

    def startDrag(self, actions):
        it = self.currentItem()
        if it is None:
            return
        md = QMimeData()
        md.setData(SOURCE_MIME, it.data(Qt.ItemDataRole.UserRole).encode("utf-8"))
        d = QDrag(self)
        d.setMimeData(md)
        d.exec(Qt.DropAction.CopyAction)

    def dragEnterEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        if ev.mimeData().hasUrls():
            ev.acceptProposedAction()

    def dropEvent(self, ev):
        paths = [u.toLocalFile() for u in ev.mimeData().urls() if u.isLocalFile()]
        if paths and self.files_dropped:
            self.files_dropped(paths)
        ev.acceptProposedAction()


class LibraryPanel(QWidget):
    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.list = LibraryList()
        self.list.setWordWrap(True)
        self.list.files_dropped = lambda paths: self.ctrl.import_files(self.window(), paths)
        self.list.itemDoubleClicked.connect(lambda it: self.add_to_timeline())
        self.list.currentItemChanged.connect(lambda *_: self._update_buttons())
        lay.addWidget(self.list, 1)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("hint")
        lay.addWidget(self.info)
        row = QHBoxLayout()
        self.btn_import = QPushButton("Импорт аудио…")
        self.btn_import.clicked.connect(self.import_dialog)
        self.btn_add = QPushButton("На шкалу")
        self.btn_add.setToolTip("Поместить выбранный исходник на новую дорожку (или перетащите на шкалу)")
        self.btn_add.clicked.connect(self.add_to_timeline)
        row.addWidget(self.btn_import)
        row.addWidget(self.btn_add)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        self.btn_analyze = QPushButton("Анализ долей")
        self.btn_analyze.clicked.connect(self.analyze)
        self.btn_remove = QPushButton("Убрать")
        self.btn_remove.setToolTip("Удалить неиспользуемый исходник из библиотеки (файл на диске не трогается)")
        self.btn_remove.clicked.connect(self.remove)
        row2.addWidget(self.btn_analyze)
        row2.addWidget(self.btn_remove)
        lay.addLayout(row2)
        self.refresh()

    def selected_source(self):
        it = self.list.currentItem()
        if it is None:
            return None
        return self.ctrl.project.source(it.data(Qt.ItemDataRole.UserRole))

    def refresh(self):
        cur = self.list.currentItem().data(Qt.ItemDataRole.UserRole) if self.list.currentItem() else None
        self.list.clear()
        sr = self.ctrl.project.sample_rate
        for s in self.ctrl.project.sources:
            ch = "стерео" if s.channels == 2 else "моно"
            miss = " · НЕТ ФАЙЛА" if s.id in self.ctrl.store.missing else ""
            beats = f" · {s.beats.bpm:.1f} BPM ({s.beats.confidence:.0%})" if s.beats else ""
            it = QListWidgetItem(f"{s.name}\n{s.format} {s.sample_rate / 1000:g} кГц, {ch}, {seconds(s.duration)}{beats}{miss}")
            it.setData(Qt.ItemDataRole.UserRole, s.id)
            tip = (f"{s.path}\nФормат: {s.format} / {s.subtype}\nЧастота: {s.sample_rate} Гц"
                   + (f" → {sr} Гц ({RESAMPLER}, при импорте)" if s.sample_rate != sr else " (без ресемплинга)")
                   + f"\nSHA-256: {s.sha256[:16]}…")
            it.setToolTip(tip)
            if miss:
                it.setForeground(QColor(S.DANGER))
            self.list.addItem(it)
            if s.id == cur:
                self.list.setCurrentItem(it)
        if not self.ctrl.project.sources:
            self.info.setText("Импортируйте WAV, FLAC или MP3 (также OGG, AIFF). Исходные файлы не изменяются.")
        else:
            self.info.setText(f"Исходников: {len(self.ctrl.project.sources)}. Частота проекта {sr} Гц, "
                              f"обработка в плавающей точке.")
        self._update_buttons()

    def _update_buttons(self):
        has = self.selected_source() is not None
        self.btn_add.setEnabled(has)
        self.btn_analyze.setEnabled(has)
        self.btn_remove.setEnabled(has)

    def import_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Импорт аудио", "", AUDIO_FILTER)
        if paths:
            self.ctrl.import_files(self.window(), paths)

    def add_to_timeline(self):
        s = self.selected_source()
        if s is None:
            return
        self.ctrl.ui_ops([op("clip.add", target={"source_id": s.id},
                             time={"coord": "timeline", "unit": "s", "start": 0.0})], text="Клип из библиотеки")

    def analyze(self):
        s = self.selected_source()
        if s is not None:
            self.ctrl.analyze_sources(self.window(), [s.id], force=True)

    def remove(self):
        s = self.selected_source()
        if s is not None:
            self.ctrl.ui_ops([op("source.remove", target={"source_id": s.id})])


class MeterWidget(QWidget):
    """Пиковые индикаторы L/R с удержанием перегрузки и корреляция."""

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.setFixedSize(150, 30)
        self.level = [0.0, 0.0]
        self.setToolTip("Пики L/R (дБFS). Красный — перегрузка ≥ 0 дБFS, щёлкните, чтобы сбросить. "
                        "Ниже — корреляция L/R (моносовместимость).")

    def tick(self):
        pk = self.ctrl.playback.peak
        for i in range(2):
            v = float(pk[i])
            self.level[i] = max(v, self.level[i] * 0.85)
        self.update()

    def mousePressEvent(self, ev):
        self.ctrl.playback.reset_clip()
        self.update()

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#16181c"))
        w = self.width() - 18
        for i in range(2):
            y = 3 + i * 10
            v = self.level[i]
            db = 20 * math.log10(v) if v > 1e-6 else -90
            frac = max(0.0, min(1.0, (db + 60) / 60))
            col = QColor(S.OK) if db < -6 else (QColor(S.ACCENT2) if db < 0 else QColor(S.DANGER))
            p.fillRect(QRectF(2, y, w * frac, 7), col)
            clip = bool(self.ctrl.playback.clip[i])
            p.fillRect(QRectF(w + 6, y, 9, 7), QColor(S.DANGER) if clip else QColor("#3a2222"))
        c = self.ctrl.playback.correlation
        x = 2 + (c + 1) / 2 * w
        p.fillRect(QRectF(2, 24, w, 3), QColor("#2a2e34"))
        p.fillRect(QRectF(x - 2, 23, 4, 5), QColor(S.ACCENT if c >= 0 else S.DANGER))
        p.end()


class HistoryPanel(QWidget):
    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        self.list = QListWidget()
        self.list.setWordWrap(True)
        self.list.itemDoubleClicked.connect(self._jump)
        lay.addWidget(self.list, 1)
        row = QHBoxLayout()
        self.btn_undo = QPushButton("Отменить")
        self.btn_undo.clicked.connect(self.ctrl.undo)
        self.btn_redo = QPushButton("Повторить")
        self.btn_redo.clicked.connect(self.ctrl.redo)
        row.addWidget(self.btn_undo)
        row.addWidget(self.btn_redo)
        lay.addLayout(row)
        hint = QLabel("Двойной щелчок — вернуться к состоянию после этого шага.")
        hint.setObjectName("hint")
        hint.setWordWrap(True)
        lay.addWidget(hint)

    def refresh(self):
        self.list.clear()
        eng = self.ctrl.engine
        icons = {"ui": "✎", "local": "⌨", "llm": "✦"}
        for i, e in enumerate(eng.undo_stack):
            t = time.strftime("%H:%M:%S", time.localtime(e.timestamp))
            it = QListWidgetItem(f"{icons.get(e.origin, '•')} {t}  {e.description}")
            it.setData(Qt.ItemDataRole.UserRole, i)
            self.list.addItem(it)
        for e in reversed(eng.redo_stack):
            it = QListWidgetItem(f"↷ (отменено) {e.description}")
            it.setForeground(QColor(S.MUTED))
            it.setData(Qt.ItemDataRole.UserRole, None)
            self.list.addItem(it)
        if eng.undo_stack:
            self.list.scrollToItem(self.list.item(len(eng.undo_stack) - 1))
        self.btn_undo.setEnabled(bool(eng.undo_stack))
        self.btn_redo.setEnabled(bool(eng.redo_stack))

    def _jump(self, it):
        idx = it.data(Qt.ItemDataRole.UserRole)
        if idx is not None:
            self.ctrl.engine.undo_to(idx)


class CommandPanel(QWidget):
    """Поле текстового задания: план → проверка → применение (с отменой)."""

    planApplied = Signal()

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.pending_plan: dict | None = None
        self.pending_kind = ""
        self._job = None
        lay = QVBoxLayout(self)
        lay.setContentsMargins(6, 6, 6, 6)
        top = QHBoxLayout()
        self.edit = QPlainTextEdit()
        self.edit.setPlaceholderText("Текстовое задание, например: «Сделай переход с 12 до 16 секунды», "
                                     "«Продли выбранный переход на 2 секунды», «Перенеси раскрытие частот в конец». "
                                     "Ctrl+Enter — выполнить.")
        self.edit.setFixedHeight(64)
        for seq in ("Ctrl+Return", "Ctrl+Enter"):
            sc = QShortcut(QKeySequence(seq), self.edit)
            sc.setContext(Qt.ShortcutContext.WidgetShortcut)
            sc.activated.connect(self.run)
        top.addWidget(self.edit, 1)
        col = QVBoxLayout()
        self.btn_run = QPushButton("Выполнить")
        self.btn_run.setObjectName("primary")
        self.btn_run.clicked.connect(self.run)
        self.btn_templates = QPushButton("Шаблоны…")
        self.btn_templates.clicked.connect(self.show_templates)
        col.addWidget(self.btn_run)
        col.addWidget(self.btn_templates)
        top.addLayout(col)
        lay.addLayout(top)
        mode = QHBoxLayout()
        self.use_llm = QCheckBox("Использовать внешнюю модель")
        self.use_llm.setToolTip("Модели отправляется только текст задания и описание структуры проекта. "
                                "Аудио и видео не отправляются.")
        self.use_llm.setChecked(self.ctrl.use_llm)
        self.use_llm.toggled.connect(self._toggle_llm)
        self.mode_label = QLabel()
        self.mode_label.setObjectName("hint")
        mode.addWidget(self.use_llm)
        mode.addWidget(self.mode_label, 1)
        lay.addLayout(mode)
        self.result = QTextBrowser()
        self.result.setOpenLinks(False)
        self.result.anchorClicked.connect(self._anchor)
        self.result.setMinimumHeight(70)
        lay.addWidget(self.result, 1)
        act = QHBoxLayout()
        self.btn_apply = QPushButton("Применить")
        self.btn_apply.setObjectName("primary")
        self.btn_apply.clicked.connect(self.apply_pending)
        self.btn_undo = QPushButton("Отменить это изменение")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_play = QPushButton("Прослушать")
        self.btn_play.clicked.connect(lambda: self.window().play_selected_transition())
        act.addWidget(self.btn_apply)
        act.addWidget(self.btn_undo)
        act.addWidget(self.btn_play)
        act.addStretch(1)
        lay.addLayout(act)
        self._set_buttons(apply=False, undo=False)
        self.update_mode()

    def _toggle_llm(self, v):
        self.ctrl.use_llm = bool(v)
        self.ctrl.settings.setValue("use_llm", "true" if v else "false")
        self.update_mode()

    def update_mode(self):
        ok, why = self.ctrl.llm().available()
        self.use_llm.setEnabled(ok)
        if not ok:
            self.use_llm.setChecked(False)
        if self.use_llm.isChecked() and ok:
            c = self.ctrl.llm_config
            self.mode_label.setText(f"Внешняя модель: {c.model}. Ответ проверяется так же строго, как локальные команды.")
        else:
            self.mode_label.setText("Локальный обработчик: поддерживает только типовые шаблоны (см. «Шаблоны…»), "
                                    "работает без сети. " + ("" if ok else why))

    def _set_buttons(self, apply: bool, undo: bool, apply_text: str = "Применить"):
        self.btn_apply.setVisible(apply)
        self.btn_apply.setText(apply_text)
        self.btn_undo.setVisible(undo)
        self.btn_play.setVisible(undo and bool(self.ctrl.selection.transition_id))

    def show_templates(self):
        items = "".join(f"<li><a href='tpl:{i}'>{html.escape(t)}</a></li>" for i, t in enumerate(SUPPORTED_TEMPLATES))
        self.result.setHtml("<b>Локальный обработчик понимает шаблоны</b> (это не произвольный русский язык):"
                            f"<ul>{items}</ul><span style='color:{S.MUTED}'>Щёлкните шаблон, чтобы вставить его.</span>")

    def _anchor(self, url):
        s = url.toString()
        if s.startswith("tpl:"):
            self.edit.setPlainText(SUPPORTED_TEMPLATES[int(s[4:])].split(" / ")[0])
        elif s.startswith("opt:"):
            i = int(s[4:])
            opts = (self.pending_plan or {}).get("question", {}).get("options", [])
            if 0 <= i < len(opts) and opts[i].get("plan"):
                self.handle_plan(opts[i]["plan"])

    def run(self):
        text = self.edit.toPlainText().strip()
        if not text:
            return
        if self.use_llm.isChecked() and self.ctrl.llm().available()[0]:
            self._run_llm(text)
        else:
            self.handle_plan(self.ctrl.parse_local(text))

    def _run_llm(self, text):
        adapter = self.ctrl.llm()
        project = self.ctrl.project.clone()
        sel = self.ctrl.selection

        def work(progress, cancel):
            progress(0.1, "Запрос к внешней модели…")
            return adapter.plan(text, project, sel, cancel=cancel)

        def failed(msg):
            self.result.setHtml(f"<span style='color:{S.DANGER}'>Внешняя модель: {html.escape(msg)}</span><br>"
                                f"Проект не изменён. Можно повторить или выполнить задание локальным обработчиком.")
            local = self.ctrl.parse_local(text)
            if local.get("kind") != "info":
                self.result.append("<br><b>Локальный обработчик предлагает (применяется только по подтверждению):</b>")
                self.handle_plan(local, append=True, auto=False)

        run_job(self.window(), "Внешняя модель формирует план…", work, on_done=lambda plan: self.handle_plan(plan),
                on_error=failed, on_cancel=lambda: self.result.setHtml("Запрос к модели отменён."))

    def _plan_html(self, plan: dict, res) -> str:
        out = []
        origin = {"local": "локальный обработчик", "llm": "внешняя модель", "ui": "интерфейс"}.get(plan.get("origin"), "")
        ops = plan.get("operations", [])
        if ops:
            out.append(f"<span style='color:{S.MUTED}'>План ({origin}, {len(ops)} оп.): "
                       + html.escape(", ".join(o.get("op", "?") for o in ops)) + "</span>")
        if res is not None:
            for s in res.summary:
                out.append(html.escape(s))
            for w in res.warnings:
                out.append(f"<span style='color:{S.ACCENT2}'>⚠ {html.escape(w)}</span>")
            for e in res.errors:
                out.append(f"<span style='color:{S.DANGER}'>✖ {html.escape(e)}</span>")
        if plan.get("message"):
            out.append(f"<i>{html.escape(plan['message'])}</i>")
        return "<br>".join(out)

    def handle_plan(self, plan: dict, append: bool = False, auto: bool = True):
        """Показывает план и применяет его по правилам: конкретные обратимые команды — сразу
        (с кнопкой отмены), неоднозначные — вопрос, художественные и устаревшие — по подтверждению."""
        self.pending_plan = plan
        kind = plan.get("kind", "edit")
        setter = self.result.append if append else self.result.setHtml
        if kind == "question":
            q = plan.get("question") or {}
            opts = "".join(f"<li><a href='opt:{i}'>{html.escape(o.get('label', ''))}</a></li>"
                           for i, o in enumerate(q.get("options", [])))
            setter(f"<b>Уточнение:</b> {html.escape(q.get('text', ''))}<ul>{opts}</ul>")
            self._set_buttons(False, False)
            return
        if kind == "info":
            setter(f"<span>{html.escape(plan.get('message', ''))}</span>")
            self._set_buttons(False, False)
            return
        if kind in ("undo", "redo"):
            r = self.ctrl.undo() if kind == "undo" else self.ctrl.redo()
            setter(html.escape(" ".join(r.summary) if r.ok else "; ".join(r.errors)))
            self._set_buttons(False, False)
            self.planApplied.emit()
            return
        res = self.ctrl.engine.check(plan)
        if not res.ok:
            setter("<b>Команда не применена.</b><br>" + self._plan_html(plan, res))
            self._set_buttons(False, False)
            return
        if res.stale:
            setter("<b>Проект изменился после формирования плана — план проверен заново.</b><br>"
                   + self._plan_html(plan, res))
            self.pending_kind = "stale"
            self._set_buttons(True, False, "Применить к текущему проекту")
            return
        if res.artistic or kind == "proposal" or not auto:
            setter("<b>Предложение — применяется только по подтверждению.</b><br>" + self._plan_html(plan, res))
            self.pending_kind = "proposal"
            self._set_buttons(True, False, "Применить предложение")
            return
        r = self.ctrl.apply(plan)
        setter(("<b>Выполнено.</b><br>" if r.applied else "<b>Не применено.</b><br>") + self._plan_html(plan, r))
        self._set_buttons(False, r.applied)
        if r.applied:
            self.planApplied.emit()

    def apply_pending(self):
        if not self.pending_plan:
            return
        r = self.ctrl.apply(self.pending_plan, confirm_stale=True, accept_proposal=True)
        self.result.append(("<br><b>Применено.</b> " if r.applied else "<br><b>Не применено.</b> ")
                           + self._plan_html(self.pending_plan, r))
        self._set_buttons(False, r.applied)
        self.pending_plan = None
        if r.applied:
            self.planApplied.emit()

    def _undo(self):
        r = self.ctrl.undo()
        self.result.append("<br>" + html.escape(" ".join(r.summary) if r.ok else "Нечего отменять."))
        self._set_buttons(False, False)
