"""Панель свойств: стартовая страница, переход «в сцене → закадровый», переход A → B, клип, дорожка."""
from __future__ import annotations

import html
import math

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..commands.schema import op
from ..model import automation as A
from ..model.automation import OVERLAP_FILTERS, seconds
from ..model.presets import (
    CURVE_SHAPES,
    PARAM_INFO,
    ROOMS,
    SOURCE_LIMITATION_NOTE,
    SPACE_PARAMS,
    preset_names,
)
from ..audio.sources import RESAMPLER
from . import style as S

# отображение параметров в интерфейсе: (множитель, суффикс, мин, макс, шаг, знаков)
DISPLAY = {
    "gain_db": (1.0, " дБ", -60, 12, 0.5, 1),
    "pan": (100.0, " % (−Л/+П)", -100, 100, 5, 0),
    "hp_hz": (1.0, " Гц", 20, 4000, 10, 0),
    "lp_hz": (1.0, " Гц", 200, 20000, 100, 0),
    "width": (100.0, " %", 0, 200, 5, 0),
    "color_mix": (100.0, " %", 0, 100, 5, 0),
    "direct_db": (1.0, " дБ", -60, 6, 0.5, 1),
    "reverb_db": (1.0, " дБ", -80, 6, 1, 1),
}


SHORT_LABELS = {"gain_db": "Усиление", "pan": "Панорама", "hp_hz": "ФВЧ (низ)", "lp_hz": "ФНЧ (верх)",
                "width": "Ширина", "color_mix": "Обработка", "direct_db": "Прямой звук", "reverb_db": "Комната"}


def make_spin(lo, hi, step, decimals, suffix="", tip="", width=None):
    s = QDoubleSpinBox()
    if width:
        s.setMaximumWidth(width)
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setDecimals(decimals)
    s.setSuffix(suffix)
    s.setKeyboardTracking(False)
    s.setAccelerated(True)
    if tip:
        s.setToolTip(tip)
    return s


def set_quiet(w, value):
    w.blockSignals(True)
    if isinstance(w, (QDoubleSpinBox, QSpinBox)):
        w.setValue(value)
    elif isinstance(w, QComboBox):
        i = w.findData(value)
        w.setCurrentIndex(max(0, i))
    elif isinstance(w, QCheckBox):
        w.setChecked(bool(value))
    elif isinstance(w, QLineEdit):
        if not w.hasFocus():
            w.setText(value)
    w.blockSignals(False)


def note_box(text: str) -> QFrame:
    f = QFrame()
    f.setObjectName("note")
    lay = QVBoxLayout(f)
    lay.setContentsMargins(8, 6, 8, 6)
    lab = QLabel(text)
    lab.setWordWrap(True)
    lay.addWidget(lab)
    return f


class WelcomePage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        t = QLabel("ASIKO")
        t.setObjectName("title")
        lay.addWidget(t)
        lab = QLabel("Подготовка аудио для игр по текстовому заданию. Начните с одного из сценариев:")
        lab.setWordWrap(True)
        lay.addWidget(lab)
        b1 = QPushButton("Звук в сцене → закадровый")
        b1.setObjectName("scenario")
        b1.clicked.connect(win.scenario_space)
        b2 = QPushButton("Переход A → B")
        b2.setObjectName("scenario")
        b2.clicked.connect(win.scenario_music)
        lay.addWidget(b1)
        lay.addWidget(b2)
        guide = QLabel(
            "<ol style='margin-left:-20px'>"
            "<li>Импортируйте аудио: перетащите файлы на шкалу или «Импорт аудио…».</li>"
            "<li>Нажмите сценарий и укажите время — переход создаётся сразу.</li>"
            "<li>Прослушайте с контекстом (кнопка «Прослушать переход»).</li>"
            "<li>Правьте текстом внизу («Продли переход на 2 секунды») или здесь, в свойствах.</li>"
            "<li>Ctrl+Z — отмена, Ctrl+S — сохранить, Ctrl+E — экспорт WAV.</li></ol>"
            "<p style='color:#9aa3ad'>Выберите клип, переход или дорожку на шкале, чтобы увидеть свойства.</p>")
        guide.setWordWrap(True)
        lay.addWidget(guide)
        lay.addStretch(1)

    def refresh(self):
        pass


class CurvePreview(QWidget):
    def __init__(self, page, parent=None):
        super().__init__(parent)
        self.page = page
        self.setMinimumHeight(90)

    def paintEvent(self, ev):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor("#16181c"))
        x = self.page.current()
        if x is None:
            return
        proj = self.page.ctrl.project
        xs_all = proj.transitions_on_track(x.track_id)
        W, H = self.width(), self.height()
        ts = np.linspace(x.start, x.end, max(2, W // 2))
        xs = np.linspace(4, W - 4, ts.size)
        p.setPen(QColor("#3a4049"))
        p.drawRect(QRectF(2, 2, W - 4, H - 4))
        y = 12
        for prm in SPACE_PARAMS:
            if prm == "reverb_db":
                continue
            a, b = x.from_state.values[prm], x.to_state.values[prm]
            if abs(a - b) < 1e-9:
                continue
            v = A.space_param(xs_all, prm, ts)
            if PARAM_INFO[prm]["domain"] == "log":
                u = (np.log(v) - math.log(a)) / (math.log(b) - math.log(a))
            else:
                u = (v - a) / (b - a)
            yy = 6 + (1 - np.clip(u, 0, 1)) * (H - 12)
            p.setPen(QPen(QColor(S.GROUP_COLORS[prm]), 1.5))
            p.drawPolyline(QPolygonF([QPointF(float(a_), float(b_)) for a_, b_ in zip(xs, yy)]))
        for r in A.track_rooms(xs_all):
            a, b = x.from_state.effective_reverb(r), x.to_state.effective_reverb(r)
            if abs(a - b) < 1e-9:
                continue
            v = A.space_param(xs_all, "reverb_db", ts, room=r)
            u = (v - a) / (b - a)
            yy = 6 + (1 - np.clip(u, 0, 1)) * (H - 12)
            p.setPen(QPen(QColor(S.GROUP_COLORS["reverb_db"]), 1.5, Qt.PenStyle.DashLine))
            p.drawPolyline(QPolygonF([QPointF(float(a_), float(b_)) for a_, b_ in zip(xs, yy)]))
        p.setPen(QColor(S.MUTED))
        p.drawText(QPointF(6, H - 6), "начальное")
        p.drawText(QPointF(W - 70, 14), "конечное")
        p.end()


class SpacePage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.ctrl = win.ctrl
        self.xid = None
        lay = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setObjectName("title")
        self.title.setWordWrap(True)
        lay.addWidget(self.title)
        self.status = QLabel()
        self.status.setObjectName("hint")
        self.status.setWordWrap(True)
        lay.addWidget(self.status)
        row = QHBoxLayout()
        self.start = make_spin(0, 36000, 0.1, 3, " с", "Начало перехода (время шкалы)", width=105)
        self.end = make_spin(0, 36000, 0.1, 3, " с", "Конец перехода", width=105)
        self.slope = QComboBox()
        self.slope.addItem("12 дБ/окт", 12)
        self.slope.addItem("24 дБ/окт", 24)
        self.slope.setToolTip("Крутизна фильтров (ФВЧ/ФНЧ)")
        row.addWidget(QLabel("Начало"))
        row.addWidget(self.start)
        row.addWidget(QLabel("Конец"))
        row.addWidget(self.end)
        row.addStretch(1)
        lay.addLayout(row)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Крутизна фильтров"))
        row2.addWidget(self.slope)
        row2.addStretch(1)
        lay.addLayout(row2)
        self.start.valueChanged.connect(self._timing)
        self.end.valueChanged.connect(self._timing)
        self.slope.currentIndexChanged.connect(self._slope)

        g = QGroupBox("Состояния звучания")
        grid = QGridLayout(g)
        grid.addWidget(QLabel(""), 0, 0)
        grid.addWidget(QLabel("<b>Начальное</b>"), 0, 1)
        grid.addWidget(QLabel("<b>Конечное</b>"), 0, 2)
        self.preset = {}
        self.room = {}
        self.vals = {"from": {}, "to": {}}
        for col, which in ((1, "from"), (2, "to")):
            cb = QComboBox()
            cb.setMaximumWidth(128)
            cb.activated.connect(lambda _i, w=which: self._preset(w))
            self.preset[which] = cb
            grid.addWidget(cb, 1, col)
            rc = QComboBox()
            rc.setMaximumWidth(128)
            for k, v in ROOMS.items():
                rc.addItem(v, k)
            rc.currentIndexChanged.connect(lambda _i, w=which: self._room(w))
            self.room[which] = rc
            grid.addWidget(rc, 2, col)
        grid.addWidget(QLabel("Пресет"), 1, 0)
        grid.addWidget(QLabel("Комната"), 2, 0)
        for r, prm in enumerate(SPACE_PARAMS, start=3):
            mul, suf, lo, hi, st, dec = DISPLAY[prm]
            grid.addWidget(QLabel(PARAM_INFO[prm]["label"]), r, 0)
            for col, which in ((1, "from"), (2, "to")):
                sp = make_spin(lo, hi, st, dec, suf.replace(" (−Л/+П)", ""), width=110,
                               tip="Панорама: минус — слева, плюс — справа" if prm == "pan" else "")
                sp.valueChanged.connect(lambda v, w=which, p=prm: self._value(w, p, v))
                self.vals[which][prm] = sp
                grid.addWidget(sp, r, col)
        lay.addWidget(g)

        g2 = QGroupBox("Когда меняется параметр (с от начала)")
        grid2 = QGridLayout(g2)
        for c, t in enumerate(("Фикс.", "Параметр", "С", "До", "Форма")):
            grid2.addWidget(QLabel(f"<span style='color:{S.MUTED}'>{t}</span>"), 0, c)
        self.curves = {}
        for r, prm in enumerate(SPACE_PARAMS, start=1):
            lock = QCheckBox()
            lock.setToolTip("Зафиксировать параметр: его значения и время изменения не будут меняться командами")
            lock.toggled.connect(lambda v, p=prm: self._lock(p, v))
            lab = QLabel(f"<span style='color:{S.GROUP_COLORS[prm]}'>■</span> {SHORT_LABELS[prm]}")
            lab.setToolTip(PARAM_INFO[prm]["label"])
            s0 = make_spin(0, 3600, 0.1, 2, " с", width=82)
            s1 = make_spin(0, 3600, 0.1, 2, " с", width=82)
            sh = QComboBox()
            sh.setMaximumWidth(120)
            for k, v in CURVE_SHAPES.items():
                sh.addItem(v, k)
            s0.valueChanged.connect(lambda _v, p=prm: self._curve(p))
            s1.valueChanged.connect(lambda _v, p=prm: self._curve(p))
            sh.currentIndexChanged.connect(lambda _i, p=prm: self._curve(p, shape=True))
            for c, w in enumerate((lock, lab, s0, s1, sh)):
                grid2.addWidget(w, r, c)
            self.curves[prm] = (lock, s0, s1, sh)
        lay.addWidget(g2)
        self.preview = CurvePreview(self)
        lay.addWidget(self.preview)
        lay.addWidget(note_box(SOURCE_LIMITATION_NOTE))
        b = QHBoxLayout()
        self.btn_play = QPushButton("Прослушать с контекстом")
        self.btn_play.setObjectName("primary")
        self.btn_play.clicked.connect(self.win.play_selected_transition)
        self.btn_level = QPushButton("Сравнить громкость состояний")
        self.btn_level.setToolTip("Измерить громкость материала в начальном и конечном состоянии (BS.1770) "
                                  "и при желании выровнять начальное состояние")
        self.btn_level.clicked.connect(self._loudness)
        b.addWidget(self.btn_play)
        b.addWidget(self.btn_level)
        lay.addLayout(b)
        b2 = QHBoxLayout()
        self.btn_preset = QPushButton("Сохранить состояние как пресет…")
        self.btn_preset.clicked.connect(self._save_preset)
        self.btn_approve = QPushButton("Утвердить")
        self.btn_approve.clicked.connect(self._approve)
        self.btn_delete = QPushButton("Удалить")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self._delete)
        b2.addWidget(self.btn_preset)
        b2.addWidget(self.btn_approve)
        b2.addWidget(self.btn_delete)
        lay.addLayout(b2)
        lay.addStretch(1)

    def current(self):
        return self.ctrl.project.space_transition(self.xid) if self.xid else None

    def show_transition(self, xid):
        self.xid = xid
        self.refresh()

    def refresh(self):
        x = self.current()
        if x is None:
            return
        self.title.setText(x.name)
        tr = self.ctrl.project.track(x.track_id)
        st = [f"Дорожка «{tr.name if tr else '?'}»", f"{seconds(x.start)}–{seconds(x.end)} ({x.length:.2f} с)"]
        if x.approved:
            st.append("<b>утверждён</b> — правки запрещены")
        self.status.setText(" · ".join(st))
        set_quiet(self.start, x.start)
        set_quiet(self.end, x.end)
        set_quiet(self.slope, x.filter_slope)
        names = preset_names(self.ctrl.user_presets)
        for which, state in (("from", x.from_state), ("to", x.to_state)):
            cb = self.preset[which]
            cb.blockSignals(True)
            cb.clear()
            for k, v in names.items():
                cb.addItem(v, k)
            i = cb.findData(state.preset)
            cb.setCurrentIndex(max(0, i))
            cb.blockSignals(False)
            set_quiet(self.room[which], state.room)
            for prm, sp in self.vals[which].items():
                mul = DISPLAY[prm][0]
                set_quiet(sp, state.values[prm] * mul)
                sp.setEnabled(not x.approved and prm not in x.locked_params)
        for prm, (lock, s0, s1, sh) in self.curves.items():
            a0, a1 = x.curve_abs(prm)
            set_quiet(lock, prm in x.locked_params)
            for s in (s0, s1):
                s.blockSignals(True)
                s.setRange(0, max(0.01, x.length))
                s.blockSignals(False)
            set_quiet(s0, a0 - x.start)
            set_quiet(s1, a1 - x.start)
            set_quiet(sh, x.curves[prm].shape)
            for w in (s0, s1, sh):
                w.setEnabled(not x.approved and prm not in x.locked_params)
        for w in (self.start, self.end, self.slope, self.btn_delete):
            w.setEnabled(not x.approved)
        self.btn_approve.setText("Снять утверждение" if x.approved else "Утвердить")
        self.preview.update()

    def _ops(self, ops, text=""):
        self.ctrl.ui_ops(ops, text)

    def _timing(self):
        x = self.current()
        if x is None:
            return
        s, e = self.start.value(), self.end.value()
        if abs(s - x.start) < 1e-6 and abs(e - x.end) < 1e-6:
            return
        r = self.ctrl.ui_ops([op("space.set_timing", target={"transition_id": x.id},
                                 time={"coord": "timeline", "unit": "s", "start": s, "end": e})])
        if not r.ok:
            self.refresh()

    def _slope(self):
        x = self.current()
        if x is None or self.slope.currentData() == x.filter_slope:
            return
        self._ops([op("space.set_slope", target={"transition_id": x.id},
                      params={"filter_slope": int(self.slope.currentData())})])

    def _preset(self, which):
        x = self.current()
        if x is None:
            return
        key = self.preset[which].currentData()
        r = self.ctrl.ui_ops([op("space.set_state", target={"transition_id": x.id},
                                 params={"state": which, "preset": key})])
        if not r.ok:
            self.refresh()

    def _room(self, which):
        x = self.current()
        if x is None:
            return
        st = x.from_state if which == "from" else x.to_state
        room = self.room[which].currentData()
        if room == st.room:
            return
        r = self.ctrl.ui_ops([op("space.set_state", target={"transition_id": x.id},
                                 params={"state": which, "room": room})])
        if not r.ok:
            self.refresh()

    def _value(self, which, prm, v):
        x = self.current()
        if x is None:
            return
        st = x.from_state if which == "from" else x.to_state
        val = v / DISPLAY[prm][0]
        if abs(st.values[prm] - val) < 1e-9:
            return
        r = self.ctrl.ui_ops([op("space.set_state", target={"transition_id": x.id},
                                 params={"state": which, "values": {prm: val}})])
        if not r.ok:
            self.refresh()

    def _curve(self, prm, shape=False):
        x = self.current()
        if x is None:
            return
        lock, s0, s1, sh = self.curves[prm]
        a, b = s0.value(), s1.value()
        if b <= a:
            self.refresh()
            return
        params = {"params": [prm]}
        if shape:
            params["shape"] = sh.currentData()
        r = self.ctrl.ui_ops([op("space.set_curve", target={"transition_id": x.id},
                                 time={"coord": "transition", "unit": "s", "start": a, "end": b}, params=params)])
        if not r.ok:
            self.refresh()

    def _lock(self, prm, v):
        x = self.current()
        if x is None:
            return
        self._ops([op("transition.lock_params", target={"transition_id": x.id},
                      params={"params": [prm], "locked": bool(v)})])

    def _approve(self):
        x = self.current()
        if x is None:
            return
        self._ops([op("transition.approve", target={"transition_id": x.id}, params={"approved": not x.approved})])

    def _delete(self):
        x = self.current()
        if x is None:
            return
        self._ops([op("space.delete", target={"transition_id": x.id})])

    def _save_preset(self):
        x = self.current()
        if x is None:
            return
        which, ok = QInputDialog.getItem(self, "Сохранить пресет", "Какое состояние сохранить?",
                                         ["Начальное", "Конечное"], 0, False)
        if not ok:
            return
        st = x.from_state if which == "Начальное" else x.to_state
        name, ok = QInputDialog.getText(self, "Сохранить пресет", "Название пресета:", text=st.name + " (мой)")
        if not ok or not name.strip():
            return
        key = "user_" + "".join(ch if ch.isalnum() else "_" for ch in name.strip().lower())[:40]
        self.ctrl.save_user_preset(key, {"name": name.strip(), "values": dict(st.values), "room": st.room,
                                         "slope": x.filter_slope, "limitation": None})
        self.win.show_message("info", f"Пресет «{name.strip()}» сохранён и доступен во всех проектах.")
        self.refresh()

    def _loudness(self):
        x = self.current()
        if x is None:
            return
        from ..audio.analysis import state_loudness
        from .workers import run_job

        project = self.ctrl.project.clone()

        def work(progress, cancel):
            progress(0.2, "Рендер состояний…")
            return state_loudness(project, self.ctrl.store, x.id, cancel=cancel)

        def done(sl):
            if math.isinf(sl.from_lufs) or math.isinf(sl.to_lufs):
                self.win.show_message("warn", "Недостаточно звука рядом с переходом для измерения громкости.")
                return
            d = sl.diff_db
            msg = (f"Громкость материала: начальное состояние {sl.from_lufs:.1f} LUFS, конечное {sl.to_lufs:.1f} LUFS "
                   f"(разница {d:+.1f} дБ).")
            if abs(d) < 1.0:
                QMessageBox.information(self, "Громкость состояний", msg + "\nРазница мала — выравнивание не требуется.")
                return
            new_gain = max(-60.0, min(12.0, x.from_state.values["gain_db"] + d))
            ans = QMessageBox.question(self, "Громкость состояний",
                                       msg + f"\n\nИзменить усиление начального состояния до {new_gain:+.1f} дБ, "
                                       "чтобы при переходе не было скачка громкости? (Это изменит звучание "
                                       "до перехода; отменяется Ctrl+Z.)")
            if ans == QMessageBox.StandardButton.Yes:
                self._ops([op("space.set_state", target={"transition_id": x.id},
                              params={"state": "from", "values": {"gain_db": new_gain}})],
                          text="Выравнивание громкости состояний")

        run_job(self.window(), "Измерение громкости состояний…", work, on_done=done,
                on_error=lambda m: self.win.show_message("error", m))


class MusicPage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.ctrl = win.ctrl
        self.mid = None
        lay = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setObjectName("title")
        self.title.setWordWrap(True)
        lay.addWidget(self.title)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("hint")
        lay.addWidget(self.info)
        vl = QLabel("<b>Варианты</b> <span style='color:#9aa3ad'>(предложения по разметке долей, "
                    "не гарантированно художественно верные)</span>")
        vl.setWordWrap(True)
        lay.addWidget(vl)
        self.variants = QListWidget()
        self.variants.setWordWrap(True)
        self.variants.setMinimumHeight(150)
        self.variants.itemClicked.connect(self._pick)
        lay.addWidget(self.variants)
        g = QGroupBox("Активные параметры (ручная правка)")
        form = QFormLayout(g)
        self.sw = make_spin(0, 36000, 0.01, 3, " с", "Момент, когда звучит точка входа B")
        self.entry = make_spin(0, 36000, 0.005, 3, " с", "Точка входа в исходнике B (обычно сильная доля)")
        nud = QGridLayout()
        nud.setSpacing(3)
        self.nudges = []
        for i, (lab, kind, val) in enumerate((("−такт", "beats", -4), ("−доля", "beats", -1), ("−10 мс", "s", -0.01),
                                              ("+10 мс", "s", 0.01), ("+доля", "beats", 1), ("+такт", "beats", 4))):
            bt = QPushButton(lab)
            bt.setObjectName("small")
            bt.clicked.connect(lambda _c=False, k=kind, v=val: self._nudge(k, v))
            nud.addWidget(bt, i // 3, i % 3)
            self.nudges.append(bt)
        self.a0 = make_spin(0, 36000, 0.05, 3, " с")
        self.al = make_spin(0.005, 120, 0.05, 3, " с")
        self.b0 = make_spin(0, 36000, 0.05, 3, " с")
        self.bl = make_spin(0.005, 120, 0.05, 3, " с")
        self.curve = QComboBox()
        self.curve.addItem("Equal-power", "equal_power")
        self.curve.addItem("Линейное", "linear")
        self.tail = QComboBox()
        self.tail.addItem("Из исходной записи", "source")
        self.tail.addItem("Без хвоста (срез)", "cut")
        self.tail.addItem("Добавленный эффект (ревер.)", "effect")
        self.tail.setToolTip("Хвост A: продолжение исходной записи или хвост добавленной реверберации (эффект)")
        self.filt = QComboBox()
        for k, v in OVERLAP_FILTERS.items():
            self.filt.addItem(v, k)
        for w in (self.curve, self.tail, self.filt):
            w.setMaximumWidth(230)
        form.addRow("Вход B на шкале", self.sw)
        form.addRow("Точка входа в B", self.entry)
        form.addRow("", nud)
        form.addRow("Затухание A: начало", self.a0)
        form.addRow("Затухание A: длительность", self.al)
        form.addRow("Нарастание B: начало", self.b0)
        form.addRow("Нарастание B: длительность", self.bl)
        form.addRow("Кривая", self.curve)
        form.addRow("Хвост A", self.tail)
        form.addRow("Частоты в зоне наложения", self.filt)
        lay.addWidget(g)
        for w, key in ((self.sw, "switch_time"), (self.entry, "b_entry_src"), (self.a0, "a_fade_start"),
                       (self.al, "a_fade_len"), (self.b0, "b_fade_start"), (self.bl, "b_fade_len")):
            w.valueChanged.connect(lambda v, k=key: self._set(k, v))
        for w, key in ((self.curve, "curve"), (self.tail, "a_tail"), (self.filt, "overlap_filter")):
            w.currentIndexChanged.connect(lambda _i, ww=w, k=key: self._set(k, ww.currentData()))
        self.notes = QLabel()
        self.notes.setWordWrap(True)
        self.notes.setTextFormat(Qt.TextFormat.RichText)
        lay.addWidget(self.notes)
        b = QHBoxLayout()
        self.btn_play = QPushButton("Прослушать с контекстом")
        self.btn_play.setObjectName("primary")
        self.btn_play.clicked.connect(self.win.play_selected_transition)
        self.btn_cmp = QPushButton("Сравнить варианты")
        self.btn_cmp.setToolTip("Зациклить зону перехода и переключать варианты в панели транспорта")
        self.btn_cmp.clicked.connect(self._compare)
        b.addWidget(self.btn_play)
        b.addWidget(self.btn_cmp)
        lay.addLayout(b)
        b2 = QHBoxLayout()
        self.btn_approve = QPushButton("Утвердить вариант")
        self.btn_approve.clicked.connect(self._approve)
        self.btn_delete = QPushButton("Удалить переход")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(self._delete)
        b2.addWidget(self.btn_approve)
        b2.addWidget(self.btn_delete)
        lay.addLayout(b2)
        lay.addStretch(1)

    def current(self):
        return self.ctrl.project.music_transition(self.mid) if self.mid else None

    def show_transition(self, mid):
        self.mid = mid
        self.refresh()

    def refresh(self):
        m = self.current()
        if m is None:
            return
        proj = self.ctrl.project
        a, b = proj.clip(m.clip_a_id), proj.clip(m.clip_b_id)
        self.title.setText(f"Переход A → B: {a.name if a else '?'} → {b.name if b else '?'}")
        cons = []
        if proj.constraints.get("keep_tempo"):
            cons.append("темп не меняется")
        if proj.constraints.get("keep_pitch"):
            cons.append("высота тона не меняется")
        info = [f"Задано около {seconds(m.target_time)}"]
        if cons:
            info.append(", ".join(cons))
        if m.approved:
            info.append("<b>утверждён</b>")
        if m.modified:
            info.append("параметры изменены вручную")
        self.info.setText(" · ".join(info))
        self.variants.blockSignals(True)
        self.variants.clear()
        for i, v in enumerate(m.variants):
            mark = "● " if v.id == m.selected_variant else "○ "
            txt = f"{mark}{i + 1}. {v.name}\n{v.description}"
            if v.conflicts:
                txt += "\nКонфликт: " + " ".join(v.conflicts)
            if v.notes:
                txt += "\n" + " ".join(v.notes[:3])
            txt += f"\nУверенность разметки: {v.confidence:.0%}"
            it = QListWidgetItem(txt)
            it.setData(Qt.ItemDataRole.UserRole, v.id)
            if v.conflicts:
                it.setForeground(QColor(S.ACCENT2))
            self.variants.addItem(it)
        self.variants.blockSignals(False)
        p = m.params
        for w, v in ((self.sw, p.switch_time), (self.entry, p.b_entry_src), (self.a0, p.a_fade_start),
                     (self.al, p.a_fade_len), (self.b0, p.b_fade_start), (self.bl, p.b_fade_len)):
            set_quiet(w, v)
        set_quiet(self.curve, p.curve)
        set_quiet(self.tail, p.a_tail)
        set_quiet(self.filt, p.overlap_filter)
        parts = []
        for c in m.conflicts:
            parts.append(f"<span style='color:{S.DANGER}'>Конфликт: {html.escape(c)}</span>")
        for n in m.notes:
            parts.append(f"<span style='color:{S.MUTED}'>{html.escape(n)}</span>")
        self.notes.setText("<br>".join(parts))
        en = not m.approved
        for w in [self.sw, self.entry, self.a0, self.al, self.b0, self.bl, self.curve, self.tail, self.filt,
                  self.btn_delete, self.variants] + self.nudges:
            w.setEnabled(en)
        src_b = proj.source(b.source_id) if b else None
        for bt in self.nudges[:2] + self.nudges[4:]:
            bt.setEnabled(en and bool(src_b and src_b.beats))
        self.btn_approve.setText("Снять утверждение" if m.approved else "Утвердить вариант")

    def _set(self, key, value):
        m = self.current()
        if m is None:
            return
        if getattr(m.params, key) == value:
            return
        r = self.ctrl.ui_ops([op("music.set", target={"transition_id": m.id}, params={key: value})])
        if not r.ok:
            self.refresh()

    def _nudge(self, kind, v):
        m = self.current()
        if m is None:
            return
        params = {"b_entry_shift_beats": v} if kind == "beats" else {"b_entry_shift": v}
        self.ctrl.ui_ops([op("music.set", target={"transition_id": m.id}, params=params)], text="Сдвиг точки входа B")

    def _pick(self, it):
        m = self.current()
        if m is None:
            return
        vid = it.data(Qt.ItemDataRole.UserRole)
        if vid == m.selected_variant and not m.modified:
            return
        self.ctrl.ui_ops([op("music.select_variant", target={"transition_id": m.id}, params={"variant": vid})])

    def _compare(self):
        m = self.current()
        if m is None:
            return
        o0, o1 = m.params.overlap
        for v in m.variants:
            o0 = min(o0, v.params.overlap[0])
            o1 = max(o1, v.params.overlap[1])
        self.ctrl.select(time_range=(max(0.0, o0 - 3.0), o1 + 3.0))
        self.win.start_compare()

    def _approve(self):
        m = self.current()
        if m is not None:
            self.ctrl.ui_ops([op("transition.approve", target={"transition_id": m.id}, params={"approved": not m.approved})])

    def _delete(self):
        m = self.current()
        if m is not None:
            self.ctrl.ui_ops([op("music.delete", target={"transition_id": m.id})])


class ClipPage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.ctrl = win.ctrl
        self.cid = None
        lay = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setObjectName("title")
        lay.addWidget(self.title)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("hint")
        lay.addWidget(self.info)
        form = QFormLayout()
        self.name = QLineEdit()
        self.name.editingFinished.connect(self._name)
        self.start = make_spin(0, 36000, 0.01, 3, " с", "Начало клипа на шкале (связанные переходы переносятся)")
        self.start.valueChanged.connect(self._start)
        self.gain = make_spin(-60, 24, 0.5, 1, " дБ")
        self.gain.valueChanged.connect(lambda v: self._set("gain_db", v))
        self.fin = make_spin(0, 60000, 5, 0, " мс")
        self.fin.valueChanged.connect(lambda v: self._set("fade_in", v / 1000))
        self.fout = make_spin(0, 60000, 5, 0, " мс")
        self.fout.valueChanged.connect(lambda v: self._set("fade_out", v / 1000))
        self.lock = QCheckBox("Зафиксировать клип (положение и настройки)")
        self.lock.toggled.connect(lambda v: self._set("locked", bool(v)))
        form.addRow("Имя", self.name)
        form.addRow("Начало", self.start)
        form.addRow("Усиление", self.gain)
        form.addRow("Нарастание", self.fin)
        form.addRow("Затухание", self.fout)
        lay.addLayout(form)
        lay.addWidget(self.lock)
        g = QGroupBox("Темп, доли, фразы")
        gl = QVBoxLayout(g)
        self.beats_info = QLabel()
        self.beats_info.setWordWrap(True)
        gl.addWidget(self.beats_info)
        f2 = QFormLayout()
        self.bpm = make_spin(20, 400, 0.1, 3, " BPM")
        self.first = make_spin(0, 36000, 0.005, 3, " с", "Первая сильная доля (время исходника)")
        self.bpb = QSpinBox()
        self.bpb.setRange(1, 16)
        self.bpb.setValue(4)
        self.phrase = QSpinBox()
        self.phrase.setRange(1, 64)
        self.phrase.setValue(8)
        f2.addRow("Темп", self.bpm)
        f2.addRow("Первая сильная доля", self.first)
        f2.addRow("Долей в такте", self.bpb)
        f2.addRow("Тактов во фразе", self.phrase)
        gl.addLayout(f2)
        r1 = QHBoxLayout()
        self.btn_analyze = QPushButton("Анализировать")
        self.btn_analyze.clicked.connect(self._analyze)
        self.btn_apply = QPushButton("Применить вручную")
        self.btn_apply.setToolTip("Задать равномерную сетку по темпу и первой сильной доле")
        self.btn_apply.clicked.connect(self._manual)
        r1.addWidget(self.btn_analyze)
        r1.addWidget(self.btn_apply)
        gl.addLayout(r1)
        r2 = QHBoxLayout()
        self.btn_x2 = QPushButton("×2")
        self.btn_x2.setToolTip("Удвоить темп (исправить ошибку октавы)")
        self.btn_x2.clicked.connect(lambda: self._scale(2.0))
        self.btn_d2 = QPushButton("÷2")
        self.btn_d2.clicked.connect(lambda: self._scale(0.5))
        self.btn_m10 = QPushButton("−10 мс")
        self.btn_m10.clicked.connect(lambda: self._shift(-0.01))
        self.btn_p10 = QPushButton("+10 мс")
        self.btn_p10.clicked.connect(lambda: self._shift(0.01))
        for w in (self.btn_x2, self.btn_d2, self.btn_m10, self.btn_p10):
            w.setObjectName("small")
            r2.addWidget(w)
        gl.addLayout(r2)
        r3 = QGridLayout()
        self.btn_down = QPushButton("Сильная доля у курсора")
        self.btn_down.clicked.connect(self._down_here)
        self.btn_phrase = QPushButton("Граница фразы у курсора")
        self.btn_phrase.clicked.connect(self._phrase_here)
        self.btn_clear = QPushButton("Сбросить фразы")
        self.btn_clear.clicked.connect(self._clear_phrases)
        r3.addWidget(self.btn_down, 0, 0)
        r3.addWidget(self.btn_phrase, 0, 1)
        r3.addWidget(self.btn_clear, 1, 1)
        gl.addLayout(r3)
        lay.addWidget(g)
        self.btn_delete = QPushButton("Удалить клип")
        self.btn_delete.setObjectName("danger")
        self.btn_delete.clicked.connect(lambda: self.ctrl.ui_ops([op("clip.delete", target={"clip_id": self.cid})]))
        lay.addWidget(self.btn_delete)
        lay.addStretch(1)

    def clip(self):
        return self.ctrl.project.clip(self.cid) if self.cid else None

    def show_clip(self, cid):
        self.cid = cid
        self.refresh()

    def refresh(self):
        c = self.clip()
        if c is None:
            return
        proj = self.ctrl.project
        src = proj.source(c.source_id)
        self.title.setText(c.name or "Клип")
        if src:
            rs = (f"{src.sample_rate} Гц → {proj.sample_rate} Гц ({RESAMPLER})" if src.sample_rate != proj.sample_rate
                  else f"{src.sample_rate} Гц")
            miss = " · <span style='color:#e05a4f'>файл не найден</span>" if src.id in self.ctrl.store.missing else ""
            self.info.setText(f"{src.format}/{src.subtype}, {rs}, {'стерео' if src.channels == 2 else 'моно'}, "
                              f"{seconds(src.duration)} · на шкале {seconds(c.start)}–{seconds(c.end)} "
                              f"(исходник {seconds(c.src_in)}–{seconds(c.src_out)}){miss}")
        set_quiet(self.name, c.name)
        set_quiet(self.start, c.start)
        set_quiet(self.gain, c.gain_db)
        set_quiet(self.fin, c.fade_in * 1000)
        set_quiet(self.fout, c.fade_out * 1000)
        set_quiet(self.lock, c.locked)
        for w in (self.name, self.start, self.gain, self.fin, self.fout, self.btn_delete):
            w.setEnabled(not c.locked)
        g = src.beats if src else None
        if g:
            txt = (f"<b>{g.bpm:.2f} BPM</b>, уверенность {g.confidence:.0%} (темп {g.tempo_confidence:.0%}, "
                   f"сильные доли {g.downbeat_confidence:.0%}), {('анализ' if g.method == 'auto' else 'вручную')}.")
            if g.notes:
                txt += "<br><span style='color:#9aa3ad'>" + html.escape(" ".join(g.notes)) + "</span>"
            if src.phrases:
                txt += f"<br>Фразы (вручную): {', '.join(seconds(v) for v in src.phrases)}"
            self.beats_info.setText(txt)
            set_quiet(self.bpm, g.bpm)
            set_quiet(self.first, g.downbeats[0] if g.downbeats else 0.0)
            set_quiet(self.bpb, g.beats_per_bar)
            set_quiet(self.phrase, g.phrase_bars)
        else:
            self.beats_info.setText("Разметки долей нет. «Анализировать» — оценка темпа и долей (librosa), "
                                    "или задайте темп и первую сильную долю вручную.")
        for w in (self.btn_x2, self.btn_d2, self.btn_m10, self.btn_p10, self.btn_down, self.btn_phrase):
            w.setEnabled(g is not None)
        self.btn_clear.setEnabled(bool(src and src.phrases))

    def _name(self):
        c = self.clip()
        if c and self.name.text().strip() and self.name.text() != c.name:
            self._set("name", self.name.text().strip())

    def _set(self, key, value):
        c = self.clip()
        if c is None:
            return
        if getattr(c, key) == value:
            return
        r = self.ctrl.ui_ops([op("clip.set", target={"clip_id": c.id}, params={key: value})])
        if not r.ok:
            self.refresh()

    def _start(self, v):
        c = self.clip()
        if c is None or abs(c.start - v) < 1e-6:
            return
        r = self.ctrl.ui_ops([op("clip.move", target={"clip_id": c.id}, time={"coord": "timeline", "unit": "s", "start": v})])
        if not r.ok:
            self.refresh()

    def _analyze(self):
        c = self.clip()
        if c:
            self.ctrl.analyze_sources(self.window(), [c.source_id], force=True)

    def _manual(self):
        c = self.clip()
        if c is None:
            return
        self.ctrl.ui_ops([op("beats.set", target={"source_id": c.source_id},
                             params={"bpm": self.bpm.value(), "first_downbeat": self.first.value(),
                                     "beats_per_bar": self.bpb.value(), "phrase_bars": self.phrase.value()})],
                         text="Ручная сетка долей")

    def _scale(self, k):
        c = self.clip()
        src = self.ctrl.project.source(c.source_id) if c else None
        if not src or not src.beats:
            return
        fd = src.beats.downbeats[0] if src.beats.downbeats else 0.0
        self.ctrl.ui_ops([op("beats.set", target={"source_id": src.id},
                             params={"bpm": min(400.0, max(20.0, src.beats.bpm * k)), "first_downbeat": fd,
                                     "beats_per_bar": src.beats.beats_per_bar, "phrase_bars": src.beats.phrase_bars})])

    def _shift(self, d):
        c = self.clip()
        if c:
            self.ctrl.ui_ops([op("beats.shift", target={"source_id": c.source_id}, params={"delta": d})])

    def _down_here(self):
        c = self.clip()
        src = self.ctrl.project.source(c.source_id) if c else None
        if not src or not src.beats:
            return
        s = c.to_src(self.ctrl.playback.position)
        if not (0 <= s <= src.duration):
            self.win.show_message("info", "Курсор вне клипа.")
            return
        self.ctrl.ui_ops([op("beats.set", target={"source_id": src.id},
                             params={"bpm": src.beats.bpm, "first_downbeat": round(s, 4),
                                     "beats_per_bar": src.beats.beats_per_bar, "phrase_bars": src.beats.phrase_bars})],
                         text="Сильная доля у курсора")

    def _phrase_here(self):
        c = self.clip()
        src = self.ctrl.project.source(c.source_id) if c else None
        if not src:
            return
        s = c.to_src(self.ctrl.playback.position)
        if not (0 <= s <= src.duration):
            self.win.show_message("info", "Курсор вне клипа.")
            return
        self.ctrl.ui_ops([op("phrases.set", target={"source_id": src.id},
                             params={"phrases": sorted(set(src.phrases + [round(s, 4)]))})])

    def _clear_phrases(self):
        c = self.clip()
        if c:
            self.ctrl.ui_ops([op("phrases.set", target={"source_id": c.source_id}, params={"phrases": []})])


class TrackPage(QWidget):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.ctrl = win.ctrl
        self.tid = None
        lay = QVBoxLayout(self)
        self.title = QLabel()
        self.title.setObjectName("title")
        lay.addWidget(self.title)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setObjectName("hint")
        lay.addWidget(self.info)
        b = QPushButton("Звук в сцене → закадровый на этой дорожке")
        b.clicked.connect(self.win.scenario_space)
        lay.addWidget(b)
        b2 = QPushButton("Защитить дорожку после курсора")
        b2.setToolTip("Настройки дорожки после текущей позиции не будут меняться командами")
        b2.clicked.connect(self._protect)
        lay.addWidget(b2)
        self.btn_del = QPushButton("Удалить дорожку")
        self.btn_del.setObjectName("danger")
        self.btn_del.clicked.connect(lambda: self.ctrl.ui_ops([op("track.delete", target={"track_id": self.tid})]))
        lay.addWidget(self.btn_del)
        lay.addStretch(1)

    def show_track(self, tid):
        self.tid = tid
        self.refresh()

    def refresh(self):
        t = self.ctrl.project.track(self.tid) if self.tid else None
        if t is None:
            return
        self.title.setText(t.name)
        nx = self.ctrl.project.transitions_on_track(t.id)
        self.info.setText(f"Клипов: {len(t.clips)} · уровень {t.gain_db:+.1f} дБ"
                          + (" · mute" if t.mute else "") + (" · solo" if t.solo else "")
                          + (f" · переходов «в сцене → закадровый»: {len(nx)}" if nx else ""))

    def _protect(self):
        t = self.ctrl.project.track(self.tid) if self.tid else None
        if t is None:
            return
        self.ctrl.ui_ops([op("protect.add", params={"start": round(self.ctrl.playback.position, 3), "end": None,
                                                    "track_id": t.id})], text="Защита участка дорожки")


class Inspector(QScrollArea):
    def __init__(self, win, parent=None):
        super().__init__(parent)
        self.win = win
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.stack = QStackedWidget()
        self.welcome = WelcomePage(win)
        self.space = SpacePage(win)
        self.music = MusicPage(win)
        self.clip = ClipPage(win)
        self.track = TrackPage(win)
        for w in (self.welcome, self.space, self.music, self.clip, self.track):
            self.stack.addWidget(w)
        self.setWidget(self.stack)
        self.setMinimumWidth(470)

    def _show(self, page):
        from PySide6.QtWidgets import QSizePolicy

        for w in (self.welcome, self.space, self.music, self.clip, self.track):
            pol = QSizePolicy.Policy.Preferred if w is page else QSizePolicy.Policy.Ignored
            w.setSizePolicy(pol, pol)
        self.stack.setCurrentWidget(page)
        self.stack.adjustSize()

    def refresh(self):
        ctrl = self.win.ctrl
        s = ctrl.selection
        p = ctrl.project
        if s.transition_id and p.space_transition(s.transition_id):
            self.space.show_transition(s.transition_id)
            self._show(self.space)
        elif s.transition_id and p.music_transition(s.transition_id):
            self.music.show_transition(s.transition_id)
            self._show(self.music)
        elif s.clip_ids and p.clip(s.clip_ids[-1]):
            self.clip.show_clip(s.clip_ids[-1])
            self._show(self.clip)
        elif s.track_id and p.track(s.track_id) and p.tracks:
            self.track.show_track(s.track_id)
            self._show(self.track)
        else:
            self._show(self.welcome)
