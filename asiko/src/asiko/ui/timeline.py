"""Многодорожечный таймлайн: волновые формы, переходы, кривые, выделение, перетаскивание."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QCursor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QScrollBar,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..audio.io import AUDIO_EXTENSIONS
from ..commands.schema import op
from ..model import automation as A
from ..model.presets import PARAM_INFO, SPACE_PARAMS
from . import style as S

RULER_H = 30
ROW_H = 96
HEADER_W = 200
SOURCE_MIME = "application/x-asiko-source"


def fmt_time(t: float, step: float = 1.0) -> str:
    if t < 0:
        return "-" + fmt_time(-t, step)
    m = int(t // 60)
    s = t - 60 * m
    if step < 0.1:
        return f"{m}:{s:05.2f}"
    if step < 1:
        return f"{m}:{s:04.1f}"
    return f"{m}:{int(round(s)):02d}"


class TimelineCanvas(QWidget):
    seekRequested = Signal(float)
    contextRequested = Signal(str, str, object, float)   # kind, id, global pos, time
    viewChanged = Signal()

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.px_per_s = 40.0
        self.offset = 0.0
        self.setMouseTracking(True)
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._drag = None
        self._drop_hint = None
        self.follow = True
        self.setMinimumHeight(RULER_H + ROW_H)

    # ---------------------------------------------------------------- geometry
    def x_of(self, t: float) -> float:
        return (t - self.offset) * self.px_per_s

    def t_of(self, x: float) -> float:
        return self.offset + x / self.px_per_s

    def row_of(self, y: float) -> int:
        return int((y - RULER_H) // ROW_H) if y >= RULER_H else -1

    def row_top(self, i: int) -> int:
        return RULER_H + i * ROW_H

    def track_index(self, tid: str) -> int:
        for i, t in enumerate(self.ctrl.project.tracks):
            if t.id == tid:
                return i
        return -1

    def visible_seconds(self) -> float:
        return max(0.1, self.width() / self.px_per_s)

    def set_zoom(self, px_per_s: float, anchor_t: float | None = None) -> None:
        px_per_s = min(4000.0, max(1.0, px_per_s))
        if anchor_t is None:
            anchor_t = self.offset + self.visible_seconds() / 2
        ax = self.x_of(anchor_t)
        self.px_per_s = px_per_s
        self.offset = max(0.0, anchor_t - ax / self.px_per_s)
        self.viewChanged.emit()
        self.update()

    def zoom_fit(self) -> None:
        end = max(self.ctrl.project.content_end() + 2.0, 10.0)
        self.offset = 0.0
        self.px_per_s = max(1.0, (self.width() - 20) / end)
        self.viewChanged.emit()
        self.update()

    def show_range(self, a: float, b: float) -> None:
        span = max(0.5, b - a)
        self.px_per_s = min(4000.0, max(1.0, self.width() / (span * 1.3)))
        self.offset = max(0.0, a - span * 0.15)
        self.viewChanged.emit()
        self.update()

    def ensure_visible(self, t: float) -> None:
        if t < self.offset or t > self.offset + self.visible_seconds():
            self.offset = max(0.0, t - self.visible_seconds() * 0.1)
            self.viewChanged.emit()

    # ---------------------------------------------------------------- painting
    def paintEvent(self, ev):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        W, H = self.width(), self.height()
        p.fillRect(0, 0, W, H, QColor(S.BG))
        proj = self.ctrl.project
        sel = self.ctrl.selection
        tracks = proj.tracks
        # дорожки
        for i, t in enumerate(tracks):
            top = self.row_top(i)
            p.fillRect(0, top, W, ROW_H, QColor("#22262c" if i % 2 == 0 else "#25292f"))
            if sel.track_id == t.id:
                p.fillRect(0, top, W, ROW_H, S.qcolor(S.ACCENT, 18))
            p.setPen(QColor("#30353c"))
            p.drawLine(0, top + ROW_H - 1, W, top + ROW_H - 1)
        self._paint_protections(p, H)
        for i, t in enumerate(tracks):
            self._paint_space_bands(p, i, t)
        for i, t in enumerate(tracks):
            color = QColor(S.TRACK_COLORS[i % len(S.TRACK_COLORS)])
            for c in t.clips:
                self._paint_clip(p, i, c, color, c.id in sel.clip_ids)
        self._paint_beats(p)
        self._paint_music(p)
        for i, t in enumerate(tracks):
            self._paint_space_curves(p, i, t)
        # выделение
        if sel.time_range:
            a, b = sel.time_range
            x0, x1 = self.x_of(a), self.x_of(b)
            p.fillRect(QRectF(x0, RULER_H, x1 - x0, H - RULER_H), S.qcolor(S.ACCENT, 32))
            p.setPen(QPen(S.qcolor(S.ACCENT, 160), 1))
            p.drawLine(QLineF(x0, RULER_H, x0, H))
            p.drawLine(QLineF(x1, RULER_H, x1, H))
        if not tracks:
            p.setPen(QColor(S.MUTED))
            f = p.font()
            f.setPointSize(f.pointSize() + 2)
            p.setFont(f)
            p.drawText(QRectF(0, RULER_H, W, max(ROW_H, H - RULER_H)), Qt.AlignmentFlag.AlignCenter,
                       "Перетащите сюда WAV / FLAC / MP3 или нажмите «Импорт аудио…»")
        if self._drop_hint is not None:
            row, t = self._drop_hint
            top = self.row_top(max(0, row))
            p.setPen(QPen(QColor(S.ACCENT2), 2, Qt.PenStyle.DashLine))
            p.drawRect(QRectF(self.x_of(t), top + 3, 120, ROW_H - 6))
        self._paint_ruler(p, W)
        # курсор воспроизведения
        x = self.x_of(self.ctrl.playback.position)
        p.setPen(QPen(QColor(S.DANGER), 1.5))
        p.drawLine(QLineF(x, 0, x, H))
        p.end()

    def _paint_ruler(self, p: QPainter, W: int):
        p.fillRect(0, 0, W, RULER_H, QColor(S.PANEL))
        steps = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300]
        step = next((s for s in steps if s * self.px_per_s >= 70), 600)
        minor = step / 5
        t0 = math.floor(self.offset / minor) * minor
        tend = self.offset + self.visible_seconds()
        p.setPen(QColor("#59616b"))
        t = t0
        lines = []
        while t <= tend:
            x = self.x_of(t)
            major = abs(round(t / step) * step - t) < minor * 0.1
            lines.append(QLineF(x, RULER_H - (12 if major else 5), x, RULER_H))
            t += minor
        p.drawLines(lines)
        p.setPen(QColor(S.MUTED))
        f = p.font()
        f.setPointSizeF(max(7.0, f.pointSizeF() - 1))
        p.setFont(f)
        t = math.floor(self.offset / step) * step
        while t <= tend:
            x = self.x_of(t)
            if x >= -40:
                p.drawText(QPointF(x + 3, 12), fmt_time(t, step))
            t += step
        # петля
        loop = self.ctrl.playback.loop
        if loop:
            sr = self.ctrl.playback.sr
            a, b = loop[0] / sr, loop[1] / sr
            p.fillRect(QRectF(self.x_of(a), RULER_H - 6, self.x_of(b) - self.x_of(a), 4), QColor(S.OK))
        sel = self.ctrl.selection.time_range
        if sel:
            p.fillRect(QRectF(self.x_of(sel[0]), RULER_H - 3, self.x_of(sel[1]) - self.x_of(sel[0]), 3),
                       QColor(S.ACCENT))
        # отметки перегрузки мастера
        env = self.ctrl.peak_env
        if env.size:
            step_s = 0.05
            i0 = max(0, int(self.offset / step_s))
            i1 = min(env.size, int(tend / step_s) + 1)
            seg = env[i0:i1]
            for k in np.nonzero(seg >= 0.891)[0]:
                v = seg[k]
                x = self.x_of((i0 + k) * step_s)
                p.fillRect(QRectF(x, 0, max(2.0, step_s * self.px_per_s), 4),
                           QColor(S.DANGER if v >= 1.0 else S.ACCENT2))
        # защищённые/зафиксированные участки на линейке
        for pr in self.ctrl.project.protections:
            e = pr.end if pr.end is not None else tend + 10
            p.fillRect(QRectF(self.x_of(pr.start), RULER_H - 10, self.x_of(e) - self.x_of(pr.start), 3),
                       S.qcolor(S.ACCENT2, 200))
        for fr in self.ctrl.project.frozen:
            p.fillRect(QRectF(self.x_of(fr.start), RULER_H - 14, self.x_of(fr.end) - self.x_of(fr.start), 3),
                       S.qcolor("#7ec8ff", 220))
        p.setPen(QColor("#3a4049"))
        p.drawLine(0, RULER_H - 1, W, RULER_H - 1)

    def _paint_protections(self, p: QPainter, H: int):
        proj = self.ctrl.project
        tend = self.offset + self.visible_seconds() + 10
        for pr in proj.protections:
            e = pr.end if pr.end is not None else tend
            x0, x1 = self.x_of(pr.start), self.x_of(e)
            rows = [self.track_index(pr.track_id)] if pr.track_id else list(range(len(proj.tracks)))
            for r in rows:
                if r < 0:
                    continue
                rect = QRectF(x0, self.row_top(r), x1 - x0, ROW_H)
                p.fillRect(rect, QBrush(S.qcolor(S.ACCENT2, 55), Qt.BrushStyle.BDiagPattern))
            p.setPen(S.qcolor(S.ACCENT2, 220))
            p.drawText(QPointF(x0 + 4, RULER_H + 12), "защищено")
        for fr in proj.frozen:
            x0, x1 = self.x_of(fr.start), self.x_of(fr.end)
            rect = QRectF(x0, RULER_H, x1 - x0, max(0, H - RULER_H))
            p.fillRect(rect, QBrush(S.qcolor("#7ec8ff", 60), Qt.BrushStyle.FDiagPattern))
            p.setPen(S.qcolor("#7ec8ff", 230))
            p.drawText(QPointF(x0 + 4, RULER_H + 26), "PCM зафиксирован")

    def _wave_columns(self, clip, peaks, x0: int, x1: int):
        sr = self.ctrl.project.sample_rate
        spp = sr / self.px_per_s
        block, lv = peaks.level_for(spp)
        cols = np.arange(x0, x1)
        t = self.offset + cols / self.px_per_s
        src_s = (clip.src_in + (t - clip.start)) * sr
        b0 = np.clip((src_s / block).astype(np.int64), 0, lv.shape[0] - 1)
        b1 = np.clip(((src_s + spp) / block).astype(np.int64) + 1, 1, lv.shape[0])
        b1 = np.maximum(b1, b0 + 1)
        mins = lv[:, :, 0].min(axis=1)
        maxs = lv[:, :, 1].max(axis=1)
        if b1[-1] - b0[0] < 4 * len(cols):
            lo = np.array([mins[a:b].min() for a, b in zip(b0, b1)])
            hi = np.array([maxs[a:b].max() for a, b in zip(b0, b1)])
        else:
            idx = np.concatenate([b0, [b1[-1]]])
            mins_p = np.concatenate([mins, [0.0]])
            maxs_p = np.concatenate([maxs, [0.0]])
            lo = np.minimum.reduceat(mins_p, idx)[:-1]
            hi = np.maximum.reduceat(maxs_p, idx)[:-1]
        return cols, lo, hi

    def _paint_clip(self, p: QPainter, row: int, clip, color: QColor, selected: bool):
        top = self.row_top(row) + 4
        h = ROW_H - 8
        x0, x1 = self.x_of(clip.start), self.x_of(clip.end)
        if x1 < 0 or x0 > self.width():
            return
        rect = QRectF(x0, top, max(2.0, x1 - x0), h)
        src = self.ctrl.project.source(clip.source_id)
        missing = src is None or src.id in self.ctrl.store.missing
        fill = QColor(color)
        fill.setAlpha(55 if not selected else 85)
        p.fillRect(rect, fill)
        peaks = self.ctrl.store.peaks(src) if (src and not missing) else None
        if peaks is not None:
            vx0 = int(max(0, x0))
            vx1 = int(min(self.width(), x1))
            if vx1 > vx0:
                cols, lo, hi = self._wave_columns(clip, peaks, vx0, vx1)
                mid = top + h / 2
                amp = h / 2 - 2
                g = 10 ** (clip.gain_db / 20)
                lines = [QLineF(float(c), mid - min(1.0, hv * g) * amp, float(c), mid - max(-1.0, lv * g) * amp)
                         for c, lv, hv in zip(cols, lo, hi)]
                p.setPen(QColor(color).lighter(140))
                p.drawLines(lines)
        # огибающая перехода A→B на клипе
        music = self.ctrl.project.music_for_clip(clip.id)
        if music:
            vx0 = max(0.0, x0)
            vx1 = min(float(self.width()), x1)
            if vx1 > vx0:
                xs = np.arange(vx0, vx1, 2.0)
                ts = self.offset + xs / self.px_per_s
                n = np.round(ts * self.ctrl.project.sample_rate).astype(np.int64)
                env = A.clip_envelope(clip, self.ctrl.project.sample_rate, n, music) / (10 ** (clip.gain_db / 20))
                poly = QPolygonF([QPointF(x, top + (1 - min(1.0, e)) * (h - 4) + 2) for x, e in zip(xs, env)])
                p.setPen(QPen(QColor("#ffd27f"), 1.6))
                p.drawPolyline(poly)
        p.setPen(QPen(QColor(S.DANGER) if missing else (QColor(S.TEXT) if selected else QColor(color)),
                      2 if selected else 1))
        p.drawRect(rect)
        p.setPen(QColor(S.TEXT))
        label = clip.name or (src.name if src else "?")
        if missing:
            label += " — нет файла"
        if clip.locked:
            label = "[зафиксирован] " + label
        p.drawText(QRectF(x0 + 5, top + 2, max(0.0, x1 - x0 - 10), 16), Qt.AlignmentFlag.AlignLeft, label)
        # фейды
        p.setPen(QPen(S.qcolor(S.TEXT, 120), 1))
        fi = clip.fade_in * self.px_per_s
        fo = clip.fade_out * self.px_per_s
        if fi > 3:
            p.drawLine(QLineF(x0, top + h, x0 + fi, top))
        if fo > 3:
            p.drawLine(QLineF(x1 - fo, top, x1, top + h))

    def _paint_space_bands(self, p: QPainter, row: int, track):
        sel = self.ctrl.selection.transition_id
        for x in self.ctrl.project.transitions_on_track(track.id):
            top = self.row_top(row)
            x0, x1 = self.x_of(x.start), self.x_of(x.end)
            rect = QRectF(x0, top + 1, x1 - x0, ROW_H - 2)
            p.fillRect(rect, S.qcolor(S.ACCENT2, 55 if x.id == sel else 32))
            pen = QPen(QColor(S.ACCENT2), 2 if x.id == sel else 1, Qt.PenStyle.DashLine)
            p.setPen(pen)
            p.drawRect(rect)

    def _paint_space_curves(self, p: QPainter, row: int, track):
        sel = self.ctrl.selection.transition_id
        for x in self.ctrl.project.transitions_on_track(track.id):
            top = self.row_top(row)
            x0, x1 = self.x_of(x.start), self.x_of(x.end)
            p.setPen(QColor(S.ACCENT2))
            lab = x.name + (" · утверждён" if x.approved else "")
            p.drawText(QRectF(x0 + 4, top + ROW_H - 18, max(0.0, x1 - x0 - 8), 16), Qt.AlignmentFlag.AlignLeft, lab)
            if x.id != sel or x1 - x0 < 8:
                continue
            xs = np.arange(max(x0, 0.0), min(x1, float(self.width())), 2.0)
            if xs.size < 2:
                continue
            ts = self.offset + xs / self.px_per_s
            xs_all = self.ctrl.project.transitions_on_track(track.id)
            for prm in SPACE_PARAMS:
                if prm == "reverb_db":
                    continue
                a, b = x.from_state.values[prm], x.to_state.values[prm]
                if abs(a - b) < 1e-9:
                    continue
                v = A.space_param(xs_all, prm, ts)
                dom = PARAM_INFO[prm]["domain"]
                if dom == "log":
                    u = (np.log(v) - math.log(a)) / (math.log(b) - math.log(a))
                else:
                    u = (v - a) / (b - a)
                ys = top + 6 + (1 - np.clip(u, 0, 1)) * (ROW_H - 28)
                p.setPen(QPen(QColor(S.GROUP_COLORS[prm]), 1.4))
                p.drawPolyline(QPolygonF([QPointF(float(xx), float(yy)) for xx, yy in zip(xs, ys)]))
            for r in A.track_rooms(xs_all):
                a, b = x.from_state.effective_reverb(r), x.to_state.effective_reverb(r)
                if abs(a - b) < 1e-9:
                    continue
                v = A.space_param(xs_all, "reverb_db", ts, room=r)
                u = (v - a) / (b - a)
                ys = top + 6 + (1 - np.clip(u, 0, 1)) * (ROW_H - 28)
                p.setPen(QPen(QColor(S.GROUP_COLORS["reverb_db"]), 1.4, Qt.PenStyle.DashLine))
                p.drawPolyline(QPolygonF([QPointF(float(xx), float(yy)) for xx, yy in zip(xs, ys)]))

    def _paint_music(self, p: QPainter):
        proj = self.ctrl.project
        sel = self.ctrl.selection.transition_id
        for m in proj.music_transitions:
            a = proj.clip(m.clip_a_id)
            b = proj.clip(m.clip_b_id)
            if a is None or b is None:
                continue
            ra = self.track_index(proj.clip_track(a.id).id)
            rb = self.track_index(proj.clip_track(b.id).id)
            r0, r1 = min(ra, rb), max(ra, rb)
            o0, o1 = m.params.overlap
            x0, x1 = self.x_of(o0), self.x_of(o1)
            rect = QRectF(x0, self.row_top(r0) + 1, max(3.0, x1 - x0), (r1 - r0 + 1) * ROW_H - 2)
            p.fillRect(rect, S.qcolor("#9b7ee0", 55 if m.id == sel else 30))
            p.setPen(QPen(QColor("#9b7ee0"), 2 if m.id == sel else 1, Qt.PenStyle.DashLine))
            p.drawRect(rect)
            xs = self.x_of(m.params.switch_time)
            p.setPen(QPen(QColor("#c8b4ff"), 2))
            p.drawLine(QLineF(xs, self.row_top(r0), xs, self.row_top(r1) + ROW_H))
            p.drawText(QPointF(xs + 4, self.row_top(r1) + ROW_H - 6), "вход B" + (" · утверждён" if m.approved else ""))

    def _paint_beats(self, p: QPainter):
        proj = self.ctrl.project
        for cid in self.ctrl.selection.clip_ids[:2]:
            c = proj.clip(cid)
            if c is None:
                continue
            src = proj.source(c.source_id)
            if src is None or src.beats is None:
                continue
            row = self.track_index(proj.clip_track(cid).id)
            top = self.row_top(row)
            vis0, vis1 = self.offset, self.offset + self.visible_seconds()
            if (60.0 / max(src.beats.bpm, 1)) * self.px_per_s >= 4:
                p.setPen(QPen(S.qcolor(S.TEXT, 55), 1))
                lines = [QLineF(self.x_of(c.to_timeline(b)), top + 18, self.x_of(c.to_timeline(b)), top + ROW_H - 6)
                         for b in src.beats.beats if c.src_in <= b <= c.src_out and vis0 <= c.to_timeline(b) <= vis1]
                p.drawLines(lines)
            p.setPen(QPen(S.qcolor(S.TEXT, 150), 1.5))
            lines = [QLineF(self.x_of(c.to_timeline(b)), top + 14, self.x_of(c.to_timeline(b)), top + ROW_H - 4)
                     for b in src.beats.downbeats if c.src_in <= b <= c.src_out and vis0 <= c.to_timeline(b) <= vis1]
            p.drawLines(lines)
            p.setPen(QPen(QColor(S.ACCENT2), 2))
            phr = src.phrases if src.phrases else src.beats.phrase_starts()
            for b in phr:
                if c.src_in <= b <= c.src_out:
                    x = self.x_of(c.to_timeline(b))
                    p.drawLine(QLineF(x, top + 4, x, top + ROW_H - 4))

    # ---------------------------------------------------------------- hit testing
    def _hit(self, pos) -> tuple[str, object]:
        x, y = pos.x(), pos.y()
        if y < RULER_H:
            return "ruler", None
        row = self.row_of(y)
        proj = self.ctrl.project
        if row < 0 or row >= len(proj.tracks):
            return "empty", None
        track = proj.tracks[row]
        t = self.t_of(x)
        for m in proj.music_transitions:
            a, b = proj.clip(m.clip_a_id), proj.clip(m.clip_b_id)
            if a is None or b is None:
                continue
            rows = {self.track_index(proj.clip_track(a.id).id), self.track_index(proj.clip_track(b.id).id)}
            if row not in rows:
                continue
            if abs(self.x_of(m.params.switch_time) - x) <= 5:
                return "switch", m
        tracks_x = proj.transitions_on_track(track.id)
        for xt in tracks_x:
            x0, x1 = self.x_of(xt.start), self.x_of(xt.end)
            if abs(x - x0) <= 5:
                return "x_start", xt
            if abs(x - x1) <= 5:
                return "x_end", xt
        for m in proj.music_transitions:
            a, b = proj.clip(m.clip_a_id), proj.clip(m.clip_b_id)
            if a is None or b is None:
                continue
            rows = {self.track_index(proj.clip_track(a.id).id), self.track_index(proj.clip_track(b.id).id)}
            o0, o1 = m.params.overlap
            if row in rows and o0 <= t <= o1 and (y - self.row_top(row)) > ROW_H - 26:
                return "music", m
        for xt in tracks_x:
            if xt.start <= t <= xt.end and (y - self.row_top(row)) > ROW_H - 26:
                return "x_body", xt
        for c in reversed(track.clips):
            if c.start <= t <= c.end:
                return "clip", c
        for xt in tracks_x:
            if xt.start <= t <= xt.end:
                return "x_body", xt
        for m in proj.music_transitions:
            o0, o1 = m.params.overlap
            a, b = proj.clip(m.clip_a_id), proj.clip(m.clip_b_id)
            if a and b and o0 <= t <= o1:
                rows = {self.track_index(proj.clip_track(a.id).id), self.track_index(proj.clip_track(b.id).id)}
                if row in rows:
                    return "music", m
        return "lane", track

    # ---------------------------------------------------------------- mouse
    def mousePressEvent(self, ev):
        self.setFocus()
        pos = ev.position()
        t = max(0.0, self.t_of(pos.x()))
        kind, obj = self._hit(pos)
        ctrl = self.ctrl
        if ev.button() == Qt.MouseButton.RightButton:
            oid = getattr(obj, "id", "") if obj is not None else ""
            self.contextRequested.emit(kind, oid, ev.globalPosition().toPoint(), t)
            return
        if ev.button() != Qt.MouseButton.LeftButton:
            return
        shift = bool(ev.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        ctrl_mod = bool(ev.modifiers() & Qt.KeyboardModifier.ControlModifier)
        if kind == "ruler" or (shift and kind in ("lane", "clip", "x_body", "music")):
            self._drag = {"mode": "range", "t0": t, "moved": False}
            if kind == "ruler" and not shift:
                self.seekRequested.emit(t)
            return
        if kind == "clip":
            track = ctrl.project.clip_track(obj.id)
            if ctrl_mod:
                ids = list(ctrl.selection.clip_ids)
                if obj.id in ids:
                    ids.remove(obj.id)
                else:
                    ids = (ids + [obj.id])[-2:]
                ctrl.select(clip_ids=ids, track_id=track.id, transition_id=None)
            else:
                ctrl.select(clip_ids=[obj.id], track_id=track.id, transition_id=None)
                self._drag = {"mode": "clip", "id": obj.id, "t0": t, "orig": obj.start, "row": self.row_of(pos.y()),
                              "moved": False, "locked": obj.locked}
        elif kind in ("x_start", "x_end", "x_body"):
            ctrl.select(transition_id=obj.id, track_id=obj.track_id)
            self._drag = {"mode": kind, "id": obj.id, "t0": t, "start": obj.start, "end": obj.end, "moved": False}
        elif kind in ("switch", "music"):
            ctrl.select(transition_id=obj.id)
            if kind == "switch":
                self._drag = {"mode": "switch", "id": obj.id, "t0": t, "orig": obj.params.switch_time, "moved": False}
        elif kind == "lane":
            ctrl.select(track_id=obj.id, clip_ids=[], transition_id=None)
            self.seekRequested.emit(t)
        self.update()

    def mouseMoveEvent(self, ev):
        pos = ev.position()
        t = max(0.0, self.t_of(pos.x()))
        d = self._drag
        if d is None:
            kind, _ = self._hit(pos)
            if kind in ("x_start", "x_end", "switch"):
                self.setCursor(QCursor(Qt.CursorShape.SizeHorCursor))
            elif kind == "clip":
                self.setCursor(QCursor(Qt.CursorShape.OpenHandCursor))
            else:
                self.unsetCursor()
            return
        d["moved"] = d["moved"] or abs(t - d["t0"]) * self.px_per_s > 3
        if not d["moved"]:
            return
        if d["mode"] == "range":
            a, b = sorted((d["t0"], t))
            self.ctrl.select(time_range=(a, b))
        elif d["mode"] == "clip" and not d["locked"]:
            d["new"] = max(0.0, d["orig"] + (t - d["t0"]))
            d["new_row"] = max(0, min(len(self.ctrl.project.tracks) - 1, self.row_of(pos.y())))
            self._drop_hint = (d["new_row"], d["new"])
        elif d["mode"] in ("x_start", "x_end", "x_body"):
            dt = t - d["t0"]
            if d["mode"] == "x_start":
                d["new"] = (min(d["start"] + dt, d["end"] - 0.05), d["end"])
            elif d["mode"] == "x_end":
                d["new"] = (d["start"], max(d["end"] + dt, d["start"] + 0.05))
            else:
                d["new"] = (max(0.0, d["start"] + dt), max(0.0, d["start"] + dt) + d["end"] - d["start"])
            self._drop_hint = None
            self._preview_band = d["new"]
        elif d["mode"] == "switch":
            d["new"] = max(0.0, d["orig"] + (t - d["t0"]))
        self.update()

    def mouseReleaseEvent(self, ev):
        d = self._drag
        self._drag = None
        self._drop_hint = None
        if d is None or not d.get("moved"):
            if d and d["mode"] == "range":
                self.ctrl.select(time_range=None)
            self.update()
            return
        ctrl = self.ctrl
        if d["mode"] == "clip" and "new" in d:
            tr = ctrl.project.tracks[d["new_row"]]
            params = {}
            if ctrl.project.clip_track(d["id"]).id != tr.id:
                params["track_id"] = tr.id
            ctrl.ui_ops([op("clip.move", target={"clip_id": d["id"]},
                            time={"coord": "timeline", "unit": "s", "start": round(d["new"], 4)},
                            params=params or None)], text="Перемещение клипа")
        elif d["mode"] == "clip" and d.get("locked"):
            ctrl.message.emit("warn", "Клип зафиксирован — снимите фиксацию в свойствах клипа, чтобы перемещать его.")
        elif d["mode"] in ("x_start", "x_end", "x_body") and "new" in d:
            s, e = d["new"]
            ctrl.ui_ops([op("space.set_timing", target={"transition_id": d["id"]},
                            time={"coord": "timeline", "unit": "s", "start": round(s, 4), "end": round(e, 4)})],
                        text="Изменение границ перехода")
        elif d["mode"] == "switch" and "new" in d:
            ctrl.ui_ops([op("music.set", target={"transition_id": d["id"]},
                            params={"shift": round(d["new"] - d["orig"], 4)})], text="Сдвиг входа B")
        self.update()

    def mouseDoubleClickEvent(self, ev):
        kind, obj = self._hit(ev.position())
        if kind in ("clip", "x_body", "music", "switch"):
            self.contextRequested.emit("open:" + kind, obj.id, ev.globalPosition().toPoint(), self.t_of(ev.position().x()))

    def wheelEvent(self, ev):
        delta = ev.angleDelta()
        if ev.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.25 if delta.y() > 0 else 0.8
            self.set_zoom(self.px_per_s * factor, self.t_of(ev.position().x()))
        else:
            dy = delta.y() if delta.y() else delta.x()
            self.offset = max(0.0, self.offset - dy / 120 * self.visible_seconds() * 0.1)
            self.viewChanged.emit()
            self.update()
        ev.accept()

    # ---------------------------------------------------------------- drag & drop
    def dragEnterEvent(self, ev):
        md = ev.mimeData()
        if md.hasFormat(SOURCE_MIME) or (md.hasUrls() and any(u.isLocalFile() for u in md.urls())):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):
        pos = ev.position()
        self._drop_hint = (self.row_of(pos.y()), max(0.0, self.t_of(pos.x())))
        self.update()
        ev.acceptProposedAction()

    def dragLeaveEvent(self, ev):
        self._drop_hint = None
        self.update()

    def dropEvent(self, ev):
        pos = ev.position()
        row = self.row_of(pos.y())
        t = max(0.0, round(self.t_of(pos.x()), 3))
        self._drop_hint = None
        proj = self.ctrl.project
        track_id = proj.tracks[row].id if 0 <= row < len(proj.tracks) else None
        md = ev.mimeData()
        if md.hasFormat(SOURCE_MIME):
            sid = bytes(md.data(SOURCE_MIME)).decode("utf-8")
            tgt = {"source_id": sid}
            if track_id:
                tgt["track_id"] = track_id
            self.ctrl.ui_ops([op("clip.add", target=tgt, time={"coord": "timeline", "unit": "s", "start": t})],
                             text="Клип из библиотеки")
        elif md.hasUrls():
            paths = [u.toLocalFile() for u in md.urls() if u.isLocalFile()]
            audio = [p for p in paths if Path(p).suffix.lower() in AUDIO_EXTENSIONS]
            if audio:
                self.ctrl.import_files(self.window(), audio, track_id=track_id, at=t)
            others = [p for p in paths if p not in audio]
            if others:
                self.window().handle_dropped_other(others)
        ev.acceptProposedAction()
        self.update()


class TrackHeader(QFrame):
    def __init__(self, ctrl, track, index, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        self.track_id = track.id
        self.setFixedHeight(ROW_H)
        self.setFrameShape(QFrame.Shape.NoFrame)
        color = S.TRACK_COLORS[index % len(S.TRACK_COLORS)]
        self.setStyleSheet(f"TrackHeader {{ background: {S.PANEL}; border-left: 4px solid {color}; border-bottom: 1px solid #30353c; }}")
        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 4, 6, 4)
        lay.setSpacing(3)
        self.name = QLineEdit(track.name)
        self.name.setFrame(False)
        self.name.setStyleSheet("background: transparent; font-weight: 600;")
        self.name.editingFinished.connect(self._rename)
        lay.addWidget(self.name)
        row = QHBoxLayout()
        row.setSpacing(4)
        self.mute = QPushButton("M")
        self.mute.setObjectName("small")
        self.mute.setCheckable(True)
        self.mute.setChecked(track.mute)
        self.mute.setToolTip("Выключить дорожку (M)")
        self.mute.clicked.connect(lambda v: self.ctrl.ui_ops([op("track.set", target={"track_id": self.track_id},
                                                                params={"mute": bool(v)})]))
        self.solo = QPushButton("S")
        self.solo.setObjectName("small")
        self.solo.setCheckable(True)
        self.solo.setChecked(track.solo)
        self.solo.setToolTip("Соло (S)")
        self.solo.clicked.connect(lambda v: self.ctrl.ui_ops([op("track.set", target={"track_id": self.track_id},
                                                                params={"solo": bool(v)})]))
        self.gain = QDoubleSpinBox()
        self.gain.setRange(-60.0, 12.0)
        self.gain.setDecimals(1)
        self.gain.setSingleStep(0.5)
        self.gain.setSuffix(" дБ")
        self.gain.setValue(track.gain_db)
        self.gain.setToolTip("Уровень дорожки")
        self.gain.setKeyboardTracking(False)
        self.gain.valueChanged.connect(self._gain)
        row.addWidget(self.mute)
        row.addWidget(self.solo)
        row.addWidget(self.gain, 1)
        lay.addLayout(row)
        self.info = QLabel()
        self.info.setObjectName("hint")
        lay.addWidget(self.info)
        self.refresh(track)

    def refresh(self, track):
        for w, v in ((self.mute, track.mute), (self.solo, track.solo)):
            w.blockSignals(True)
            w.setChecked(v)
            w.blockSignals(False)
        self.gain.blockSignals(True)
        self.gain.setValue(track.gain_db)
        self.gain.blockSignals(False)
        if self.name.text() != track.name and not self.name.hasFocus():
            self.name.setText(track.name)
        nx = len(self.ctrl.project.transitions_on_track(track.id))
        chans = {self.ctrl.project.source(c.source_id).channels for c in track.clips if self.ctrl.project.source(c.source_id)}
        mode = "стерео" if 2 in chans else ("моно" if chans else "пусто")
        self.info.setText(f"{len(track.clips)} клип., {mode}" + (f", переходов: {nx}" if nx else ""))

    def _rename(self):
        t = self.ctrl.project.track(self.track_id)
        if t and self.name.text().strip() and self.name.text() != t.name:
            self.ctrl.ui_ops([op("track.set", target={"track_id": self.track_id}, params={"name": self.name.text().strip()})])

    def _gain(self, v):
        t = self.ctrl.project.track(self.track_id)
        if t and abs(t.gain_db - v) > 1e-9:
            self.ctrl.ui_ops([op("track.set", target={"track_id": self.track_id}, params={"gain_db": float(v)})])

    def mousePressEvent(self, ev):
        self.ctrl.select(track_id=self.track_id)


class TimelinePanel(QWidget):
    """Заголовки дорожек + холст + горизонтальная прокрутка."""

    def __init__(self, ctrl, parent=None):
        super().__init__(parent)
        self.ctrl = ctrl
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        self.scroll = QScrollArea()
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        inner = QWidget()
        h = QHBoxLayout(inner)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        self.headers = QWidget()
        self.headers.setFixedWidth(HEADER_W)
        self.hlay = QVBoxLayout(self.headers)
        self.hlay.setContentsMargins(0, 0, 0, 0)
        self.hlay.setSpacing(0)
        top = QLabel("  Дорожки")
        top.setFixedHeight(RULER_H)
        top.setObjectName("hint")
        top.setStyleSheet(f"background: {S.PANEL}; border-bottom: 1px solid #3a4049;")
        self.hlay.addWidget(top)
        self.hlay.addStretch(1)
        self.canvas = TimelineCanvas(ctrl)
        h.addWidget(self.headers)
        h.addWidget(self.canvas, 1)
        self.scroll.setWidget(inner)
        outer.addWidget(self.scroll, 1)
        self.hbar = QScrollBar(Qt.Orientation.Horizontal)
        self.hbar.valueChanged.connect(self._scrolled)
        outer.addWidget(self.hbar)
        self.canvas.viewChanged.connect(self._sync_bar)
        self._headers: list[TrackHeader] = []
        self._track_ids: list[str] = []

    def refresh(self):
        proj = self.ctrl.project
        ids = [t.id for t in proj.tracks]
        if ids != self._track_ids:
            for hd in self._headers:
                hd.setParent(None)
                hd.deleteLater()
            self._headers = []
            for i, t in enumerate(proj.tracks):
                hd = TrackHeader(self.ctrl, t, i)
                self.hlay.insertWidget(1 + i, hd)
                self._headers.append(hd)
            self._track_ids = ids
        else:
            for hd, t in zip(self._headers, proj.tracks):
                hd.refresh(t)
        n = max(1, len(proj.tracks))
        self.canvas.setMinimumHeight(RULER_H + n * ROW_H + 8)
        self.headers.setMinimumHeight(RULER_H + n * ROW_H + 8)
        self._sync_bar()
        self.canvas.update()

    def _sync_bar(self):
        total = max(self.ctrl.project.content_end() + 10.0, self.canvas.offset + self.canvas.visible_seconds())
        vis = self.canvas.visible_seconds()
        self.hbar.blockSignals(True)
        self.hbar.setRange(0, int(max(0.0, total - vis) * 100))
        self.hbar.setPageStep(int(vis * 100))
        self.hbar.setSingleStep(max(1, int(vis * 10)))
        self.hbar.setValue(int(self.canvas.offset * 100))
        self.hbar.blockSignals(False)

    def _scrolled(self, v):
        self.canvas.offset = v / 100.0
        self.canvas.update()
