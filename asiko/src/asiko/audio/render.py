"""Офлайн-рендер ASIKO — единственная модель обработки для прослушивания и экспорта.

Рендер идёт фрагментами по фиксированной сетке шкалы (CHUNK сэмплов от нуля).
Для фрагмента [s0, s1) каждый блок получает вход с нужным контекстом:

    клип:     источник × огибающая → (КИХ-фильтр клипа) → (+ хвост эффекта)
    дорожка:  Σ клипов → ФВЧ → ФНЧ → смешивание прямой/обработанный → ширина →
              панорама → усиление → прямой звук + реверберация (моно-посыл, стерео ИХ)
    мастер:   Σ дорожек × уровень (mute/solo) → зафиксированные PCM-участки

Все блоки с памятью конечной длины, поэтому фрагмент — чистая функция описания
проекта и номера фрагмента: результат детерминирован, кэш корректен, рендер
участка совпадает с соответствующей частью полного рендера побитово.
"""
from __future__ import annotations

import hashlib
import json
import math
import threading
from collections import OrderedDict
from dataclasses import asdict, dataclass, field

import numpy as np

from ..model import automation as A
from ..model.presets import REVERB_OFF_DB
from ..model.project import Clip, MusicTransition, Project, SpaceTransition, Track
from . import dsp
from .io import Cancelled

ENGINE_VERSION = "asiko-dsp-1.0"
CHUNK = 1 << 18
GATE_FADE_S = 0.010   # затухание источников на конце участка при экспорте «с хвостами»


@dataclass
class ClipPlan:
    clip: Clip
    music: list[MusicTransition]
    audio: np.ndarray | None          # (frames, ch) float32 на частоте проекта, None — нет файла
    start: int
    src_in: int
    length: int
    channels: int
    filt: bool                         # есть автоматизация фильтра клипа
    tail_room: str | None              # хвост добавленного эффекта
    key: dict
    bypass: bool = False

    @property
    def end(self) -> int:
        return self.start + self.length

    def lookback(self, sr: int) -> int:
        lb = dsp.taps_half(sr) if self.filt else 0
        if self.tail_room:
            lb = max(lb, dsp.ir_length(self.tail_room, sr) - 1 + (dsp.taps_half(sr) if self.filt else 0))
        return lb

    def lookahead(self, sr: int) -> int:
        return dsp.taps_half(sr) if self.filt else 0

    def audible_end(self, sr: int) -> int:
        return self.end + self.lookahead(sr) + (dsp.ir_length(self.tail_room, sr) if self.tail_room else 0)


@dataclass
class TrackPlan:
    track: Track
    clips: list[ClipPlan]
    transitions: list[SpaceTransition]
    rooms: list[str]
    channels: int
    gain: float
    audible: bool
    space_active: bool
    slope: int

    def chain_lookback(self, sr: int) -> int:
        if not self.space_active:
            return 0
        lb = 2 * dsp.taps_half(sr)
        if self.rooms:
            lb += max(dsp.ir_length(r, sr) for r in self.rooms) - 1
        return lb

    def chain_lookahead(self, sr: int) -> int:
        return 2 * dsp.taps_half(sr) if self.space_active else 0

    def tail_samples(self, sr: int) -> int:
        if not self.space_active:
            return 0
        t = 2 * dsp.taps_half(sr)
        if self.rooms:
            t += max(dsp.ir_length(r, sr) for r in self.rooms)
        return t


@dataclass
class FrozenPlan:
    id: str
    start: int
    end: int
    margin: int
    data: np.ndarray | None      # (end-start + 2*margin, 2) float32
    key: str


@dataclass
class RenderPlan:
    sr: int
    tracks: list[TrackPlan]
    frozen: list[FrozenPlan]
    gate_end: int | None = None        # источники молчат после этого сэмпла (хвосты участка)
    solo_active: bool = False
    content_end: int = 0
    missing_sources: list[str] = field(default_factory=list)

    def end_with_tails(self) -> int:
        end = 0
        for tp in self.tracks:
            if not tp.audible:
                continue
            for cp in tp.clips:
                end = max(end, cp.audible_end(self.sr) + tp.tail_samples(self.sr))
        return end


def _clip_key(cp_clip: Clip, music: list[MusicTransition], src_key: str | None, bypass: bool) -> dict:
    return {"clip": asdict(cp_clip), "music": [asdict(m.params) | {"a": m.clip_a_id, "b": m.clip_b_id}
                                              for m in music], "src": src_key, "bypass": bypass}


def compile_plan(project: Project, store, *, bypass: bool = False, music_override: dict | None = None,
                 gate_end_s: float | None = None, apply_frozen: bool = True, frozen_audio=None) -> RenderPlan:
    """Снимок проекта → неизменяемый план рендера (безопасно передавать в поток)."""
    project = project.clone()
    sr = project.sample_rate
    if music_override:
        for mt in project.music_transitions:
            if mt.id in music_override:
                from ..commands import music as music_mod
                mt.params = music_override[mt.id]
                try:
                    music_mod.apply_layout(project, mt, allow_locked={mt.clip_a_id, mt.clip_b_id})
                except ValueError:
                    pass
    solo = any(t.solo for t in project.tracks)
    tracks = []
    missing = []
    for t in project.tracks:
        cps = []
        ch = 1
        for c in t.clips:
            src = project.source(c.source_id)
            audio = store.load(src) if src is not None else None
            if audio is None:
                missing.append(c.source_id)
            nch = src.channels if src else 1
            ch = max(ch, nch)
            music = project.music_for_clip(c.id)
            st, si, ln = A.clip_bounds(c, sr)
            if audio is not None:
                ln = max(0, min(ln, audio.shape[0] - si))
            filt = (not bypass) and any(m.clip_a_id == c.id and m.params.overlap_filter != "none" for m in music)
            tail = None
            if not bypass:
                for m in music:
                    if m.clip_a_id == c.id and m.params.a_tail == "effect":
                        tail = m.params.effect_tail_room
            cps.append(ClipPlan(clip=c, music=music, audio=audio, start=st, src_in=si, length=ln, channels=nch,
                                filt=filt, tail_room=tail,
                                key=_clip_key(c, music, store.key_for(src) if src else None, bypass),
                                bypass=bypass))
        xs = [] if bypass else project.transitions_on_track(t.id)
        rooms = A.track_rooms(xs)
        slope = max((x.filter_slope for x in xs), default=12)
        audible = (not t.mute) and (t.solo or not solo)
        tracks.append(TrackPlan(track=t, clips=cps, transitions=xs, rooms=rooms, channels=ch,
                                gain=10.0 ** (t.gain_db / 20.0), audible=audible, space_active=bool(xs),
                                slope=slope))
    frozen = []
    if apply_frozen and not solo and not bypass and frozen_audio is not None:
        for fr in project.frozen:
            if fr.sample_rate != sr:
                continue
            data = frozen_audio(fr)
            frozen.append(FrozenPlan(id=fr.id, start=int(round(fr.start * sr)), end=int(round(fr.end * sr)),
                                     margin=int(round(fr.margin * sr)), data=data, key=fr.sha256))
    plan = RenderPlan(sr=sr, tracks=tracks, frozen=frozen, solo_active=solo,
                      content_end=int(math.ceil(project.content_end() * sr)), missing_sources=sorted(set(missing)))
    if gate_end_s is not None:
        plan.gate_end = int(round(gate_end_s * sr))
    return plan


# --------------------------------------------------------------------------- clip & track blocks


def _clip_input(cp: ClipPlan, a: int, b: int, sr: int, gate_end: int | None) -> np.ndarray:
    """Сэмплы клипа × огибающая на абсолютном интервале [a, b) (вне клипа — нули)."""
    n = b - a
    out = np.zeros((n, cp.channels))
    if cp.audio is None or cp.length <= 0:
        return out
    lo = max(a, cp.start)
    hi = min(b, cp.end)
    if gate_end is not None:
        hi = min(hi, gate_end)
    if hi <= lo:
        return out
    s0 = cp.src_in + (lo - cp.start)
    s1 = cp.src_in + (hi - cp.start)
    seg = np.asarray(cp.audio[s0:s1], dtype=np.float64)
    if seg.shape[0] < hi - lo:
        seg = np.concatenate([seg, np.zeros((hi - lo - seg.shape[0], seg.shape[1]))], axis=0)
    idx = np.arange(lo, hi, dtype=np.int64)
    env = A.clip_envelope(cp.clip, sr, idx, cp.music, bypass=cp.bypass)
    if gate_end is not None:
        gf = max(1, int(round(GATE_FADE_S * sr)))
        env = env * A.crossfade_curve("smooth", (gate_end - idx - 0.5) / gf, True)
    out[lo - a:hi - a] = seg * env[:, None]
    return out


def render_clip(cp: ClipPlan, a: int, b: int, sr: int, gate_end: int | None) -> np.ndarray:
    """Выход клипа на [a, b): с фильтром наложения и хвостом добавленного эффекта."""
    h = dsp.taps_half(sr)
    tail_lb = dsp.ir_length(cp.tail_room, sr) - 1 if cp.tail_room else 0
    fa = a - tail_lb   # фильтрованный сигнал нужен с этого сэмпла (для хвоста)
    if cp.filt:
        x = _clip_input(cp, fa - h, b + h, sr, gate_end)
        idx = np.arange(fa, b, dtype=np.int64)
        kind, vals = A.clip_filter(cp.clip, idx, sr, cp.music)
        y = dsp.fir_bank(kind, 12, sr, x, vals)
    else:
        y = _clip_input(cp, fa, b, sr, gate_end)
    if cp.tail_room:
        idx = np.arange(fa, b, dtype=np.int64)
        room, send = A.clip_effect_tail(cp.clip, idx, sr, cp.music)
        mono = y.mean(axis=1) * send
        wet = dsp.convolve_ir(mono, dsp.room_ir(room, sr))
        dry = y[tail_lb:]
        if dry.shape[1] == 1:
            dry = np.repeat(dry, 2, axis=1)
        return dry + wet
    return y


def _clip_overlaps(cp: ClipPlan, a: int, b: int, sr: int) -> bool:
    """Звучит ли выход клипа (с учётом фильтра и хвоста эффекта) на [a, b)."""
    lo = cp.start - cp.lookahead(sr)
    hi = cp.audible_end(sr)
    return cp.audio is not None and cp.length > 0 and lo < b and hi > a


def track_input(tp: TrackPlan, a: int, b: int, sr: int, gate_end: int | None) -> np.ndarray:
    ch = tp.channels
    if any(cp.tail_room for cp in tp.clips):
        ch = 2
    x = np.zeros((b - a, ch))
    for cp in tp.clips:
        if not _clip_overlaps(cp, a, b, sr):
            continue
        y = render_clip(cp, a, b, sr, gate_end)
        if y.shape[1] == 1 and ch == 2:
            y = np.repeat(y, 2, axis=1)
        x += y
    return x


def render_track(tp: TrackPlan, s0: int, s1: int, sr: int, gate_end: int | None = None) -> np.ndarray:
    """Стерео выход дорожки (до уровня фейдера) на [s0, s1)."""
    if not tp.space_active:
        x = track_input(tp, s0, s1, sr, gate_end)
        if x.shape[1] == 1:
            x = np.repeat(x, 2, axis=1)  # моно на стерео-шину без обработки: оба канала
        return x
    h = dsp.taps_half(sr)
    L = max((dsp.ir_length(r, sr) for r in tp.rooms), default=1) - 1 if tp.rooms else 0
    a = s0 - L
    x = track_input(tp, a - 2 * h, s1 + 2 * h, sr, gate_end)
    t_hp = np.arange(a - h, s1 + h, dtype=np.int64) / sr
    hp = A.space_param(tp.transitions, "hp_hz", t_hp)
    y = dsp.fir_bank("hp", tp.slope, sr, x, hp)                     # [a-h, s1+h)
    t = np.arange(a, s1, dtype=np.int64) / sr
    lp = A.space_param(tp.transitions, "lp_hz", t)
    y = dsp.fir_bank("lp", tp.slope, sr, y, lp)                     # [a, s1)
    dry = x[2 * h:2 * h + (s1 - a)]
    mix = A.space_param(tp.transitions, "color_mix", t)[:, None]
    z = dry + (y - dry) * mix                                       # когерентное линейное смешивание
    if z.shape[1] == 2:
        z = dsp.apply_width(z, A.space_param(tp.transitions, "width", t))
    z = dsp.apply_pan(z, A.space_param(tp.transitions, "pan", t))
    z *= A.db_to_lin(A.space_param(tp.transitions, "gain_db", t))[:, None]
    direct = A.db_to_lin(A.space_param(tp.transitions, "direct_db", t[L:]))[:, None]
    out = z[L:] * direct
    if tp.rooms:
        mono = 0.5 * (z[:, 0] + z[:, 1])
        for r in tp.rooms:
            Lr = dsp.ir_length(r, sr) - 1
            send_db = A.space_param(tp.transitions, "reverb_db", t, room=r)
            send = A.db_to_lin(send_db, floor_db=REVERB_OFF_DB) * mono
            seg = send[L - Lr:]
            if np.any(seg != 0.0):
                out += dsp.convolve_ir(seg, dsp.room_ir(r, sr))
    return out


# --------------------------------------------------------------------------- keys & cache


def _hash(obj) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _segment_key(tp: TrackPlan, a: float, b: float) -> list:
    """Описание автоматизации, от которого зависят значения параметров на [a, b] (секунды)."""
    xs = tp.transitions
    if not xs:
        return []
    a -= 0.05
    b += 0.05
    keep = [x for x in xs if x.start <= b and x.end >= a]
    prior = [x for x in xs if x.end < a]
    head = asdict(prior[-1].to_state) if prior else asdict(xs[0].from_state)
    out = [head]
    for x in keep:
        d = asdict(x)
        for k in ("name", "approved", "locked_params", "anchor_clip_id", "id"):
            d.pop(k, None)
        out.append(d)
    return out


def track_chunk_key(tp: TrackPlan, k: int, sr: int, gate_end: int | None) -> str:
    s0, s1 = k * CHUNK, (k + 1) * CHUNK
    a, b = s0 - tp.chain_lookback(sr), s1 + tp.chain_lookahead(sr)
    clips = [cp.key for cp in tp.clips if _clip_overlaps(cp, a, b, sr)]
    seg = _segment_key(tp, a / sr, b / sr) if tp.space_active else []
    max_lb = max((cp.lookback(sr) for cp in tp.clips), default=0)
    gate = gate_end if gate_end is not None and gate_end < b + max_lb + GATE_FADE_S * sr + 1 else None
    return _hash([ENGINE_VERSION, sr, CHUNK, k, tp.channels, tp.space_active, tp.rooms, tp.slope, clips, seg, gate])


class ChunkCache:
    """LRU-кэш фрагментов дорожек (float32)."""

    def __init__(self, max_bytes: int = 1 << 30):
        self.max_bytes = max_bytes
        self._d: OrderedDict[str, np.ndarray] = OrderedDict()
        self._bytes = 0
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def get(self, key: str):
        with self._lock:
            v = self._d.get(key)
            if v is not None:
                self._d.move_to_end(key)
                self.hits += 1
            else:
                self.misses += 1
            return v

    def put(self, key: str, arr: np.ndarray) -> None:
        with self._lock:
            if key in self._d:
                return
            self._d[key] = arr
            self._bytes += arr.nbytes
            while self._bytes > self.max_bytes and self._d:
                _, old = self._d.popitem(last=False)
                self._bytes -= old.nbytes

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0


# --------------------------------------------------------------------------- master


def _silent_track_chunk(tp: TrackPlan, k: int, sr: int) -> bool:
    s0, s1 = k * CHUNK, (k + 1) * CHUNK
    a = s0 - tp.chain_lookback(sr)
    b = s1 + tp.chain_lookahead(sr)
    return not any(_clip_overlaps(cp, a, b, sr) for cp in tp.clips)


def render_track_chunk(tp: TrackPlan, k: int, sr: int, cache: ChunkCache | None, gate_end: int | None) -> np.ndarray | None:
    if _silent_track_chunk(tp, k, sr):
        return None
    key = track_chunk_key(tp, k, sr, gate_end)
    if cache is not None:
        v = cache.get(key)
        if v is not None:
            return v
    y = render_track(tp, k * CHUNK, (k + 1) * CHUNK, sr, gate_end).astype(np.float32)
    if cache is not None:
        cache.put(key, y)
    return y


def master_chunk_key(plan: RenderPlan, k: int) -> str:
    parts = []
    for tp in plan.tracks:
        if not tp.audible or _silent_track_chunk(tp, k, plan.sr):
            continue
        parts.append((track_chunk_key(tp, k, plan.sr, plan.gate_end), round(tp.gain, 12)))
    fr = [(f.key, f.start, f.end) for f in plan.frozen if f.start - f.margin < (k + 1) * CHUNK and f.end + f.margin > k * CHUNK]
    return _hash([k, parts, fr])


def _apply_frozen(plan: RenderPlan, k: int, out: np.ndarray) -> None:
    s0, s1 = k * CHUNK, (k + 1) * CHUNK
    for f in plan.frozen:
        if f.data is None:
            continue
        lo = max(s0, f.start - f.margin)
        hi = min(s1, f.end + f.margin)
        if hi <= lo:
            continue
        idx = np.arange(lo, hi)
        fr = f.data[idx - (f.start - f.margin)].astype(np.float64)
        live = out[lo - s0:hi - s0].astype(np.float64)
        w = np.ones(idx.shape[0])
        if f.margin > 0:
            pre = idx < f.start
            w[pre] = 0.5 - 0.5 * np.cos(np.pi * (idx[pre] - (f.start - f.margin)) / f.margin)
            post = idx >= f.end
            w[post] = 0.5 - 0.5 * np.cos(np.pi * ((f.end + f.margin) - idx[post]) / f.margin)
        res = live * (1.0 - w[:, None]) + fr * w[:, None]
        inside = (idx >= f.start) & (idx < f.end)
        res[inside] = f.data[idx[inside] - (f.start - f.margin)]
        out[lo - s0:hi - s0] = res.astype(np.float32)


def render_master_chunk(plan: RenderPlan, k: int, cache: ChunkCache | None = None) -> np.ndarray:
    acc = np.zeros((CHUNK, 2))
    for tp in plan.tracks:
        if not tp.audible:
            continue
        y = render_track_chunk(tp, k, plan.sr, cache, plan.gate_end)
        if y is not None:
            acc += y.astype(np.float64) * tp.gain
    out = acc.astype(np.float32)
    if plan.frozen:
        _apply_frozen(plan, k, out)
    return out


def render_range(plan: RenderPlan, s0: int, s1: int, cache: ChunkCache | None = None, progress=None,
                 cancel=None) -> np.ndarray:
    """Рендер мастера на [s0, s1) (сэмплы шкалы) по сетке фрагментов."""
    s0 = max(0, s0)
    if s1 <= s0:
        return np.zeros((0, 2), np.float32)
    k0, k1 = s0 // CHUNK, (s1 - 1) // CHUNK
    out = np.zeros((s1 - s0, 2), np.float32)
    total = k1 - k0 + 1
    for i, k in enumerate(range(k0, k1 + 1)):
        if cancel is not None and cancel():
            raise Cancelled()
        y = render_master_chunk(plan, k, cache)
        c0, c1 = k * CHUNK, (k + 1) * CHUNK
        lo, hi = max(c0, s0), min(c1, s1)
        out[lo - s0:hi - s0] = y[lo - c0:hi - c0]
        if progress:
            progress((i + 1) / total)
    return out


@dataclass
class FrozenBoundary:
    id: str
    at: float
    mismatch_db: float


def frozen_boundary_report(plan: RenderPlan, cache: ChunkCache | None = None) -> list[FrozenBoundary]:
    """Сравнение «живого» рендера и зафиксированного PCM на границах участков."""
    if not plan.frozen:
        return []
    live_plan = RenderPlan(sr=plan.sr, tracks=plan.tracks, frozen=[], gate_end=plan.gate_end,
                           solo_active=plan.solo_active, content_end=plan.content_end)
    out = []
    for f in plan.frozen:
        if f.data is None:
            continue
        for edge, a, b in (("start", f.start - f.margin, f.start), ("end", f.end, f.end + f.margin)):
            if b <= a:
                continue
            live = render_range(live_plan, a, b, cache)
            fr = f.data[a - (f.start - f.margin): b - (f.start - f.margin)]
            diff = np.sqrt(np.mean((live.astype(np.float64) - fr) ** 2)) if live.size else 0.0
            db = 20 * math.log10(diff) if diff > 1e-12 else -240.0
            out.append(FrozenBoundary(id=f.id, at=(f.start if edge == "start" else f.end) / plan.sr, mismatch_db=db))
    return out
