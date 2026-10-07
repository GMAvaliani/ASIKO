"""Вычисление автоматизации по сэмплам.

Единственная реализация кривых: её используют и рендер, и проверка защищённых
участков, и отрисовка кривых в интерфейсе. Все функции чистые (зависят только от
описания проекта и момента времени), что делает рендер фрагментами точным.
"""
from __future__ import annotations

import math

import numpy as np

from .presets import PARAM_INFO, REVERB_OFF_DB, SPACE_PARAMS
from .project import Clip, MusicTransition, Project, SpaceTransition

MIN_RAMP_S = 0.010          # минимальная длительность изменения параметра
JUMP_RAMP_S = 0.010         # сглаживание скачка между соседними переходами
BYPASS_CUT_S = 0.005        # анти-щелчок при «жёстком» переключении в режиме сравнения


# --------------------------------------------------------------------------- shapes


def shape(name: str, u: np.ndarray) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    if name == "linear":
        return u
    if name == "ease_in":
        return u * u
    if name == "ease_out":
        return 1.0 - (1.0 - u) * (1.0 - u)
    # smooth: косинусная S-кривая (непрерывная первая производная)
    return 0.5 - 0.5 * np.cos(np.pi * u)


def interp(domain: str, a: float, b: float, s: np.ndarray) -> np.ndarray:
    if domain == "log":
        a = max(a, 1e-6)
        b = max(b, 1e-6)
        return a * np.power(b / a, s)
    return a + (b - a) * s


def db_to_lin(db: np.ndarray, floor_db: float | None = None) -> np.ndarray:
    lin = np.power(10.0, np.asarray(db, dtype=np.float64) / 20.0)
    if floor_db is not None:
        lin = np.where(np.asarray(db) <= floor_db + 1e-6, 0.0, lin)
    return lin


def crossfade_curve(kind: str, u: np.ndarray, fade_in: bool) -> np.ndarray:
    u = np.clip(u, 0.0, 1.0)
    if kind == "linear":
        return u if fade_in else 1.0 - u
    if kind == "smooth":
        s = 0.5 - 0.5 * np.cos(np.pi * u)
        return s if fade_in else 1.0 - s
    # equal_power
    return np.sin(0.5 * np.pi * u) if fade_in else np.cos(0.5 * np.pi * u)


# --------------------------------------------------------------------------- space params


def _ramp_bounds(x: SpaceTransition, param: str) -> tuple[float, float]:
    p0, p1 = x.curve_abs(param)
    if p1 - p0 < MIN_RAMP_S:
        p1 = p0 + MIN_RAMP_S
    return p0, p1


def _state_value(state, param: str, room: str | None) -> float:
    if room is not None:
        return state.effective_reverb(room)
    return float(state.values[param])


def space_param(transitions: list[SpaceTransition], param: str, t: np.ndarray,
                room: str | None = None) -> np.ndarray:
    """Значение параметра (в его единицах) на моменты t (секунды шкалы).

    transitions — переходы одной дорожки, отсортированные по началу.
    room — если задано, вычисляется посыл (дБ) в эту комнату (param='reverb_db').
    """
    t = np.asarray(t, dtype=np.float64)
    info_param = "reverb_db" if room is not None else param
    domain = PARAM_INFO[info_param]["domain"]
    curve_param = info_param
    if not transitions:
        return np.full(t.shape, PARAM_INFO[info_param]["neutral"] if room is None else REVERB_OFF_DB)
    first = transitions[0]
    v = np.full(t.shape, _state_value(first.from_state, info_param, room), dtype=np.float64)
    prev = None
    for x in transitions:
        a = _state_value(x.from_state, info_param, room)
        b = _state_value(x.to_state, info_param, room)
        if prev is not None:
            pb = _state_value(prev.to_state, info_param, room)
            if abs(pb - a) > 1e-12:
                m = (t >= x.start) & (t < x.start + JUMP_RAMP_S)
                if m.any():
                    u = (t[m] - x.start) / JUMP_RAMP_S
                    v[m] = interp(domain, pb, a, shape("smooth", u))
                v[t >= x.start + JUMP_RAMP_S] = a
            else:
                v[t >= x.start] = a
        p0, p1 = _ramp_bounds(x, curve_param)
        c = x.curves[curve_param]
        m = (t >= p0) & (t < p1)
        if m.any():
            u = (t[m] - p0) / (p1 - p0)
            v[m] = interp(domain, a, b, shape(c.shape, u))
        v[t >= p1] = b
        prev = x
    return v


def track_rooms(transitions: list[SpaceTransition]) -> list[str]:
    rooms = []
    for x in transitions:
        for st in (x.from_state, x.to_state):
            if st.room != "none" and st.room not in rooms:
                rooms.append(st.room)
    return rooms


def all_space_params(transitions: list[SpaceTransition], t: np.ndarray) -> dict[str, np.ndarray]:
    out = {p: space_param(transitions, p, t) for p in SPACE_PARAMS if p != "reverb_db"}
    for r in track_rooms(transitions):
        out[f"reverb_db@{r}"] = space_param(transitions, "reverb_db", t, room=r)
    return out


# --------------------------------------------------------------------------- clips


def clip_bounds(clip: Clip, sr: int) -> tuple[int, int, int]:
    """(начало на шкале, начало в исходнике, длина) в сэмплах."""
    start = int(round(clip.start * sr))
    src_in = int(round(clip.src_in * sr))
    src_out = int(round(clip.src_out * sr))
    return start, src_in, max(0, src_out - src_in)


def clip_envelope(clip: Clip, sr: int, n: np.ndarray, music: list[MusicTransition],
                  bypass: bool = False) -> np.ndarray:
    """Усиление клипа (линейное) для абсолютных сэмплов шкалы n. Вне клипа — 0."""
    n = np.asarray(n, dtype=np.int64)
    start, _src_in, length = clip_bounds(clip, sr)
    rel = n - start
    g = np.where((rel >= 0) & (rel < length), 1.0, 0.0)
    g *= 10.0 ** (clip.gain_db / 20.0)
    fi = int(round(clip.fade_in * sr))
    fo = int(round(clip.fade_out * sr))
    if fi > 0:
        m = (rel >= 0) & (rel < fi)
        g[m] *= crossfade_curve(clip.fade_shape, (rel[m] + 0.5) / fi, True)
    if fo > 0:
        m = (rel >= length - fo) & (rel < length)
        g[m] *= crossfade_curve(clip.fade_shape, (length - rel[m] - 0.5) / fo, True)
    t = n / float(sr)
    for mt in music:
        p = mt.params
        if clip.id == mt.clip_a_id:
            if bypass:
                cut = p.switch_time
                u = (t - cut) / BYPASS_CUT_S
                g *= crossfade_curve("linear", u, False)
            else:
                u = (t - p.a_fade_start) / max(p.a_fade_len, 1e-4)
                g *= crossfade_curve(p.curve, u, False)
        if clip.id == mt.clip_b_id:
            if bypass:
                cut = p.switch_time
                u = (t - cut) / BYPASS_CUT_S
                g *= crossfade_curve("linear", u, True)
            else:
                u = (t - p.b_fade_start) / max(p.b_fade_len, 1e-4)
                g *= crossfade_curve(p.curve, u, True)
    return g


OVERLAP_FILTERS = {
    "none": "Без обработки",
    "hp_a": "Убрать низ A (ФВЧ → 400 Гц)",
    "lp_a": "Завалить верх A (ФНЧ → 800 Гц)",
}


def clip_filter(clip: Clip, n: np.ndarray, sr: int, music: list[MusicTransition],
                bypass: bool = False) -> tuple[str, np.ndarray] | None:
    """Автоматизация фильтра на уровне клипа (частотная обработка зоны наложения)."""
    if bypass:
        return None
    t = np.asarray(n, dtype=np.float64) / float(sr)
    for mt in music:
        if clip.id != mt.clip_a_id or mt.params.overlap_filter == "none":
            continue
        p = mt.params
        u = (t - p.a_fade_start) / max(p.a_fade_len, 1e-4)
        s = shape("smooth", u)
        if p.overlap_filter == "hp_a":
            return "hp", interp("log", 20.0, 400.0, s)
        if p.overlap_filter == "lp_a":
            return "lp", interp("log", 20000.0, 800.0, s)
    return None


EFFECT_TAIL_LEAD_S = 0.25
EFFECT_TAIL_DB = -6.0


def clip_effect_tail(clip: Clip, n: np.ndarray, sr: int, music: list[MusicTransition],
                     bypass: bool = False) -> tuple[str, np.ndarray] | None:
    """Посыл в реверберацию «хвоста добавленного эффекта» (не исходная запись)."""
    if bypass:
        return None
    t = np.asarray(n, dtype=np.float64) / float(sr)
    for mt in music:
        if clip.id != mt.clip_a_id or mt.params.a_tail != "effect":
            continue
        p = mt.params
        t0 = p.a_fade_start - EFFECT_TAIL_LEAD_S
        u = (t - t0) / EFFECT_TAIL_LEAD_S
        send = shape("smooth", u) * 10.0 ** (EFFECT_TAIL_DB / 20.0)
        return p.effect_tail_room, send
    return None


def project_music_for(project: Project, clip_id: str) -> list[MusicTransition]:
    return project.music_for_clip(clip_id)


def seconds(x: float) -> str:
    """Человеко-читаемое время: 12.0 → «12 с», 75.25 → «1:15.25»."""
    if x is None or (isinstance(x, float) and math.isinf(x)):
        return "конец"
    if abs(x) < 60:
        s = f"{x:.2f}".rstrip("0").rstrip(".")
        return f"{s} с"
    m = int(x // 60)
    s = x - 60 * m
    return f"{m}:{s:05.2f}"
