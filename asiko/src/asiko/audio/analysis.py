"""Анализ аудио: темп и доли (librosa) с оценкой уверенности, громкость (BS.1770),
пики/true peak, сравнение громкости состояний перехода.

Оценка уверенности — эвристика поверх результата librosa:
* стабильность межударных интервалов (коэффициент вариации),
* выраженность онсетов на долях относительно промежутков между ними,
* для сильных долей — контраст энергии низких частот по фазам такта.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..model.project import BeatGrid
from .io import Cancelled
from .sources import resample

ANALYSIS_SR = 22050
HOP = 512


def _check(cancel):
    if cancel is not None and cancel():
        raise Cancelled()


def _refine_phase(y: np.ndarray, sr: int, grid: np.ndarray, max_shift: float = 0.04) -> float:
    """Уточняет общую фазу равномерной сетки (≈1 мс): сдвиг, при котором суммарный рост
    энергии (окно 2 мс) на долях максимален. Возвращает сдвиг в секундах."""
    hop = max(1, int(0.0005 * sr))
    w = max(2, int(0.002 * sr))
    sq = np.concatenate([[0.0], np.cumsum(y.astype(np.float64) ** 2)])
    idx = np.arange(w, y.shape[0], hop)
    if idx.size < 4:
        return 0.0
    e = (sq[idx] - sq[idx - w]) / w
    le = np.log10(e + 1e-10)
    rise = np.maximum(0.0, le[1:] - le[:-1])
    tr = (idx[:-1] + idx[1:]) / 2.0 / sr - 0.001
    deltas = np.arange(-max_shift, max_shift + 1e-9, 0.0005)
    scores = [float(np.sum(np.interp(grid + d, tr, rise, left=0.0, right=0.0))) for d in deltas]
    best = int(np.argmax(scores))
    if scores[best] <= 0:
        return 0.0
    return float(deltas[best])


def _regular_grid(times: np.ndarray, start: float, duration: float):
    """Подгонка равномерной сетки (постоянный темп). Возвращает (сетка, период) или (None, None)."""
    k = np.arange(times.shape[0], dtype=np.float64)
    A = np.vstack([k, np.ones_like(k)]).T
    (period, t0), *_ = np.linalg.lstsq(A, times, rcond=None)
    resid = times - (t0 + period * k)
    if period <= 0 or float(np.std(resid)) > 0.02:
        return None, None
    kmin = int(np.ceil((start - 0.03 - t0) / period))
    kmax = int(np.floor((duration - t0) / period))
    grid = t0 + period * np.arange(kmin, kmax + 1)
    return grid[grid >= 0.0], float(period)


def analyze_beats(audio: np.ndarray, sr: int, beats_per_bar: int = 4, cancel=None, progress=None) -> BeatGrid:
    import librosa  # отложенный импорт: тяжёлая библиотека (numba)

    def prog(f, msg):
        if progress:
            progress(f, msg)

    prog(0.02, "Подготовка сигнала…")
    mono = np.asarray(audio, dtype=np.float32).mean(axis=1, keepdims=True)
    y = resample(mono, sr, ANALYSIS_SR, cancel=cancel)[:, 0]
    _check(cancel)
    duration = y.shape[0] / ANALYSIS_SR
    if duration < 3.0:
        return BeatGrid(bpm=120.0, beats=[], downbeats=[], beats_per_bar=beats_per_bar, confidence=0.0,
                        notes=["Слишком короткий фрагмент для оценки темпа (нужно ≥ 3 с)."])
    prog(0.15, "Огибающая онсетов…")
    # По сегментам (кратным шагу анализа), чтобы отмена срабатывала и на длинных файлах.
    seg = HOP * 2560
    pad = HOP * 16
    nframes_total = 1 + y.shape[0] // HOP
    env = np.zeros(nframes_total)
    low = np.zeros(nframes_total)
    for i in range(0, y.shape[0], seg):
        _check(cancel)
        a = max(0, i - pad)
        b = min(y.shape[0], i + seg + pad)
        part = y[a:b]
        e = librosa.onset.onset_strength(y=part, sr=ANALYSIS_SR, hop_length=HOP)
        lo = librosa.onset.onset_strength(y=part, sr=ANALYSIS_SR, hop_length=HOP, fmax=220.0, n_mels=24)
        g0 = i // HOP
        g1 = min(nframes_total, (i + seg) // HOP)
        off = (i - a) // HOP
        m = max(0, min(g1 - g0, e.shape[0] - off))
        env[g0:g0 + m] = e[off:off + m]
        low[g0:g0 + m] = lo[off:off + m]
        prog(0.15 + 0.45 * min(1.0, (i + seg) / y.shape[0]), "Огибающая онсетов…")
    _check(cancel)
    prog(0.65, "Оценка темпа и долей…")
    tempo, beat_frames = librosa.beat.beat_track(onset_envelope=env, sr=ANALYSIS_SR, hop_length=HOP)
    _check(cancel)
    beat_frames = np.asarray(beat_frames, dtype=np.int64)
    beat_frames = beat_frames[beat_frames < env.shape[0]]
    notes = ["Темп и доли: librosa.beat.beat_track; сильные доли — по энергии низких частот (эвристика)."]
    if beat_frames.size < 2 * beats_per_bar:
        return BeatGrid(bpm=float(np.atleast_1d(tempo)[0]) or 120.0, beats=[], downbeats=[],
                        beats_per_bar=beats_per_bar, confidence=0.0,
                        notes=notes + ["Доли не найдены: материал без выраженного ритма."])
    times = librosa.frames_to_time(beat_frames, sr=ANALYSIS_SR, hop_length=HOP)
    ibi = np.diff(times)
    med = float(np.median(ibi))
    cv = float(np.std(ibi) / max(np.mean(ibi), 1e-9))
    c_stab = float(np.clip(1.0 - cv / 0.08, 0.0, 1.0))
    mids = ((beat_frames[:-1] + beat_frames[1:]) // 2).astype(np.int64)
    on = float(np.mean(env[beat_frames]))
    off = float(np.mean(env[mids])) if mids.size else 0.0
    ratio = on / max(off, 1e-9)
    c_align = float(np.clip((ratio - 1.0) / 1.5, 0.0, 1.0))
    tempo_conf = math.sqrt(c_stab * c_align)
    prog(0.85, "Сильные доли…")
    scores = []
    for ph in range(beats_per_bar):
        fr = beat_frames[ph::beats_per_bar]
        s = float(np.mean(low[fr])) + 0.25 * float(np.mean(env[fr]))
        scores.append(s)
    order = np.argsort(scores)[::-1]
    best = int(order[0])
    second = scores[int(order[1])] if len(order) > 1 else 0.0
    d_conf = float(np.clip(3.0 * (scores[best] - second) / max(scores[best], 1e-9), 0.0, 1.0))
    d0 = float(times[best])
    thr = 0.02 * float(np.max(np.abs(y))) if y.size else 0.0
    nz = np.nonzero(np.abs(y) > thr)[0]
    first_sound = nz[0] / ANALYSIS_SR if nz.size else 0.0
    grid, period = _regular_grid(times, first_sound, duration)
    if grid is not None:
        bpm = 60.0 / period
        shift = _refine_phase(y, ANALYSIS_SR, grid)
        grid = grid + shift
        d0 += shift
        times = grid[grid >= 0.0]
        grid = times
        idx0 = int(np.argmin(np.abs(grid - d0)))
        downbeats = grid[idx0 % beats_per_bar::beats_per_bar]
        notes.append("Темп постоянный: сетка выровнена по всем долям (точность ≈1 мс).")
    else:
        bpm = 60.0 / med
        downbeats = times[best::beats_per_bar]
        notes.append("Темп меняется: используются доли, найденные по отдельности.")
    conf = tempo_conf * (0.6 + 0.4 * d_conf)
    if tempo_conf < 0.35:
        notes.append("Низкая уверенность: неровный ритм или нет чётких ударов. Проверьте сетку вручную.")
    if d_conf < 0.3:
        notes.append("Сильные доли определены неуверенно — задайте первую сильную долю вручную.")
    prog(1.0, "Готово")
    return BeatGrid(bpm=round(bpm, 3), beats=[round(float(t), 5) for t in times],
                    downbeats=[round(float(t), 5) for t in downbeats], beats_per_bar=beats_per_bar,
                    confidence=round(conf, 3), tempo_confidence=round(tempo_conf, 3),
                    downbeat_confidence=round(d_conf, 3), method="auto", notes=notes)


# --------------------------------------------------------------------------- loudness & peaks


def integrated_lufs(audio: np.ndarray, sr: int) -> float:
    import pyloudnorm

    x = np.asarray(audio, dtype=np.float64)
    if x.shape[0] < int(0.4 * sr) or not np.any(x):
        return float("-inf")
    meter = pyloudnorm.Meter(sr)
    try:
        v = float(meter.integrated_loudness(x))
    except ValueError:
        return float("-inf")
    return v


def peak_dbfs(audio: np.ndarray) -> float:
    p = float(np.max(np.abs(audio))) if audio.size else 0.0
    return 20 * math.log10(p) if p > 0 else float("-inf")


def true_peak_dbfs(audio: np.ndarray, sr: int) -> float:
    """Оценка true peak: 4× передискретизация (ITU-R BS.1770, упрощённо)."""
    from scipy import signal

    if audio.size == 0:
        return float("-inf")
    x = np.asarray(audio, dtype=np.float64)
    if x.shape[0] > sr * 600:
        x = x[: sr * 600]
    up = signal.resample_poly(x, 4, 1, axis=0)
    p = float(np.max(np.abs(up)))
    return 20 * math.log10(p) if p > 0 else float("-inf")


def peak_envelope(audio: np.ndarray, sr: int, step_s: float = 0.05) -> np.ndarray:
    """Пики мастера с шагом step_s (для отметок перегрузки на шкале)."""
    step = max(1, int(step_s * sr))
    n = audio.shape[0] // step
    if n == 0:
        return np.zeros(0, np.float32)
    a = np.abs(audio[: n * step]).reshape(n, step, -1).max(axis=(1, 2))
    return a.astype(np.float32)


@dataclass
class StateLoudness:
    from_lufs: float
    to_lufs: float

    @property
    def diff_db(self) -> float:
        if math.isinf(self.from_lufs) or math.isinf(self.to_lufs):
            return 0.0
        return self.to_lufs - self.from_lufs


def state_loudness(project, store, transition_id: str, window: float = 4.0, cancel=None) -> StateLoudness:
    """Громкость материала в начальном и конечном состоянии перехода (одна дорожка)."""
    from .render import compile_plan, render_range

    x = project.space_transition(transition_id)
    if x is None:
        raise ValueError("Переход не найден")
    res = []
    for which in ("from", "to"):
        p = project.clone()
        y = p.space_transition(transition_id)
        st = y.from_state if which == "from" else y.to_state
        y.from_state = st
        y.to_state = st
        p.space_transitions = [y]
        for t in p.tracks:
            t.solo = t.id == y.track_id
            t.mute = False
        a = max(0.0, x.start - window)
        b = x.end + window
        plan = compile_plan(p, store, apply_frozen=False)
        sr = p.sample_rate
        audio = render_range(plan, int(a * sr), int(b * sr), None, cancel=cancel)
        res.append(integrated_lufs(audio, sr))
    return StateLoudness(res[0], res[1])
