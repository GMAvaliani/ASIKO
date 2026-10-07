"""DSP-блоки ASIKO. Все блоки с памятью — КИХ (конечная память), поэтому выход на любом
интервале — функция входа в окне известной длины (см. render.py).

* Банк линейно-фазовых КИХ-фильтров на сетке 1/12 октавы. Все ядра одной длины,
  задержка (N−1)/2 одинакова и компенсирована: смешивание узлов банка, а также
  прямого и отфильтрованного сигналов фазово-когерентно (без гребенчатых провалов).
* Синтетические импульсные характеристики помещений (детерминированный шум с
  частотно-зависимым затуханием + ранние отражения), декоррелированное стерео.
* Ширина (M/S) и панорама с учётом моно/стерео.
"""
from __future__ import annotations

import functools
import math

import numpy as np
from scipy import signal

HP_MIN, HP_MAX = 20.0, 4100.0
LP_MAX, LP_MIN = 20000.0, 198.0
STEPS_PER_OCT = 12


def taps_half(sr: int) -> int:
    """Половина длины КИХ-фильтра (задержка в сэмплах) ≈ 42.7 мс."""
    return int(round(0.0427 * sr))


def fir_taps(sr: int) -> int:
    return 2 * taps_half(sr) + 1


def hp_nodes() -> int:
    return int(math.ceil(STEPS_PER_OCT * math.log2(HP_MAX / HP_MIN)))


def lp_nodes() -> int:
    return int(math.ceil(STEPS_PER_OCT * math.log2(LP_MAX / LP_MIN)))


def node_freq(kind: str, k: int) -> float:
    if kind == "hp":
        return HP_MIN * 2.0 ** (k / STEPS_PER_OCT)
    return LP_MAX * 2.0 ** (-k / STEPS_PER_OCT)


def positions(kind: str, values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if kind == "hp":
        pos = STEPS_PER_OCT * np.log2(np.maximum(v, HP_MIN) / HP_MIN)
        return np.clip(pos, 0.0, hp_nodes())
    pos = STEPS_PER_OCT * np.log2(LP_MAX / np.minimum(np.maximum(v, 1.0), LP_MAX))
    return np.clip(pos, 0.0, lp_nodes())


def target_magnitude(kind: str, fc: float, order: int, f: np.ndarray) -> np.ndarray:
    """Амплитудная характеристика Баттерворта: 12 дБ/окт (n=2) или 24 дБ/окт (n=4)."""
    n = 2 if order <= 12 else 4
    f = np.maximum(f, 1e-9)
    if kind == "hp":
        return 1.0 / np.sqrt(1.0 + (fc / f) ** (2 * n))
    return 1.0 / np.sqrt(1.0 + (f / fc) ** (2 * n))


@functools.lru_cache(maxsize=1024)
def kernel(kind: str, order: int, sr: int, k: int) -> np.ndarray:
    """Линейно-фазовое ядро узла k (k=0 — тождественное, без фильтрации)."""
    N = fir_taps(sr)
    h = np.zeros(N)
    if k == 0:
        h[N // 2] = 1.0
        h.setflags(write=False)
        return h
    fc = node_freq(kind, k)
    nf = 1 + 2 ** int(math.ceil(math.log2(N)))
    freqs = np.linspace(0.0, sr / 2.0, nf)
    mag = target_magnitude(kind, fc, order, freqs)
    h = signal.firwin2(N, freqs, mag, fs=sr, window=("kaiser", 5.0))
    if kind == "lp":
        h /= np.sum(h)
    else:
        alt = np.sum(h * ((-1.0) ** np.arange(N)))
        if abs(alt) > 1e-6:
            h /= abs(alt) / target_magnitude("hp", fc, order, np.array([sr / 2.0]))[0]
    h.setflags(write=False)
    return h


def _conv_valid(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    return signal.oaconvolve(x, h[:, None], mode="valid", axes=0)


def fir_bank(kind: str, order: int, sr: int, x_ext: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Фильтр с автоматизацией частоты среза по сэмплам.

    x_ext: (n + 2h, C) — вход с контекстом h слева и справа; values: (n,) частота, Гц.
    Возвращает (n, C), выровненный по времени со входом (задержка компенсирована).
    """
    h = taps_half(sr)
    n = values.shape[0]
    assert x_ext.shape[0] == n + 2 * h, (x_ext.shape, n, h)
    pos = positions(kind, values)
    k0 = np.floor(pos).astype(np.int64)
    frac = pos - k0
    hi = frac > 1e-12
    kmax = hp_nodes() if kind == "hp" else lp_nodes()
    k1 = np.minimum(k0 + 1, kmax)
    nodes = set(np.unique(k0).tolist()) | set(np.unique(k1[hi]).tolist())
    if len(nodes) == 1:
        k = nodes.pop()
        if k == 0:
            return x_ext[h:h + n].copy()
        return _conv_valid(x_ext, kernel(kind, order, sr, k))
    if np.all(k0 == k0[0]) and np.all(np.abs(frac - frac[0]) < 1e-12):
        a = kernel(kind, order, sr, int(k0[0]))
        b = kernel(kind, order, sr, int(k1[0]))
        return _conv_valid(x_ext, (1.0 - frac[0]) * a + frac[0] * b)
    out = np.zeros((n, x_ext.shape[1]))
    for k in sorted(nodes):
        w = np.where(k0 == k, 1.0 - frac, 0.0) + np.where((k1 == k) & hi, frac, 0.0)
        idx = np.nonzero(w > 0.0)[0]
        if idx.size == 0:
            continue
        i0, i1 = int(idx[0]), int(idx[-1]) + 1
        if k == 0:
            y = x_ext[h + i0:h + i1]
        else:
            y = _conv_valid(x_ext[i0:i1 + 2 * h], kernel(kind, order, sr, k))
        out[i0:i1] += w[i0:i1, None] * y
    return out


# --------------------------------------------------------------------------- rooms

ROOM_SPECS = {
    "small_room": dict(rt60=0.35, predelay=0.003, er=(0.003, 0.028, 10), bands=(1.1, 1.0, 0.8, 0.5), length=0.6, seed=11),
    "room": dict(rt60=0.7, predelay=0.007, er=(0.005, 0.045, 14), bands=(1.15, 1.0, 0.8, 0.55), length=1.1, seed=23),
    "hall": dict(rt60=1.9, predelay=0.018, er=(0.012, 0.085, 18), bands=(1.2, 1.0, 0.8, 0.6), length=2.8, seed=37),
}

_BAND_EDGES = (250.0, 1000.0, 4000.0)


def ir_length(room: str, sr: int) -> int:
    spec = ROOM_SPECS.get(room)
    return int(spec["length"] * sr) if spec else 0


def max_ir_length(sr: int) -> int:
    return max(ir_length(r, sr) for r in ROOM_SPECS)


@functools.lru_cache(maxsize=16)
def room_ir(room: str, sr: int) -> np.ndarray:
    """Детерминированная стерео ИХ помещения (L, 2), энергия каждого канала = 1."""
    spec = ROOM_SPECS[room]
    L = int(spec["length"] * sr)
    rng = np.random.default_rng(spec["seed"])
    t = np.arange(L) / sr
    noise = rng.standard_normal((L, 2))
    nfft = 1 << int(math.ceil(math.log2(L)))
    X = np.fft.rfft(noise, n=nfft, axis=0)
    f = np.fft.rfftfreq(nfft, 1.0 / sr)
    lf = np.log2(np.maximum(f, 1.0))

    def cross(fc):  # 0 → 1 через октаву вокруг fc
        u = np.clip((lf - (math.log2(fc) - 0.5)), 0.0, 1.0)
        return 0.5 - 0.5 * np.cos(np.pi * u)

    cs = [cross(fc) for fc in _BAND_EDGES]
    masks = [1.0 - cs[0], cs[0] - cs[1], cs[1] - cs[2], cs[2]]
    tail = np.zeros((L, 2))
    for m, ratio in zip(masks, spec["bands"]):
        xb = np.fft.irfft(X * m[:, None], n=nfft, axis=0)[:L]
        rt = spec["rt60"] * ratio
        tail += xb * np.exp(-6.907755 * t / rt)[:, None]
    pd = spec["predelay"]
    onset = np.clip((t - pd) / 0.004, 0.0, 1.0)
    tail *= (0.5 - 0.5 * np.cos(np.pi * onset))[:, None]
    e0, e1, cnt = spec["er"]
    times = np.sort(rng.uniform(e0, e1, cnt))
    gains = rng.uniform(0.4, 1.0, cnt) * np.exp(-6.907755 * times / spec["rt60"])
    signs = rng.choice([-1.0, 1.0], cnt)
    pans = rng.uniform(0.0, np.pi / 2, cnt)
    er = np.zeros((L, 2))
    for tt, g, s, pa in zip(times, gains, signs, pans):
        i = int(tt * sr)
        if i < L:
            er[i, 0] += s * g * math.cos(pa)
            er[i, 1] += s * g * math.sin(pa)
    tail_e = np.sqrt(np.sum(tail ** 2, axis=0)).mean()
    er_e = np.sqrt(np.sum(er ** 2, axis=0)).mean()
    if er_e > 0:
        er *= 0.6 * tail_e / er_e
    ir = tail + er
    nfade = max(1, int(0.08 * L))
    fade = 0.5 + 0.5 * np.cos(np.pi * np.arange(nfade) / nfade)
    ir[-nfade:] *= fade[:, None]
    ir /= np.sqrt(np.sum(ir ** 2, axis=0, keepdims=True))
    ir.setflags(write=False)
    return ir


def convolve_ir(send_ext: np.ndarray, ir: np.ndarray) -> np.ndarray:
    """send_ext: (n + L − 1,) моно-посыл с контекстом слева; возвращает (n, 2)."""
    return signal.oaconvolve(send_ext[:, None], ir, mode="valid", axes=0)


# --------------------------------------------------------------------------- stereo


def apply_width(x: np.ndarray, width: np.ndarray) -> np.ndarray:
    m = 0.5 * (x[:, 0] + x[:, 1])
    s = 0.5 * (x[:, 0] - x[:, 1]) * width
    return np.stack([m + s, m - s], axis=1)


def apply_pan(x: np.ndarray, pan: np.ndarray) -> np.ndarray:
    """Моно → стерео по закону постоянной мощности (−3 дБ в центре).
    Стерео → баланс (в центре без изменения, к краю ослабляется противоположный канал)."""
    theta = (np.clip(pan, -1.0, 1.0) + 1.0) * (np.pi / 4.0)
    c, s = np.cos(theta), np.sin(theta)
    if x.shape[1] == 1:
        return np.stack([x[:, 0] * c, x[:, 0] * s], axis=1)
    gl = np.minimum(1.0, np.sqrt(2.0) * c)
    gr = np.minimum(1.0, np.sqrt(2.0) * s)
    return np.stack([x[:, 0] * gl, x[:, 1] * gr], axis=1)
