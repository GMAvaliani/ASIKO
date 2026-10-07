"""Детерминированные тестовые и демонстрационные сигналы (без чужой музыки).

Тоны, импульсы, шум с фиксированным seed, ритмические последовательности с известным
темпом и простые синтезированные музыкальные фрагменты.
"""
from __future__ import annotations

import math

import numpy as np
from scipy import signal

NOTE = {"C": 0, "C#": 1, "D": 2, "D#": 3, "E": 4, "F": 5, "F#": 6, "G": 7, "G#": 8, "A": 9, "A#": 10, "B": 11}


def midi_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


def sine(freq: float, dur: float, sr: int = 48000, amp: float = 0.5, channels: int = 1, phase: float = 0.0) -> np.ndarray:
    t = np.arange(int(round(dur * sr))) / sr
    x = amp * np.sin(2 * np.pi * freq * t + phase)
    return np.repeat(x[:, None], channels, axis=1).astype(np.float32)


def impulses(times: list[float], dur: float, sr: int = 48000, amp: float = 0.9, channels: int = 1) -> np.ndarray:
    x = np.zeros((int(round(dur * sr)), channels), np.float32)
    for t in times:
        i = int(round(t * sr))
        if 0 <= i < x.shape[0]:
            x[i, :] = amp
    return x


def noise(dur: float, sr: int = 48000, seed: int = 1, amp: float = 0.3, channels: int = 2) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal((int(round(dur * sr)), channels))).astype(np.float32)


def _env(n: int, sr: int, attack: float, decay: float) -> np.ndarray:
    t = np.arange(n) / sr
    a = np.clip(t / max(attack, 1e-4), 0, 1)
    return a * np.exp(-t / max(decay, 1e-4))


def _kick(sr: int, amp: float = 0.9) -> np.ndarray:
    n = int(0.35 * sr)
    t = np.arange(n) / sr
    f = 45 + 90 * np.exp(-t / 0.03)
    ph = 2 * np.pi * np.cumsum(f) / sr
    return amp * np.sin(ph) * _env(n, sr, 0.002, 0.12)


def _snare(sr: int, rng, amp: float = 0.5) -> np.ndarray:
    n = int(0.25 * sr)
    t = np.arange(n) / sr
    nz = rng.standard_normal(n)
    b, a = signal.butter(2, [800 / (sr / 2), 9000 / (sr / 2)], "band")
    nz = signal.lfilter(b, a, nz)
    tone = np.sin(2 * np.pi * 190 * t) * _env(n, sr, 0.001, 0.05)
    return amp * (0.6 * nz * _env(n, sr, 0.001, 0.07) + 0.5 * tone)


def _hat(sr: int, rng, amp: float = 0.18) -> np.ndarray:
    n = int(0.08 * sr)
    nz = rng.standard_normal(n)
    b, a = signal.butter(2, 7000 / (sr / 2), "high")
    return amp * signal.lfilter(b, a, nz) * _env(n, sr, 0.0005, 0.02)


def _add(dst: np.ndarray, src: np.ndarray, at: int, gains=(1.0, 1.0)) -> None:
    if at >= dst.shape[0]:
        return
    n = min(src.shape[0], dst.shape[0] - at)
    dst[at:at + n, 0] += src[:n] * gains[0]
    dst[at:at + n, 1] += src[:n] * gains[1]


def click_track(bpm: float, dur: float, sr: int = 48000, beats_per_bar: int = 4, offset: float = 0.0,
                channels: int = 2) -> np.ndarray:
    """Ритмическая последовательность: акцент (ниже и громче) на сильной доле."""
    x = np.zeros((int(round(dur * sr)), 2))
    period = 60.0 / bpm
    k = 0
    while True:
        t = offset + k * period
        if t >= dur:
            break
        strong = k % beats_per_bar == 0
        n = int(0.05 * sr)
        tt = np.arange(n) / sr
        f = 1000.0 if strong else 2000.0
        amp = 0.8 if strong else 0.4
        click = amp * np.sin(2 * np.pi * f * tt) * np.exp(-tt / 0.01)
        if strong:
            click = click + _kick(sr, 0.6)[:n]
        _add(x, click, int(round(t * sr)))
        k += 1
    if channels == 1:
        return x[:, :1].astype(np.float32)
    return x.astype(np.float32)


def music_piece(bpm: float = 120.0, bars: int = 16, sr: int = 48000, root: str = "A", minor: bool = True,
                seed: int = 7, lead_in: float = 0.0, final_ring: float = 0.0, drums: bool = True,
                brightness: float = 1.0) -> np.ndarray:
    """Простой музыкальный фрагмент: аккорды, бас, ударные, мелодия. Стерео float32."""
    rng = np.random.default_rng(seed)
    period = 60.0 / bpm
    bar = 4 * period
    dur = lead_in + bars * bar + final_ring
    n = int(round(dur * sr))
    x = np.zeros((n, 2))
    r = 57 + NOTE[root] - 9  # A3 = 57
    scale = [0, 2, 3, 5, 7, 8, 10] if minor else [0, 2, 4, 5, 7, 9, 11]
    prog_deg = [0, 5, 3, 4] if minor else [0, 4, 5, 3]
    t_all = np.arange(n) / sr

    def chord_notes(deg):
        return [r + scale[(deg + i) % 7] + 12 * ((deg + i) // 7) for i in (0, 2, 4)]

    # pad
    for b in range(bars + (1 if final_ring > 0 else 0)):
        deg = prog_deg[b % len(prog_deg)] if b < bars else prog_deg[0]
        t0 = lead_in + b * bar
        length = bar if b < bars else final_ring
        i0 = int(round(t0 * sr))
        m = int(round(length * sr)) + int(0.3 * sr)
        tt = np.arange(m) / sr
        if b < bars:
            env = np.clip(tt / 0.05, 0, 1) * np.clip((length + 0.25 - tt) / 0.25, 0, 1)
        else:
            env = np.clip(tt / 0.05, 0, 1) * np.exp(-tt / (final_ring / 4.0 + 1e-3))
        for k, note in enumerate(chord_notes(deg)):
            f = midi_hz(note)
            for ch, det in ((0, -0.15), (1, 0.15)):
                ff = f * 2 ** (det / 12)
                tone = sum((0.5 ** h) * np.sin(2 * np.pi * ff * (h + 1) * tt + k + h) for h in range(5))
                _add(x, 0.07 * brightness ** 0.5 * tone * env, i0, (1.0, 0.0) if ch == 0 else (0.0, 1.0))
    # bass
    for b in range(bars):
        deg = prog_deg[b % len(prog_deg)]
        note = r - 12 + scale[deg]
        for q in range(4):
            t0 = lead_in + b * bar + q * period
            i0 = int(round(t0 * sr))
            m = int(period * 0.9 * sr)
            tt = np.arange(m) / sr
            env = _env(m, sr, 0.005, period * 0.6)
            f = midi_hz(note + (12 if q == 3 else 0))
            tone = np.tanh(1.5 * np.sin(2 * np.pi * f * tt)) * 0.22 * env
            _add(x, tone, i0, (0.8, 0.8))
    # drums
    if drums:
        kick = _kick(sr)
        for b in range(bars):
            for q in range(4):
                t0 = lead_in + b * bar + q * period
                i0 = int(round(t0 * sr))
                if q in (0, 2):
                    _add(x, kick * (1.0 if q == 0 else 0.75), i0, (0.75, 0.75))
                if q in (1, 3):
                    _add(x, _snare(sr, rng), i0, (0.6, 0.6))
                for e in (0, 1):
                    _add(x, _hat(sr, rng, 0.14 if e else 0.2), int(round((t0 + e * period / 2) * sr)), (0.45, 0.6))
    # melody
    pent = [0, 3, 5, 7, 10] if minor else [0, 2, 4, 7, 9]
    for b in range(bars):
        if b % 2 == 0 and b > 0:
            continue
        for e in range(8):
            if rng.random() < 0.45:
                continue
            t0 = lead_in + b * bar + e * period / 2
            note = r + 12 + pent[int(rng.integers(0, len(pent)))]
            m = int(period * 0.45 * sr)
            tt = np.arange(m) / sr
            env = _env(m, sr, 0.004, period * 0.25)
            f = midi_hz(note)
            tone = (np.sin(2 * np.pi * f * tt) + 0.3 * np.sin(4 * np.pi * f * tt)) * 0.12 * env * brightness
            _add(x, tone, int(round(t0 * sr)), (0.55, 0.85))
    pk = np.max(np.abs(x))
    if pk > 0:
        x *= 10 ** (-3 / 20) / pk
    return x.astype(np.float32)
