"""2–4. Точность временных границ, отсутствие NaN/inf/перегрузки, автоматизация и сглаживание."""
from __future__ import annotations

import math

import numpy as np
import pytest

from asiko import testsignals as ts
from asiko.audio import dsp
from asiko.audio.render import compile_plan, render_range
from asiko.commands.schema import op
from asiko.model import automation as A
from asiko.model.presets import BUILTIN_PRESETS
from conftest import write

SR = 48000


def render_all(builder):
    plan = compile_plan(builder.project, builder.store)
    return render_range(plan, 0, plan.end_with_tails(), None)


def test_clip_placement_sample_exact(builder, work):
    imp = ts.impulses([0.5], 2.0, SR, 0.8, 1)
    p = write(work / "импульс.wav", imp, SR, "FLOAT")
    _src, clip = builder.add_file(p, start=3.0)
    builder.apply([op("clip.set", target={"clip_id": clip.id}, params={"fade_in": 0.0, "fade_out": 0.0})])
    y = render_all(builder)
    # моно на стерео-шине без обработки: оба канала, уровень сохранён
    i = int(np.argmax(np.abs(y[:, 0])))
    assert i == round(3.5 * SR)
    assert abs(y[i, 0] - 0.8) < 1e-6 and abs(y[i, 1] - 0.8) < 1e-6


def test_fir_latency_compensated_in_transition(builder, work):
    imp = ts.impulses([1.0, 6.0], 8.0, SR, 0.8, 2)
    p = write(work / "импульсы.wav", imp, SR, "FLOAT")
    _src, clip = builder.add_file(p)
    tid = builder.project.tracks[0].id
    builder.space(tid, 3.0, 4.0, from_preset="phone", to_preset="offscreen")
    y = render_all(builder)
    # импульс в зоне «телефона» (до перехода): пик отфильтрованного сигнала строго на месте
    seg = np.abs(y[:int(2 * SR), :]).sum(axis=1)
    assert int(np.argmax(seg)) == SR
    seg2 = np.abs(y[int(5 * SR):int(7 * SR), :]).sum(axis=1)
    assert int(np.argmax(seg2)) + 5 * SR == 6 * SR


def test_transition_boundaries_exact(builder, work):
    dc = np.full((int(10 * SR), 1), 0.25, np.float32)
    p = write(work / "постоянный.wav", dc, SR, "FLOAT")
    builder.add_file(p)
    tid = builder.project.tracks[0].id
    x = builder.space(tid, 4.0, 6.0, from_preset="offscreen", to_preset="offscreen",
                      to_values={"gain_db": -6.0})
    t = np.arange(int(3.9 * SR), int(6.1 * SR)) / SR
    g = A.space_param(builder.project.transitions_on_track(tid), "gain_db", t)
    s0 = int(4.0 * SR) - int(3.9 * SR)
    s1 = int(6.0 * SR) - int(3.9 * SR)
    assert np.all(g[:s0] == 0.0)            # до начала — начальное состояние
    assert g[s0] == 0.0 and g[s0 + 1] < 0.0  # изменение начинается ровно в 4.000 с
    assert np.all(g[s1:] == -6.0)            # с 6.000 с — конечное состояние
    y = render_all(builder)
    # моно-дорожка → постоянная мощность: 0.25 · cos(π/4) в каждом канале
    a = 0.25 * math.cos(math.pi / 4)
    assert abs(y[int(2 * SR), 0] - a) < 1e-6
    assert abs(y[int(8 * SR), 0] - a * 10 ** (-6 / 20)) < 1e-6


def test_separate_param_timings(builder, work):
    p = write(work / "шум.wav", ts.noise(12.0, SR, seed=2), SR)
    builder.add_file(p)
    tid = builder.project.tracks[0].id
    x = builder.space(tid, 2.0, 8.0, from_preset="radio_room", to_preset="offscreen")
    builder.apply([op("space.set_curve", target={"transition_id": x.id},
                      time={"coord": "transition_end", "unit": "s", "start": -2.0, "end": 0.0},
                      params={"group": "filter"}),
                   op("space.set_curve", target={"transition_id": x.id},
                      time={"coord": "transition", "unit": "s", "start": 0.0, "end": 1.0},
                      params={"group": "width", "shape": "linear"})])
    xs = builder.project.transitions_on_track(tid)
    t = np.array([2.5, 3.5, 5.0, 6.0, 7.0, 8.0])
    hp = A.space_param(xs, "hp_hz", t)
    width = A.space_param(xs, "width", t)
    assert np.allclose(hp[:4], 180.0)            # частоты ещё не раскрываются до 6 с
    assert 20.0 < hp[4] < 180.0 and hp[5] == 20.0
    assert abs(width[0] - 0.5) < 1e-9 and np.allclose(width[1:], 1.0)   # ширина — за первую секунду


def test_no_click_on_step_request(builder, work):
    """Даже «мгновенное» изменение растягивается минимум на 10 мс и не даёт щелчка."""
    p = write(work / "синус.wav", ts.sine(200.0, 4.0, SR, 0.5, 1), SR, "FLOAT")
    builder.add_file(p)
    tid = builder.project.tracks[0].id
    x = builder.space(tid, 2.0, 3.0, from_preset="offscreen", to_preset="offscreen", to_values={"gain_db": -40.0})
    builder.apply([op("space.set_curve", target={"transition_id": x.id},
                      time={"coord": "transition", "unit": "s", "start": 0.0, "end": 0.01},
                      params={"params": ["gain_db"], "shape": "linear"})])
    y = render_all(builder)[:, 0].astype(np.float64)
    d2 = np.abs(np.diff(y, 2))
    natural = 0.5 * (2 * math.pi * 200 / SR) ** 2 * math.cos(math.pi / 4)
    region = d2[int(1.9 * SR):int(2.1 * SR)]
    assert region.max() < natural * 3.0
    t = np.arange(int(2.0 * SR), int(2.02 * SR)) / SR
    g = A.space_param(builder.project.transitions_on_track(tid), "gain_db", t)
    assert g[0] == 0.0 and g[int(0.005 * SR)] < 0 and g[int(0.0099 * SR)] > -40.0


def test_curve_shapes():
    u = np.linspace(0, 1, 11)
    for name in ("linear", "smooth", "ease_in", "ease_out"):
        s = A.shape(name, u)
        assert s[0] == 0.0 and abs(s[-1] - 1.0) < 1e-12
        assert np.all(np.diff(s) >= 0)
    assert A.shape("ease_in", np.array([0.5]))[0] < 0.5 < A.shape("ease_out", np.array([0.5]))[0]


@pytest.mark.parametrize("kind,order,k", [("hp", 12, 40), ("hp", 24, 52), ("lp", 12, 24), ("lp", 24, 40), ("lp", 12, 64)])
def test_filter_kernels(kind, order, k):
    from scipy import signal

    h = dsp.kernel(kind, order, SR, k)
    fc = dsp.node_freq(kind, k)
    oct_ = fc / 2 if kind == "hp" else fc * 2
    _w, H = signal.freqz(h, worN=[fc, oct_], fs=SR)
    db = 20 * np.log10(np.abs(H))
    assert abs(db[0] + 3.01) < 0.35
    slope = 12.3 if order == 12 else 24.1
    assert abs(db[1] + slope) < 1.0
    assert np.allclose(h, h[::-1])  # линейная фаза (симметричное ядро)


def test_fir_bank_sweep_monotonic():
    x = np.random.default_rng(1).standard_normal((SR * 2 + 2 * dsp.taps_half(SR), 1))
    vals = np.geomspace(500.0, 15000.0, SR * 2)
    y = dsp.fir_bank("lp", 12, SR, x, vals)
    assert np.all(np.isfinite(y))
    blocks = y[:, 0].reshape(16, -1)
    energy = (blocks ** 2).mean(axis=1)
    assert np.all(np.diff(energy[1:]) > 0)   # раскрытие частот → энергия растёт


def test_pan_laws():
    mono = np.ones((3, 1))
    out = dsp.apply_pan(mono, np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(out[1], [math.sqrt(0.5)] * 2)     # −3 дБ в центре
    assert np.allclose(out[0], [1, 0], atol=1e-12) and np.allclose(out[2], [0, 1], atol=1e-12)
    st = np.ones((3, 2))
    out = dsp.apply_pan(st, np.array([-1.0, 0.0, 1.0]))
    assert np.allclose(out[1], [1, 1])                   # стерео: баланс, в центре без изменения
    assert np.allclose(out[2], [0, 1], atol=1e-12)
    w = dsp.apply_width(np.array([[1.0, -0.2]]), np.array([0.0]))
    assert np.allclose(w[0, 0], w[0, 1])                 # ширина 0 → моно


def test_room_ir_properties():
    for room in dsp.ROOM_SPECS:
        ir = dsp.room_ir(room, SR)
        assert np.allclose(np.sum(ir ** 2, axis=0), 1.0)
        assert np.all(np.isfinite(ir))
        corr = np.corrcoef(ir[:, 0], ir[:, 1])[0, 1]
        assert abs(corr) < 0.4                           # декоррелированное стерео (моносовместимость)
        assert np.array_equal(ir, dsp.room_ir(room, SR))  # детерминированно


@pytest.mark.parametrize("preset", [k for k in BUILTIN_PRESETS if k != "offscreen"])
def test_presets_no_nan_and_headroom(builder, work, preset):
    p = write(work / "полный уровень.wav", ts.sine(220.0, 6.0, SR, 0.999, 2), SR, "FLOAT")
    builder.add_file(p)
    tid = builder.project.tracks[0].id
    builder.space(tid, 2.0, 4.0, from_preset=preset, to_preset="offscreen")
    y = render_all(builder)
    assert np.all(np.isfinite(y))
    peak = float(np.max(np.abs(y)))
    assert peak < 10 ** (2.0 / 20)   # запас: пресеты не дают больше +2 дБ к исходному пику
    # моносовместимость: сумма в моно не проваливается относительно каналов
    m = 0.5 * (y[:, 0] + y[:, 1])
    assert np.sqrt(np.mean(m ** 2)) > 0.5 * np.sqrt(np.mean(y ** 2))


def test_random_projects_finite(builder, work):
    rng = np.random.default_rng(42)
    p = write(work / "шум2.wav", ts.noise(20.0, SR, seed=9, amp=0.5), SR)
    builder.add_file(p)
    tid = builder.project.tracks[0].id
    keys = list(BUILTIN_PRESETS)
    t = 1.0
    for _ in range(4):
        a, b = rng.choice(keys, 2)
        d = float(rng.uniform(0.2, 3.0))
        builder.space(tid, t, t + d, from_preset=str(a), to_preset=str(b),
                      curves={"hp_hz": {"start": 0.0, "end": float(rng.uniform(0.1, 1.0)), "shape": "ease_in"}})
        t += d + float(rng.uniform(0.5, 2.0))
    y = render_all(builder)
    assert np.all(np.isfinite(y))
    assert float(np.max(np.abs(y))) < 4.0
