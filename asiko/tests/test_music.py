"""Сценарий «Переход A → B»: анализ долей с уверенностью, варианты, конфликты, ручная правка."""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from asiko import testsignals as ts
from asiko.audio.analysis import analyze_beats
from asiko.commands.schema import make_plan, op
from conftest import write

SR = 48000


@pytest.mark.parametrize("bpm,offset", [(120.0, 0.5), (96.0, 0.0), (140.0, 1.1)])
def test_beat_analysis_on_click_tracks(bpm, offset):
    x = ts.click_track(bpm, 24.0, SR, offset=offset)
    g = analyze_beats(x, SR)
    assert abs(g.bpm - bpm) < 0.2
    assert g.confidence > 0.6
    assert abs(g.downbeats[0] - offset) < 0.004   # сильная доля найдена с точностью ~мс


def test_beat_analysis_low_confidence_on_noise():
    g = analyze_beats(ts.noise(20.0, SR, seed=5), SR)
    assert g.confidence < 0.35


def setup_ab(builder, work, bpm_a=120.0, bpm_b=120.0, analyse=True):
    pa = write(work / "A.wav", ts.music_piece(bpm_a, 24, SR, seed=7, final_ring=2.0), SR)
    pb = write(work / "B.wav", ts.music_piece(bpm_b, 16, SR, root="C", minor=False, seed=11), SR)
    sa, ca = builder.add_file(pa, 0.0)
    sb, cb = builder.add_file(pb, 44.0)
    if analyse:
        ops = [op("beats.set", target={"source_id": s.id},
                  params={"grid": asdict(analyze_beats(np.asarray(builder.store.load(s)), SR))}) for s in (sa, sb)]
        builder.apply(ops)
    return sa, sb, ca, cb


def create(builder, ca, cb, at=40.0, **params):
    cons = params.pop("constraints", [{"kind": "keep_tempo"}, {"kind": "keep_pitch"}])
    r = builder.apply([op("music.create", target={"clip_a_id": ca.id, "clip_b_id": cb.id},
                          time={"coord": "timeline", "unit": "s", "at": at}, params=params or None,
                          constraints=cons)])
    return builder.project.music_transition(r.select), r


def test_variants_distinct_and_on_downbeats(builder, work):
    sa, sb, ca, cb = setup_ab(builder, work)
    mt, r = create(builder, ca, cb)
    kinds = [v.kind for v in mt.variants]
    assert kinds == ["short", "long", "phrase"]
    pa_down = [ca.start + d - ca.src_in for d in builder.project.source(sa.id).beats.downbeats]
    for v in mt.variants:
        p = v.params
        assert min(abs(p.switch_time - d) for d in pa_down) < 0.01        # вход — на сильную долю A
        assert abs(p.b_entry_src - builder.project.source(sb.id).beats.downbeats[0]) < 0.01
    lens = [v.params.a_fade_len for v in mt.variants]
    assert len({round(x, 2) for x in lens}) == 3                            # реально различаются
    b = builder.project.clip(cb.id)
    assert abs(b.to_timeline(mt.params.b_entry_src) - mt.params.switch_time) < 1e-9
    assert "не гарантируют" in " ".join(r.summary)


def test_tempo_conflict_shown(builder, work):
    sa, sb, ca, cb = setup_ab(builder, work, bpm_b=100.0)
    mt, r = create(builder, ca, cb)
    long_v = mt.variant("long")
    assert long_v.conflicts and "темп" in long_v.conflicts[0].lower()
    assert builder.project.constraints["keep_tempo"]
    assert any("Конфликт" in w for w in r.warnings)


def test_manual_entry_nudge_and_no_invented_material(builder, work):
    sa, sb, ca, cb = setup_ab(builder, work)
    mt, _ = create(builder, ca, cb)
    before = mt.params.b_entry_src
    builder.apply([op("music.set", target={"transition_id": mt.id}, params={"b_entry_shift_beats": 1})])
    mt = builder.project.music_transition(mt.id)
    beat = 60.0 / builder.project.source(sb.id).beats.bpm
    assert abs(mt.params.b_entry_src - before - beat) < 1e-9 and abs(beat - 0.5) < 0.001
    b = builder.project.clip(cb.id)
    assert abs(b.to_timeline(mt.params.b_entry_src) - mt.params.switch_time) < 1e-9
    builder.apply([op("music.set", target={"transition_id": mt.id}, params={"a_fade_len": 60.0})])
    a = builder.project.clip(ca.id)
    assert a.src_out <= builder.project.source(sa.id).duration + 1e-9   # продолжение не выдумывается
    assert any("доступного материала" in n or "заканчивается раньше" in n for n in builder.project.music_transition(mt.id).notes)


def test_no_grid_falls_back_with_note(builder, work):
    sa, sb, ca, cb = setup_ab(builder, work, analyse=False)
    mt, r = create(builder, ca, cb, at=40.0)
    assert abs(mt.params.switch_time - 40.0) < 1e-9
    assert any("разметк" in n.lower() for n in mt.notes)


def test_locked_b_conflict(builder, work):
    sa, sb, ca, cb = setup_ab(builder, work)
    builder.apply([op("clip.set", target={"clip_id": cb.id}, params={"locked": True})])
    r = builder.engine.apply(make_plan([op("music.create", target={"clip_a_id": ca.id, "clip_b_id": cb.id},
                                           time={"coord": "timeline", "unit": "s", "at": 40.0})],
                                       builder.project.revision))
    assert not r.ok and "зафиксирован" in r.errors[0]


def test_delete_restores_clips(builder, work):
    sa, sb, ca, cb = setup_ab(builder, work)
    b0 = (cb.start, cb.src_in)
    a0 = ca.src_out
    mt, _ = create(builder, ca, cb)
    builder.apply([op("music.delete", target={"transition_id": mt.id})])
    b = builder.project.clip(cb.id)
    assert (b.start, b.src_in) == b0 and builder.project.clip(ca.id).src_out == a0


def test_moving_a_moves_transition_and_b(builder, work):
    sa, sb, ca, cb = setup_ab(builder, work)
    mt, _ = create(builder, ca, cb)
    sw = mt.params.switch_time
    bstart = builder.project.clip(cb.id).start
    builder.apply([op("clip.move", target={"clip_id": ca.id}, time={"coord": "relative", "unit": "s", "delta": 2.0})])
    mt2 = builder.project.music_transition(mt.id)
    assert abs(mt2.params.switch_time - sw - 2.0) < 1e-9
    assert abs(builder.project.clip(cb.id).start - bstart - 2.0) < 1e-9
