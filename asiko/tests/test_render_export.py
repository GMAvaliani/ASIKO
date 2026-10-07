"""5, 6, 10. Длительность экспорта с хвостами, повторяемость рендера, соответствие
предпрослушивания и экспорта, кэш с учётом зависимостей по времени."""
from __future__ import annotations

import time

import numpy as np
import soundfile as sf

from asiko import testsignals as ts
from asiko.audio import dsp
from asiko.audio.export import ExportOptions, export_wav, quantize
from asiko.audio.playback import PlaybackEngine
from asiko.audio.preview import PreviewRenderer, PreviewRequest, loudness_offsets
from asiko.audio.render import CHUNK, ChunkCache, compile_plan, master_chunk_key, render_range
from asiko.commands.schema import op
from conftest import write

SR = 48000


def setup_space(builder, work, dur=20.0):
    p = write(work / "музыка.wav", ts.music_piece(120, int(dur / 2), SR, seed=3), SR)
    src, clip = builder.add_file(p)
    tid = builder.project.tracks[0].id
    x = builder.space(tid, 6.0, 9.0, from_preset="radio_room", to_preset="offscreen")
    return src, clip, x


def test_repeatable_and_region_equals_full(builder, work):
    setup_space(builder, work)
    plan = compile_plan(builder.project, builder.store)
    n = plan.end_with_tails()
    y1 = render_range(plan, 0, n, ChunkCache())
    y2 = render_range(compile_plan(builder.project, builder.store), 0, n, None)
    assert np.array_equal(y1, y2)
    for a, b in ((0.3, 1.7), (5.43, 12.9), (CHUNK / SR - 0.5, CHUNK / SR + 0.5)):
        s0, s1 = int(a * SR), int(b * SR)
        assert np.array_equal(render_range(plan, s0, s1, None), y1[s0:s1])


def test_cache_invalidation_is_time_local(builder, work):
    _s, _c, x = setup_space(builder, work, dur=40.0)
    plan1 = compile_plan(builder.project, builder.store)
    keys1 = [master_chunk_key(plan1, k) for k in range(8)]
    builder.apply([op("space.set_state", target={"transition_id": x.id},
                      params={"state": "to", "values": {"width": 1.4}})])
    plan2 = compile_plan(builder.project, builder.store)
    keys2 = [master_chunk_key(plan2, k) for k in range(8)]
    changed = [k for k in range(8) if keys1[k] != keys2[k]]
    # изменилось конечное состояние → затронуты фрагменты от перехода и дальше, но не до него
    assert 0 not in changed and changed
    cache = ChunkCache()
    render_range(plan2, 0, 8 * CHUNK, cache)
    y_cached = render_range(plan2, 0, 8 * CHUNK, cache)
    assert np.array_equal(y_cached, render_range(plan2, 0, 8 * CHUNK, None))


def test_export_whole_with_and_without_tails(builder, work, tmp_path):
    src, clip, _x = setup_space(builder, work)
    out1 = work / "весь проект.wav"
    rep = export_wav(builder.project, builder.store, ExportOptions(path=str(out1)))
    info = sf.info(str(out1))
    assert info.samplerate == 48000 and info.subtype == "PCM_24" and info.channels == 2
    tail = dsp.ir_length("room", SR) + 2 * dsp.taps_half(SR)
    assert info.frames == rep.frames == int(round(clip.end * SR)) + tail
    assert abs(rep.tail_seconds - tail / SR) < 2 / SR
    out2 = work / "без хвостов.wav"
    rep2 = export_wav(builder.project, builder.store, ExportOptions(path=str(out2), tails=False))
    assert rep2.frames == int(np.ceil(builder.project.content_end() * SR))
    assert rep.frames - rep2.frames == tail
    assert not list(work.glob(".*partial*"))


def test_export_region_tails_and_rates(builder, work):
    setup_space(builder, work)
    out = work / "участок.wav"
    rep = export_wav(builder.project, builder.store, ExportOptions(path=str(out), start=4.0, end=11.0, tails=True,
                                                                   edge_fade_ms=5.0))
    room_tail = dsp.ir_length("room", SR) + 2 * dsp.taps_half(SR)
    assert rep.frames == 7 * SR + room_tail
    data, _ = sf.read(str(out))
    # источники остановлены на конце участка: дальше звучит только затухающий хвост эффекта
    after = data[7 * SR + int(0.02 * SR):]
    assert np.max(np.abs(after)) < 0.5 * np.max(np.abs(data[:7 * SR]))
    assert np.max(np.abs(data[-100:])) < 1e-3
    rep_cut = export_wav(builder.project, builder.store, ExportOptions(path=str(work / "срез.wav"), start=4.0,
                                                                       end=11.0, tails=False))
    assert rep_cut.frames == 7 * SR
    for rate, sub in ((44100, "PCM_16"), (96000, "FLOAT")):
        r = export_wav(builder.project, builder.store, ExportOptions(path=str(work / f"{rate}.wav"), start=4.0,
                                                                     end=11.0, tails=False, sample_rate=rate,
                                                                     subtype=sub))
        i = sf.info(str(work / f"{rate}.wav"))
        assert i.samplerate == rate and i.subtype == sub and i.frames == round(7 * rate) == r.frames


def test_preview_matches_export_bitexact(builder, work):
    setup_space(builder, work)
    engine = PlaybackEngine(SR)
    pr = PreviewRenderer(builder.store, ChunkCache(), engine)
    pr.request(PreviewRequest("result", builder.project))
    t0 = time.time()
    while (pr.busy() or pr.status.get("result", 0) < 1.0) and time.time() - t0 < 60:
        time.sleep(0.02)
    pr.shutdown()
    buf = engine.buffers["result"]
    assert buf.complete()
    out = work / "экспорт.wav"
    rep = export_wav(builder.project, builder.store, ExportOptions(path=str(out), subtype="FLOAT"))
    exported, _ = sf.read(str(out), dtype="float32")
    assert np.array_equal(exported, buf.data[:rep.frames])
    # то, что звучит в аудиопотоке (громкость 1, без выравнивания), — те же сэмплы
    engine.volume = 1.0
    engine.play(start=5.0)
    pulled = np.concatenate([engine.pull(1024) for _ in range(200)], axis=0)
    s0 = 5 * SR
    assert np.array_equal(pulled, buf.data[s0:s0 + pulled.shape[0]])
    # 24 бит: квантование экспортированного файла = квантование буфера прослушивания
    out24 = work / "экспорт24.wav"
    export_wav(builder.project, builder.store, ExportOptions(path=str(out24)))
    ints, _ = sf.read(str(out24), dtype="int32")
    q, _c = quantize(buf.data[:rep.frames].astype(np.float64), "PCM_24", False, 0)
    assert np.array_equal(ints, q)


def test_loudness_match_is_monitoring_only(builder, work):
    setup_space(builder, work)
    engine = PlaybackEngine(SR)
    pr = PreviewRenderer(builder.store, ChunkCache(), engine)
    pr.request(PreviewRequest("result", builder.project))
    pr.request(PreviewRequest("bypass", builder.project, bypass=True))
    t0 = time.time()
    while (pr.busy() or min(pr.status.get("result", 0), pr.status.get("bypass", 0)) < 1.0) and time.time() - t0 < 60:
        time.sleep(0.02)
    pr.shutdown()
    offs = loudness_offsets(engine.buffers, SR, 2.0, 12.0)
    assert offs["result"] == 0.0 and abs(offs["bypass"]) > 0.1
    engine.loudness_match = True
    engine.monitor_gain_db = offs
    engine.volume = 1.0
    engine.set_active("bypass")
    engine.play(start=3.0)
    engine.pull(2048)  # пропускаем перекрёстное затухание
    a = engine.pull(4096)
    s0 = 3 * SR + 2048
    raw = engine.buffers["bypass"].data[s0:s0 + 4096]
    assert np.allclose(a, raw * 10 ** (offs["bypass"] / 20), atol=1e-6)
    # экспорт не знает о мониторинговых поправках
    out = work / "без выравнивания.wav"
    rep = export_wav(builder.project, builder.store, ExportOptions(path=str(out), subtype="FLOAT"))
    exported, _ = sf.read(str(out), dtype="float32")
    assert np.array_equal(exported, engine.buffers["result"].data[:rep.frames])


def test_overload_reported_not_hidden(builder, work):
    p = write(work / "громко.wav", ts.sine(100.0, 3.0, SR, 0.9, 2), SR, "FLOAT")
    _src, clip = builder.add_file(p)
    builder.apply([op("clip.set", target={"clip_id": clip.id}, params={"gain_db": 6.0})])
    out = work / "перегруз.wav"
    rep = export_wav(builder.project, builder.store, ExportOptions(path=str(out)))
    assert rep.clipped_samples > 0 and any("Перегрузка" in w for w in rep.warnings)
    rep_f = export_wav(builder.project, builder.store, ExportOptions(path=str(work / "f.wav"), subtype="FLOAT"))
    assert rep_f.clipped_samples == 0 and rep_f.peak_dbfs > 0
    rep_n = export_wav(builder.project, builder.store, ExportOptions(path=str(work / "n.wav"), normalize_dbfs=-1.0))
    assert rep_n.clipped_samples == 0 and abs(rep_n.peak_dbfs + 1.0) < 0.01


def test_effect_tail_vs_source_tail(builder, work):
    """Хвост добавленного эффекта звучит после конца клипа A; исходный материал не выдумывается."""
    pa = write(work / "A.wav", ts.click_track(120, 12.0, SR), SR)
    pb = write(work / "B.wav", ts.click_track(120, 12.0, SR), SR)
    sa, ca = builder.add_file(pa, start=0.0)
    sb, cb = builder.add_file(pb, start=8.0)
    builder.apply([op("music.create", target={"clip_a_id": ca.id, "clip_b_id": cb.id},
                      time={"coord": "timeline", "unit": "s", "at": 10.0}, params={"exact": True})])
    mt = builder.project.music_transitions[0]
    builder.apply([op("music.set", target={"transition_id": mt.id},
                      params={"a_tail": "effect", "a_fade_start": 10.0, "a_fade_len": 0.03})])
    plan = compile_plan(builder.project, builder.store)
    tp_a = next(t for t in plan.tracks if any(c.clip.id == ca.id for c in t.clips))
    cp = tp_a.clips[0]
    assert cp.tail_room == "hall"
    from asiko.audio.render import render_track

    y = render_track(tp_a, int(10.1 * SR), int(11.0 * SR), SR)
    assert np.max(np.abs(y)) > 1e-3       # реверберация звучит после среза A
    a = builder.project.clip(ca.id)
    assert a.src_out <= sa.duration + 1e-9  # клип A не продлён за конец исходника
