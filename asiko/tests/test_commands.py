"""7–9, 11. Сохранение без потери настроек, undo/redo, защищённые параметры и участки,
неприменение невалидных и устаревших команд."""
from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from asiko import testsignals as ts
from asiko.audio.export import ExportOptions, export_wav
from asiko.audio.freeze import FrozenLoader, freeze_params
from asiko.commands.schema import PLAN_SCHEMA, make_plan, op, plan_json_schema, validate_plan_schema
from asiko.model.project import Project
from asiko.storage.project_io import load_project, save_project
from conftest import write

SR = 48000


@pytest.fixture
def proj(builder, work):
    p = write(work / "музыка.wav", ts.music_piece(120, 12, SR, seed=4), SR)
    src, clip = builder.add_file(p)
    tid = builder.project.tracks[0].id
    x = builder.space(tid, 6.0, 10.0, from_preset="radio_room", to_preset="offscreen")
    return builder, tid, x, clip


def test_undo_redo_exact(proj):
    b, tid, x, clip = proj
    eng = b.engine
    snapshots = [eng.project.to_dict()]
    ops = [
        [op("space.set_timing", target={"transition_id": x.id}, time={"coord": "timeline", "unit": "s", "end": 12.0})],
        [op("space.set_curve", target={"transition_id": x.id},
            time={"coord": "transition_end", "unit": "s", "start": -2, "end": 0}, params={"group": "filter"})],
        [op("track.set", target={"track_id": tid}, params={"delta_db": -3.0})],
        [op("clip.move", target={"clip_id": clip.id}, time={"coord": "relative", "unit": "s", "delta": 1.0})],
        [op("space.set_state", target={"transition_id": x.id}, params={"state": "from", "values": {"pan": -0.5}})],
    ]
    for o in ops:
        b.apply(o)
        snapshots.append(eng.project.to_dict())

    def strip(d):
        d = dict(d)
        d.pop("revision")
        return json.dumps(d, sort_keys=True)

    for i in range(len(ops), 0, -1):
        assert eng.undo().ok
        assert strip(eng.project.to_dict()) == strip(snapshots[i - 1])
    assert len(eng.redo_stack) == len(ops)
    for i in range(1, len(ops) + 1):
        assert eng.redo().ok
        assert strip(eng.project.to_dict()) == strip(snapshots[i])
    assert not eng.redo().ok
    while eng.undo_stack:
        assert eng.undo().ok
    assert not eng.undo().ok and eng.project.tracks == []


def test_clip_move_carries_linked_transition(proj):
    b, tid, x, clip = proj
    b.apply([op("clip.move", target={"clip_id": clip.id}, time={"coord": "relative", "unit": "s", "delta": 2.5})])
    y = b.project.space_transition(x.id)
    assert (y.start, y.end) == (8.5, 12.5)


def test_save_load_roundtrip(proj, work):
    b, tid, x, clip = proj
    b.apply([op("transition.lock_params", target={"transition_id": x.id}, params={"params": ["pan"], "locked": True}),
             op("protect.add", params={"start": 15.0, "end": None, "track_id": tid}),
             op("constraint.set", params={"keep_tempo": True, "keep_pitch": True})])
    path = work / "проект с пробелами.asiko"
    save_project(b.project, path)
    assert not list(work.glob(".*tmp"))
    p2 = load_project(path)
    a = b.project.to_dict()
    c = p2.to_dict()
    for d in (a, c):
        for s in d["sources"]:
            s.pop("rel_path", None)
    assert a == c
    raw = json.loads(path.read_text("utf-8"))
    assert raw["format"] == "asiko.project" and raw["format_version"] == 1
    assert raw["sources"][0]["sha256"] and raw["sources"][0]["rel_path"]


def test_corrupted_project_error(work):
    from asiko.model.migrate import ProjectFormatError

    p = work / "битый.asiko"
    p.write_text("{ не json", "utf-8")
    with pytest.raises(ProjectFormatError):
        load_project(p)
    p.write_text(json.dumps({"format": "asiko.project", "format_version": 99}), "utf-8")
    with pytest.raises(ProjectFormatError) as e:
        load_project(p)
    assert "новой версией" in str(e.value)


# ------------------------------------------------------------------ invalid & stale


@pytest.mark.parametrize("plan,needle", [
    ("not a dict", "JSON-объектом"),
    ({"schema": "other/1", "base_revision": 0, "kind": "edit", "operations": [{"op": "track.add"}]}, "schema"),
    ({"schema": PLAN_SCHEMA, "base_revision": -1, "kind": "edit", "operations": [{"op": "track.add"}]}, "base_revision"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": [{"op": "shell.exec", "params": {"cmd": "rm -rf /"}}]}, "неизвестная операция"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": [{"op": "track.set", "target": {"track_id": "t"}, "params": {"gain_db": "громко"}}]}, "ожидалось число"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": [{"op": "track.set", "target": {"track_id": "t"}, "params": {"gain_db": 99}}]}, "максимума"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": [{"op": "track.set", "target": {"track_id": "t"}, "params": {"gain_db": float("nan")}}]}, "недопустимое"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": [{"op": "track.set", "target": {"track_id": "t"}, "params": {"volume": 1}}]}, "неизвестное поле"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": [{"op": "space.create", "target": {"track_id": "t"}}]}, "time"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": []}, "не содержит операций"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "origin": "llm", "kind": "edit", "operations": [{"op": "video.set", "params": {"path": "/etc/passwd"}}]}, "недоступна внешней модели"),
    ({"schema": PLAN_SCHEMA, "base_revision": 0, "kind": "edit", "operations": [{"op": "space.set_state", "target": {"transition_id": "x"}, "params": {"state": "from", "values": {"lp_hz": 50}}}]}, "вне диапазона"),
])
def test_invalid_plans_rejected(proj, plan, needle):
    b = proj[0]
    before = json.dumps(b.project.to_dict(), sort_keys=True)
    r = b.engine.apply(plan)
    assert not r.ok and not r.applied
    assert any(needle in e for e in r.errors), r.errors
    assert json.dumps(b.project.to_dict(), sort_keys=True) == before


def test_semantic_errors_rejected_atomically(proj):
    b, tid, x, clip = proj
    before = json.dumps(b.project.to_dict(), sort_keys=True)
    plan = make_plan([op("track.set", target={"track_id": tid}, params={"delta_db": -3}),
                      op("space.set_timing", target={"transition_id": x.id},
                         time={"coord": "timeline", "unit": "s", "start": 9.0, "end": 8.0})], b.project.revision)
    r = b.engine.apply(plan)
    assert not r.ok
    assert json.dumps(b.project.to_dict(), sort_keys=True) == before   # первая операция тоже не применена
    r = b.engine.apply(make_plan([op("space.delete", target={"transition_id": "spc_nothere"})], b.project.revision))
    assert not r.ok and "не найден" in r.errors[0]
    r = b.engine.apply(make_plan([op("space.create", target={"track_id": tid},
                                     time={"coord": "timeline", "unit": "s", "start": 8.0, "end": 11.0})],
                                 b.project.revision))
    assert not r.ok and "пересекается" in r.errors[0]


def test_stale_plan_revalidated_not_auto_applied(proj):
    b, tid, x, clip = proj
    old_rev = b.project.revision
    stale = make_plan([op("space.set_timing", target={"transition_id": x.id},
                          time={"coord": "relative", "unit": "s", "end_delta": 1.0})], old_rev)
    b.apply([op("track.set", target={"track_id": tid}, params={"delta_db": -1.0})])
    r = b.engine.apply(stale)
    assert r.ok and r.stale and not r.applied            # проверен заново, но не применён без подтверждения
    assert b.project.space_transition(x.id).end == 10.0
    r = b.engine.apply(stale, confirm_stale=True)
    assert r.applied and b.project.space_transition(x.id).end == 11.0
    # устаревший план, ссылающийся на удалённый объект, отклоняется
    stale2 = make_plan([op("space.set_timing", target={"transition_id": x.id},
                           time={"coord": "relative", "unit": "s", "end_delta": 1.0})], b.project.revision)
    b.apply([op("space.delete", target={"transition_id": x.id})])
    r = b.engine.apply(stale2, confirm_stale=True)
    assert not r.ok and not r.applied


def test_preconditions(proj):
    b, tid, x, clip = proj
    r = b.engine.apply(make_plan([op("track.set", target={"track_id": tid}, params={"delta_db": -1},
                                     preconditions=[{"kind": "revision", "value": 999}])], b.project.revision))
    assert not r.ok and "Предусловие" in r.errors[0]
    r = b.engine.apply(make_plan([op("track.set", target={"track_id": tid}, params={"delta_db": -1},
                                     preconditions=[{"kind": "duration_at_least", "id": x.id, "seconds": 3.0},
                                                    {"kind": "exists", "id": clip.id}])], b.project.revision))
    assert r.ok and r.applied


def test_json_schema_export_is_valid_json():
    s = plan_json_schema()
    txt = json.dumps(s, ensure_ascii=False)
    assert "space.create" in txt and "freeze.add" not in txt   # операции только для интерфейса не видны модели


# ------------------------------------------------------------------ protections


def test_param_lock(proj):
    b, tid, x, clip = proj
    b.apply([op("transition.lock_params", target={"transition_id": x.id}, params={"params": ["pan"], "locked": True})])
    r = b.engine.apply(make_plan([op("space.set_state", target={"transition_id": x.id},
                                     params={"state": "from", "values": {"pan": -0.9}})], b.project.revision))
    assert not r.ok and "зафиксирован" in r.errors[0]
    r = b.engine.apply(make_plan([op("space.set_curve", target={"transition_id": x.id},
                                     time={"coord": "transition_fraction", "unit": "fraction", "start": 0.5, "end": 1},
                                     params={"group": "pan"})], b.project.revision))
    assert not r.ok
    # продление перехода: зафиксированная панорама сохраняет абсолютное время
    b.apply([op("space.set_timing", target={"transition_id": x.id}, time={"coord": "timeline", "unit": "s", "end": 12.0})])
    y = b.project.space_transition(x.id)
    assert y.curve_abs("pan") == pytest.approx((6.0, 10.0))
    assert y.curve_abs("hp_hz") == pytest.approx((6.0, 12.0))


def test_clip_lock_and_approval(proj):
    b, tid, x, clip = proj
    b.apply([op("clip.set", target={"clip_id": clip.id}, params={"locked": True})])
    r = b.engine.apply(make_plan([op("clip.move", target={"clip_id": clip.id},
                                     time={"coord": "relative", "unit": "s", "delta": 1})], b.project.revision))
    assert not r.ok
    b.apply([op("clip.set", target={"clip_id": clip.id}, params={"locked": False})])
    b.apply([op("transition.approve", target={"transition_id": x.id}, params={"approved": True})])
    for bad in ([op("space.delete", target={"transition_id": x.id})],
                [op("space.set_timing", target={"transition_id": x.id}, time={"coord": "timeline", "unit": "s", "end": 11})]):
        r = b.engine.apply(make_plan(bad, b.project.revision))
        assert not r.ok and "утвержд" in r.errors[0].lower()
    b.apply([op("transition.approve", target={"transition_id": x.id}, params={"approved": False}),
             op("space.set_timing", target={"transition_id": x.id}, time={"coord": "timeline", "unit": "s", "end": 11})])


def test_range_protection_settings(proj):
    b, tid, x, clip = proj
    b.apply([op("protect.add", params={"start": 10.0, "end": None, "track_id": tid})])
    # продление в защищённый участок — запрещено
    r = b.engine.apply(make_plan([op("space.set_timing", target={"transition_id": x.id},
                                     time={"coord": "timeline", "unit": "s", "end": 12.0})], b.project.revision))
    assert not r.ok and "Защищённый участок" in r.errors[0]
    # уровень всей дорожки тоже затрагивает участок
    r = b.engine.apply(make_plan([op("track.set", target={"track_id": tid}, params={"delta_db": -3})], b.project.revision))
    assert not r.ok
    # правка внутри перехода (до 10 с) разрешена; предупреждение о хвосте реверберации
    r = b.engine.apply(make_plan([op("space.set_curve", target={"transition_id": x.id},
                                     time={"coord": "transition_end", "unit": "s", "start": -1, "end": 0},
                                     params={"group": "filter"})], b.project.revision))
    assert r.ok and r.applied
    assert any("хвост" in w for w in r.warnings)


def test_protect_after_constraint_from_text_example(builder, work):
    p = write(work / "м.wav", ts.music_piece(120, 12, SR, seed=6), SR)
    builder.add_file(p)
    tid = builder.project.tracks[0].id
    x = builder.space(tid, 12.0, 16.0, constraints=[{"kind": "protect_after", "time": 16.0}])
    assert builder.project.protections[0].start == 16.0
    r = builder.engine.apply(make_plan([op("space.create", target={"track_id": tid},
                                           time={"coord": "timeline", "unit": "s", "start": 2, "end": 18},
                                           constraints=[{"kind": "protect_after", "time": 16}])],
                                       builder.project.revision))
    assert not r.ok


def test_freeze_bit_exact(proj, work):
    b, tid, x, clip = proj
    project_path = work / "заморозка.asiko"
    save_project(b.project, project_path)
    data_dir = work / "заморозка.asiko_data"
    loader = FrozenLoader()
    loader.data_dir = data_dir
    params = freeze_params(b.project, b.store, data_dir, 12.0, 16.0, frozen_audio=loader)
    b.apply([op("freeze.add", params=params)])
    e1 = work / "до.wav"
    export_wav(b.project, b.store, ExportOptions(path=str(e1)), frozen_audio=loader)
    # меняем настройки, которые влияют на звучание замороженного участка (хвосты, уровень)
    r = b.engine.apply(make_plan([op("space.set_state", target={"transition_id": x.id},
                                     params={"state": "to", "values": {"gain_db": -6.0}})], b.project.revision))
    assert r.ok and any("зафиксирован в аудио" in w for w in r.warnings)
    e2 = work / "после.wav"
    export_wav(b.project, b.store, ExportOptions(path=str(e2)), frozen_audio=loader)
    a, _ = sf.read(str(e1), dtype="int32")
    c, _ = sf.read(str(e2), dtype="int32")
    s0, s1 = 12 * SR, 16 * SR
    assert np.array_equal(a[s0:s1], c[s0:s1])          # участок побитово неизменен
    assert not np.array_equal(a[s1 + SR:s1 + 2 * SR], c[s1 + SR:s1 + 2 * SR])  # вне участка изменения слышны
    # повреждённый файл фиксации не применяется молча
    f = data_dir / params["file"]
    f.write_bytes(f.read_bytes()[:-100] + b"\0" * 100)
    loader2 = FrozenLoader()
    loader2.data_dir = data_dir
    assert loader2(b.project.frozen[0]) is None and loader2.errors
