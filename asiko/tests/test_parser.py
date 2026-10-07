"""Локальный обработчик: обязательные команды и поведение при неоднозначности."""
from __future__ import annotations

import pytest

from asiko import testsignals as ts
from asiko.nlp.local_parser import ParseContext, Selection, normalize, parse
from conftest import write

SR = 48000


@pytest.fixture
def setup(builder, work):
    p = write(work / "музыка.wav", ts.music_piece(120, 16, SR, seed=4), SR)
    builder.add_file(p)
    sel = Selection(track_id=builder.project.tracks[0].id)
    return builder, sel


def run(builder, sel, text, **kw):
    plan = parse(text, ParseContext(builder.project, sel))
    r = builder.engine.apply(plan, **kw)
    if r.select and builder.project.any_transition(r.select):
        sel.transition_id = r.select
    return plan, r


def test_normalize_numbers():
    assert normalize("на сороковой секунде") == "на 40 секунде"
    assert normalize("с двенадцатой по шестнадцатую") == "с 12 по 16"
    assert normalize("на две секунды") == "на 2 секунды"
    assert normalize("в 0:40") == "в 40"


def test_required_commands(setup):
    b, sel = setup
    plan, r = run(b, sel, "Сделай переход с 12 до 16 секунды")
    assert plan["origin"] == "local" and r.applied
    x = b.project.space_transitions[0]
    assert (x.start, x.end) == (12, 16)
    plan, r = run(b, sel, "Продли выбранный переход на 2 секунды")
    assert r.applied and b.project.space_transitions[0].end == 18
    plan, r = run(b, sel, "Сдвинь выбранный переход на 1 секунду вправо")
    x = b.project.space_transitions[0]
    assert r.applied and (x.start, x.end) == (13, 19)
    plan, r = run(b, sel, "Уменьши громкость выбранной дорожки на 3 дБ")
    assert r.applied and b.project.tracks[0].gain_db == -3.0
    plan, r = run(b, sel, "Не меняй темп и высоту тона")
    assert r.applied and b.project.constraints == {"keep_tempo": True, "keep_pitch": True}
    plan, r = run(b, sel, "Отмени последнее изменение")
    assert plan["kind"] == "undo" and r.ok
    assert b.project.constraints["keep_tempo"] is False


def test_spec_example_space(setup):
    b, sel = setup
    plan, r = run(b, sel, "До 12-й секунды музыка звучит из радио справа в комнате. С 12-й по 16-ю переходит в "
                          "закадровую. Без резкого скачка громкости. После 16-й не менять.")
    assert r.applied, r.errors
    x = b.project.space_transitions[0]
    assert x.from_state.preset == "radio_room" and x.from_state.values["pan"] > 0 and x.from_state.room == "room"
    assert x.to_state.preset == "offscreen" and (x.start, x.end) == (12, 16)
    assert x.curves["gain_db"].shape == "smooth" and (x.curves["gain_db"].start, x.curves["gain_db"].end) == (0, 1)
    assert b.project.protections[0].start == 16 and b.project.protections[0].end is None


def test_curve_edit_minimal_change(setup):
    b, sel = setup
    run(b, sel, "Сделай переход с 12 до 18 секунды")
    before = b.project.clone()
    plan, r = run(b, sel, "В выбранном переходе эффект радио убирай только в самом конце")
    assert r.applied
    x = b.project.space_transitions[0]
    assert x.curve_abs("lp_hz") == pytest.approx((17.0, 18.0))
    assert x.curve_abs("color_mix") == pytest.approx((17.0, 18.0))
    y = before.space_transitions[0]
    for p in ("gain_db", "pan", "width", "reverb_db", "direct_db"):
        assert x.curves[p] == y.curves[p]
    assert (x.start, x.end) == (y.start, y.end)
    assert [c.start for c in b.project.all_clips()] == [c.start for c in before.all_clips()]
    assert "остальные параметры сохранены" in " ".join(r.summary)
    plan, r = run(b, sel, "Перенеси раскрытие частот в конец")
    assert b.project.space_transitions[0].curve_abs("hp_hz") == pytest.approx((16.0, 18.0))
    assert "последние 2 с" in " ".join(r.summary)


def test_question_when_ambiguous(builder, work):
    for i in range(2):
        p = write(work / f"т{i}.wav", ts.noise(20.0, SR, seed=i), SR)
        builder.add_file(p)
    plan = parse("Уменьши громкость дорожки на 3 дБ", ParseContext(builder.project, Selection()))
    assert plan["kind"] == "question" and len(plan["question"]["options"]) == 2
    opt = plan["question"]["options"][1]["plan"]
    r = builder.engine.apply(opt)
    assert r.applied and builder.project.tracks[1].gain_db == -3.0


def test_artistic_proposal_not_applied_silently(setup):
    b, sel = setup
    run(b, sel, "Сделай переход с 12 до 16 секунды")
    rev = b.project.revision
    plan, r = run(b, sel, "Сделай драматичнее")
    assert plan["kind"] == "proposal" and plan["artistic"] is True
    assert r.ok and not r.applied and b.project.revision == rev
    r = b.engine.apply(plan, accept_proposal=True)
    assert r.applied
    assert b.engine.undo_stack[-1].description.startswith("[Художественное предложение]")


def test_stem_request_explained(setup):
    b, sel = setup
    plan = parse("Убери ударные отдельно", ParseContext(b.project, sel))
    assert plan["kind"] == "info" and "отдельная дорожка" in plan["message"]


def test_unknown_text(setup):
    b, sel = setup
    plan = parse("напиши стихотворение про осень", ParseContext(b.project, sel))
    assert plan["kind"] == "info" and "шаблон" in plan["message"]


def test_music_command(builder, work):
    pa = write(work / "Track A.wav", ts.click_track(120, 50.0, SR), SR)
    pb = write(work / "Track B.wav", ts.click_track(120, 30.0, SR), SR)
    builder.add_file(pa, 0.0)
    builder.add_file(pb, 45.0)
    plan = parse("Перейти с A на B примерно на 40-й секунде. B должен войти на сильную долю. "
                 "Темп и высоту тона не менять. Хвост A сохранить.", ParseContext(builder.project, Selection()))
    o = plan["operations"][0]
    assert o["op"] == "music.create" and o["time"]["at"] == 40
    assert o["params"]["b_on_downbeat"] and o["params"]["keep_a_tail"]
    assert {c["kind"] for c in o["constraints"]} == {"keep_tempo", "keep_pitch"}
    r = builder.engine.apply(plan)
    assert r.applied, r.errors
