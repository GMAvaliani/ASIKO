"""Минимальное описание проекта и инструкции для внешней модели.

Модели отправляется только текст задания и описание структуры проекта
(идентификаторы, времена, параметры). Аудио, видео, пути к файлам и контрольные
суммы не отправляются; имена клипов/дорожек — только если пользователь разрешил.
"""
from __future__ import annotations

import json

from ..commands.schema import OP_SPECS, PLAN_SCHEMA
from ..model.presets import BUILTIN_PRESETS, CURVE_SHAPES, PARAM_GROUPS, PARAM_INFO, ROOMS, SPACE_PARAMS
from ..model.project import Project
from .local_parser import Selection


def project_summary(project: Project, selection: Selection | None, include_names: bool = False) -> dict:
    sel = selection or Selection()

    def nm(obj, fallback):
        return (obj.name or fallback) if include_names else fallback

    tracks = []
    for i, t in enumerate(project.tracks):
        clips = []
        for j, c in enumerate(sorted(t.clips, key=lambda c: c.start)):
            src = project.source(c.source_id)
            d = {"id": c.id, "name": nm(c, f"клип {i + 1}.{j + 1}"), "start": round(c.start, 4),
                 "end": round(c.end, 4), "src_in": round(c.src_in, 4), "locked": c.locked,
                 "gain_db": c.gain_db}
            if src is not None:
                d["source_id"] = src.id
                d["source_duration"] = round(src.duration, 4)
                d["channels"] = src.channels
                if src.beats:
                    d["beat_grid"] = {"bpm": src.beats.bpm, "confidence": src.beats.confidence,
                                      "first_downbeat_src": src.beats.downbeats[0] if src.beats.downbeats else None}
            clips.append(d)
        tracks.append({"id": t.id, "name": nm(t, f"дорожка {i + 1}"), "gain_db": t.gain_db, "mute": t.mute,
                       "solo": t.solo, "clips": clips})
    spaces = []
    for x in project.space_transitions:
        spaces.append({
            "id": x.id, "track_id": x.track_id, "start": x.start, "end": x.end,
            "from": {"preset": x.from_state.preset, "room": x.from_state.room, "values": x.from_state.values},
            "to": {"preset": x.to_state.preset, "room": x.to_state.room, "values": x.to_state.values},
            "curves": {k: {"start": v.start, "end": v.end, "shape": v.shape} for k, v in x.curves.items()},
            "filter_slope": x.filter_slope, "locked_params": x.locked_params, "approved": x.approved,
        })
    musics = []
    for m in project.music_transitions:
        p = m.params
        musics.append({"id": m.id, "clip_a_id": m.clip_a_id, "clip_b_id": m.clip_b_id,
                       "switch_time": p.switch_time, "b_entry_src": p.b_entry_src,
                       "a_fade": [p.a_fade_start, p.a_fade_end], "b_fade": [p.b_fade_start, p.b_fade_end],
                       "curve": p.curve, "a_tail": p.a_tail, "overlap_filter": p.overlap_filter,
                       "variants": [v.kind for v in m.variants],
                       "selected": (m.variant(m.selected_variant).kind if m.selected_variant and m.variant(m.selected_variant) else None),
                       "approved": m.approved})
    return {
        "revision": project.revision,
        "sample_rate": project.sample_rate,
        "content_end": round(project.content_end(), 4),
        "constraints": project.constraints,
        "tracks": tracks,
        "space_transitions": spaces,
        "music_transitions": musics,
        "protections": [{"id": p.id, "start": p.start, "end": p.end, "track_id": p.track_id} for p in project.protections],
        "frozen_regions": [{"id": f.id, "start": f.start, "end": f.end} for f in project.frozen],
        "selection": {"track_id": sel.track_id, "clip_ids": sel.clip_ids, "transition_id": sel.transition_id,
                      "time_range": sel.time_range, "playhead": round(sel.playhead, 3)},
    }


def system_prompt() -> str:
    ops = "\n".join(f"- {name}: {spec.desc}" for name, spec in OP_SPECS.items() if spec.llm)
    params = "\n".join(f"- {p}: {PARAM_INFO[p]['label']}, {PARAM_INFO[p]['min']}…{PARAM_INFO[p]['max']} {PARAM_INFO[p]['unit']}"
                       for p in SPACE_PARAMS)
    presets = ", ".join(f"{k} ({v['name']})" for k, v in BUILTIN_PRESETS.items())
    return f"""Ты — модуль перевода технического задания звукорежиссёра видеоигр (на русском) в план операций приложения ASIKO.
Ты НЕ выполняешь действия сам: ты только возвращаешь JSON-план по схеме {PLAN_SCHEMA}. Приложение проверит план и применит его само.

Правила:
1. Верни ровно один JSON-объект плана без пояснений вокруг. Поле schema = "{PLAN_SCHEMA}", base_revision = revision из описания проекта, origin = "llm".
2. Используй только идентификаторы из описания проекта. Не придумывай объекты, пути к файлам, команды ОС или код.
3. Все времена — в секундах. time.coord: "timeline" (шкала проекта), "relative" (сдвиг), "transition" (секунды от начала перехода), "transition_fraction" (доля 0..1), "transition_end" (секунды до конца перехода, ≤0).
4. Меняй минимально необходимое: не трогай параметры, о которых не просили. Уважай locked_params, approved, protections — если просьба их нарушает, верни kind="info" с объяснением.
5. Если задание существенно неоднозначно — kind="question", одно короткое question.text и до 4 вариантов options[].label с готовым plan.
6. Художественные просьбы без конкретики («драматичнее», «красивее») — kind="proposal", artistic=true, и опиши предложение в message.
7. Темп и высоту тона приложение не меняет никогда. Убрать отдельный инструмент из стереомикса нельзя — объясни это в kind="info".
8. «После N не менять» при создании перехода — constraints [{{"kind": "protect_after", "time": N}}] и переход должен закончиться к N.
9. Группы параметров для space.set_curve: {", ".join(PARAM_GROUPS)} (filter — раскрытие частот, color — эффект радио, room — комнатная обработка).

Операции:
{ops}

Параметры состояний (space):
{params}
Комнаты: {", ".join(ROOMS)}. Формы кривых: {", ".join(CURVE_SHAPES)}. Пресеты: {presets}.
"""


def user_message(text: str, summary: dict) -> str:
    return ("Описание проекта (JSON):\n" + json.dumps(summary, ensure_ascii=False) +
            "\n\nЗадание пользователя:\n" + text.strip())
