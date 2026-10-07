"""Версионируемая схема команд ASIKO: asiko.plan/1.

План — это JSON-объект:

    {
      "schema": "asiko.plan/1",
      "base_revision": 12,             # версия проекта, на которой сформирован план
      "origin": "local" | "llm" | "ui",
      "kind": "edit" | "proposal" | "question" | "info" | "undo" | "redo",
      "artistic": false,               # художественное предложение (не применяется молча)
      "text": "исходный текст задания",
      "message": "сообщение пользователю (для info/question)",
      "question": {"text": "...", "options": [{"label": "...", "plan": {...}}]},
      "operations": [ <операция>, ... ]
    }

Операция:

    {
      "op": "space.create",
      "target": {"track_id": "trk_1a2b3c4d"},
      "time": {"coord": "timeline", "unit": "s", "start": 12, "end": 16},
      "params": {...},
      "constraints": [{"kind": "protect_after", "time": 16}],
      "preconditions": [{"kind": "exists", "id": "trk_1a2b3c4d"}]
    }

Проверка типов/диапазонов/лишних полей выполняется здесь; проверка ссылок на объекты
и совместимости с проектом — в обработчиках на копии проекта (handlers.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from ..model.presets import CURVE_SHAPES, PARAM_GROUPS, PARAM_INFO, ROOMS, SPACE_PARAMS
from ..model.automation import OVERLAP_FILTERS

PLAN_SCHEMA = "asiko.plan/1"
PLAN_KINDS = ("edit", "proposal", "question", "info", "undo", "redo")
ORIGINS = ("local", "llm", "ui")
MAX_OPERATIONS = 64
MAX_TIME_S = 36000.0


@dataclass
class F:
    type: str                       # id|str|num|int|bool|enum|values|curves|list_str|list_num|dict|num_or_null|id_or_null
    required: bool = False
    min: float | None = None
    max: float | None = None
    enum: tuple | None = None
    desc: str = ""


@dataclass
class OpSpec:
    desc: str
    target: dict[str, F] = field(default_factory=dict)
    time: dict[str, F] | None = None
    time_required: bool = False
    params: dict[str, F] = field(default_factory=dict)
    constraints: tuple[str, ...] = ()
    llm: bool = True                 # доступна ли операция внешней модели


_T = lambda coords, extra: {  # noqa: E731
    "coord": F("enum", True, enum=tuple(coords), desc="система координат времени"),
    "unit": F("enum", False, enum=("s", "fraction"), desc="единицы: s — секунды, fraction — доля 0..1"),
    **extra,
}
_num = lambda req=False, mn=-MAX_TIME_S, mx=MAX_TIME_S, desc="": F("num", req, mn, mx, desc=desc)  # noqa: E731

MUSIC_KINDS = ("short", "long", "phrase")
A_TAIL = ("source", "cut", "effect")
XFADE = ("equal_power", "linear")

OP_SPECS: dict[str, OpSpec] = {
    "space.create": OpSpec(
        "Создать переход между двумя состояниями звучания (в сцене ↔ закадровый) на дорожке",
        target={"track_id": F("id", True)},
        time=_T(["timeline"], {"start": _num(True, 0), "end": _num(True, 0)}),
        time_required=True,
        params={
            "from_preset": F("str", desc="ключ пресета начального состояния"),
            "to_preset": F("str", desc="ключ пресета конечного состояния"),
            "from_values": F("values"), "to_values": F("values"),
            "from_room": F("enum", enum=tuple(ROOMS)), "to_room": F("enum", enum=tuple(ROOMS)),
            "curves": F("curves"),
            "filter_slope": F("int", enum=(12, 24)),
            "name": F("str"),
        },
        constraints=("protect_after", "protect_range"),
    ),
    "space.set_timing": OpSpec(
        "Изменить начало/конец перехода (абсолютно или сдвигом)",
        target={"transition_id": F("id", True)},
        time=_T(["timeline", "relative"], {
            "start": _num(False, 0), "end": _num(False, 0),
            "start_delta": _num(), "end_delta": _num()}),
        time_required=True,
        constraints=("protect_after", "protect_range"),
    ),
    "space.set_curve": OpSpec(
        "Изменить, когда и как меняются параметры внутри перехода",
        target={"transition_id": F("id", True)},
        time=_T(["transition", "transition_fraction", "transition_end", "timeline"], {
            "start": _num(), "end": _num()}),
        params={"params": F("list_str"), "group": F("enum", enum=tuple(PARAM_GROUPS)),
                "shape": F("enum", enum=tuple(CURVE_SHAPES))},
    ),
    "space.set_state": OpSpec(
        "Изменить начальное (from) или конечное (to) состояние перехода",
        target={"transition_id": F("id", True)},
        params={"state": F("enum", True, enum=("from", "to")), "preset": F("str"),
                "values": F("values"), "room": F("enum", enum=tuple(ROOMS))},
    ),
    "space.set_slope": OpSpec(
        "Крутизна фильтров перехода",
        target={"transition_id": F("id", True)},
        params={"filter_slope": F("int", True, enum=(12, 24))},
    ),
    "space.delete": OpSpec("Удалить переход", target={"transition_id": F("id", True)}),
    "transition.lock_params": OpSpec(
        "Зафиксировать/разблокировать отдельные параметры перехода",
        target={"transition_id": F("id", True)},
        params={"params": F("list_str", True), "locked": F("bool", True)},
    ),
    "transition.approve": OpSpec(
        "Утвердить (заблокировать) переход или снять утверждение",
        target={"transition_id": F("id", True)},
        params={"approved": F("bool", True)},
    ),
    "music.create": OpSpec(
        "Создать переход между двумя музыкальными клипами A → B с вариантами",
        target={"clip_a_id": F("id", True), "clip_b_id": F("id", True)},
        time=_T(["timeline"], {"at": _num(True, 0, desc="желаемый момент перехода")}),
        time_required=True,
        params={"b_on_downbeat": F("bool"), "keep_a_tail": F("bool"), "exact": F("bool"),
                "overlap_filter": F("enum", enum=tuple(OVERLAP_FILTERS)),
                "select": F("enum", enum=MUSIC_KINDS), "name": F("str")},
        constraints=("keep_tempo", "keep_pitch"),
    ),
    "music.select_variant": OpSpec(
        "Выбрать вариант перехода A → B",
        target={"transition_id": F("id", True)},
        params={"variant": F("str", True, desc="short|long|phrase или id варианта")},
    ),
    "music.set": OpSpec(
        "Ручная правка активных параметров перехода A → B",
        target={"transition_id": F("id", True)},
        params={
            "switch_time": _num(False, 0), "b_entry_src": _num(False, 0),
            "a_fade_start": _num(False, 0), "a_fade_len": _num(False, 0, 120),
            "b_fade_start": _num(False, 0), "b_fade_len": _num(False, 0, 120),
            "curve": F("enum", enum=XFADE), "a_tail": F("enum", enum=A_TAIL),
            "overlap_filter": F("enum", enum=tuple(OVERLAP_FILTERS)),
            "shift": _num(False, -600, 600, "сдвиг всего перехода (с)"),
            "extend": _num(False, -120, 120, "удлинение наложения (с)"),
            "b_entry_shift": _num(False, -600, 600, "сдвиг точки входа B в исходнике (с)"),
            "b_entry_shift_beats": _num(False, -64, 64, "сдвиг точки входа B в долях"),
        },
    ),
    "music.delete": OpSpec("Удалить переход A → B (клипы возвращаются на места)",
                           target={"transition_id": F("id", True)}),
    "track.add": OpSpec("Добавить дорожку", params={"name": F("str")}),
    "track.set": OpSpec(
        "Изменить дорожку: громкость, mute, solo, имя",
        target={"track_id": F("id", True)},
        params={"gain_db": _num(False, -60, 12), "delta_db": _num(False, -72, 72),
                "mute": F("bool"), "solo": F("bool"), "name": F("str")},
    ),
    "track.delete": OpSpec("Удалить дорожку вместе с клипами", target={"track_id": F("id", True)}),
    "clip.add": OpSpec(
        "Поместить импортированный исходник на дорожку",
        target={"source_id": F("id", True), "track_id": F("id")},
        time=_T(["timeline"], {"start": _num(False, 0)}),
        params={"src_in": _num(False, 0), "src_out": _num(False, 0), "name": F("str")},
    ),
    "clip.move": OpSpec(
        "Переместить клип (связанные переходы переносятся вместе с ним)",
        target={"clip_id": F("id", True)},
        time=_T(["timeline", "relative"], {"start": _num(False, 0), "delta": _num()}),
        params={"track_id": F("id")},
    ),
    "clip.set": OpSpec(
        "Изменить клип: усиление, фейды, границы, блокировку",
        target={"clip_id": F("id", True)},
        params={"gain_db": _num(False, -60, 24), "delta_db": _num(False, -72, 72),
                "fade_in": _num(False, 0, 60), "fade_out": _num(False, 0, 60),
                "fade_shape": F("enum", enum=("smooth", "linear", "equal_power")),
                "src_in": _num(False, 0), "src_out": _num(False, 0),
                "locked": F("bool"), "name": F("str")},
    ),
    "clip.delete": OpSpec("Удалить клип с дорожки", target={"clip_id": F("id", True)}),
    "constraint.set": OpSpec(
        "Ограничения проекта: не менять темп / высоту тона",
        params={"keep_tempo": F("bool"), "keep_pitch": F("bool")},
    ),
    "protect.add": OpSpec(
        "Защитить временной участок от изменения настроек",
        params={"start": _num(True, 0), "end": F("num_or_null", False, 0, MAX_TIME_S),
                "track_id": F("id_or_null"), "note": F("str")},
    ),
    "protect.remove": OpSpec("Снять защиту участка", target={"protection_id": F("id", True)}),
    "beats.set": OpSpec(
        "Задать сетку долей исходника (ручная разметка или результат анализа)",
        target={"source_id": F("id", True)},
        params={"grid": F("dict"), "bpm": _num(False, 20, 400), "first_downbeat": _num(False, 0),
                "beats_per_bar": F("int", False, 1, 16), "phrase_bars": F("int", False, 1, 64)},
    ),
    "beats.shift": OpSpec(
        "Сдвинуть сетку долей исходника",
        target={"source_id": F("id", True)},
        params={"delta": _num(True, -60, 60)},
    ),
    "phrases.set": OpSpec(
        "Задать границы музыкальных фраз (время исходника)",
        target={"source_id": F("id", True)},
        params={"phrases": F("list_num", True)},
    ),
    "freeze.add": OpSpec(
        "Зафиксировать участок в отрендеренный PCM",
        params={"start": _num(True, 0), "end": _num(True, 0), "file": F("str", True),
                "sha256": F("str", True), "sample_rate": F("int", True, 8000, 384000),
                "margin": _num(False, 0, 1), "note": F("str"), "id": F("id")},
        llm=False,
    ),
    "freeze.remove": OpSpec("Снять фиксацию PCM", target={"frozen_id": F("id", True)}),
    "video.set": OpSpec("Подключить видео для просмотра",
                        params={"path": F("str", True), "offset": _num(False, -3600, 3600),
                                "name": F("str")}, llm=False),
    "video.offset": OpSpec("Смещение видео на шкале", params={"offset": _num(True, -3600, 3600)}),
    "video.clear": OpSpec("Отключить видео"),
    "project.rename": OpSpec("Переименовать проект", params={"name": F("str", True)}),
    "source.add": OpSpec("Добавить исходник в библиотеку (из интерфейса)",
                         params={"source": F("dict", True)}, llm=False),
    "source.relink": OpSpec("Обновить путь к перемещённому исходнику",
                            target={"source_id": F("id", True)},
                            params={"path": F("str", True)}, llm=False),
    "source.remove": OpSpec("Удалить неиспользуемый исходник из библиотеки",
                            target={"source_id": F("id", True)}),
}

CONSTRAINT_KINDS = {
    "protect_after": {"time": F("num", True, 0, MAX_TIME_S)},
    "protect_range": {"start": F("num", True, 0, MAX_TIME_S), "end": F("num_or_null", False, 0, MAX_TIME_S)},
    "keep_tempo": {},
    "keep_pitch": {},
}

PRECONDITION_KINDS = {
    "exists": {"id": F("id", True)},
    "not_approved": {"id": F("id", True)},
    "revision": {"value": F("int", True, 0, 10**9)},
    "duration_at_least": {"id": F("id", True), "seconds": F("num", True, 0, MAX_TIME_S)},
}

OP_KEYS = {"op", "target", "time", "params", "constraints", "preconditions", "note"}
PLAN_KEYS = {"schema", "base_revision", "origin", "kind", "artistic", "text", "message",
             "question", "operations", "summary"}


class SchemaError(ValueError):
    def __init__(self, errors: list[str]):
        super().__init__("; ".join(errors))
        self.errors = errors


def _check_field(path: str, spec: F, v: Any, errs: list[str]) -> None:
    t = spec.type
    if t in ("num_or_null", "id_or_null") and v is None:
        return
    if t == "id" or t == "id_or_null":
        if not isinstance(v, str) or not v or len(v) > 64:
            errs.append(f"{path}: ожидался идентификатор (строка)")
        return
    if t == "str":
        if not isinstance(v, str) or len(v) > 2000:
            errs.append(f"{path}: ожидалась строка")
        elif spec.enum and v not in spec.enum:
            errs.append(f"{path}: недопустимое значение «{v}»")
        return
    if t == "bool":
        if not isinstance(v, bool):
            errs.append(f"{path}: ожидалось true/false")
        return
    if t == "enum":
        if v not in (spec.enum or ()):
            errs.append(f"{path}: недопустимое значение {v!r}; допустимо: {', '.join(map(str, spec.enum or ()))}")
        return
    if t in ("num", "num_or_null", "int"):
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            errs.append(f"{path}: ожидалось число")
            return
        if not math.isfinite(float(v)):
            errs.append(f"{path}: недопустимое число")
            return
        if t == "int" and float(v) != int(v):
            errs.append(f"{path}: ожидалось целое число")
            return
        if spec.enum and v not in spec.enum:
            errs.append(f"{path}: допустимо {', '.join(map(str, spec.enum))}")
            return
        if spec.min is not None and v < spec.min:
            errs.append(f"{path}: {v} меньше допустимого минимума {spec.min:g}")
        if spec.max is not None and v > spec.max:
            errs.append(f"{path}: {v} больше допустимого максимума {spec.max:g}")
        return
    if t == "values":
        if not isinstance(v, dict):
            errs.append(f"{path}: ожидался объект параметров")
            return
        from ..model.presets import check_param_value

        for k, val in v.items():
            e = check_param_value(k, val)
            if e:
                errs.append(f"{path}.{k}: {e}")
        return
    if t == "curves":
        if not isinstance(v, dict):
            errs.append(f"{path}: ожидался объект кривых")
            return
        for k, c in v.items():
            if k not in SPACE_PARAMS:
                errs.append(f"{path}: неизвестный параметр «{k}»")
                continue
            if not isinstance(c, dict) or set(c) - {"start", "end", "shape"}:
                errs.append(f"{path}.{k}: ожидалось {{start, end, shape}}")
                continue
            for kk in ("start", "end"):
                if kk in c:
                    _check_field(f"{path}.{k}.{kk}", F("num", False, 0, 1), c[kk], errs)
            if "shape" in c:
                _check_field(f"{path}.{k}.shape", F("enum", enum=tuple(CURVE_SHAPES)), c["shape"], errs)
        return
    if t == "list_str":
        if not isinstance(v, list) or not all(isinstance(x, str) for x in v) or len(v) > 64:
            errs.append(f"{path}: ожидался список строк")
        return
    if t == "list_num":
        if (not isinstance(v, list) or len(v) > 10000
                or not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in v)):
            errs.append(f"{path}: ожидался список чисел")
        return
    if t == "dict":
        if not isinstance(v, dict):
            errs.append(f"{path}: ожидался объект")
        return
    errs.append(f"{path}: неизвестный тип поля {t}")


def _check_obj(path: str, specs: dict[str, F], obj: Any, errs: list[str]) -> None:
    if obj is None:
        obj = {}
    if not isinstance(obj, dict):
        errs.append(f"{path}: ожидался объект")
        return
    for k in obj:
        if k not in specs:
            errs.append(f"{path}: неизвестное поле «{k}»")
    for k, spec in specs.items():
        if k not in obj:
            if spec.required:
                errs.append(f"{path}.{k}: обязательное поле отсутствует")
            continue
        _check_field(f"{path}.{k}", spec, obj[k], errs)


def validate_operation(i: int, op: Any, origin: str, errs: list[str]) -> None:
    path = f"operations[{i}]"
    if not isinstance(op, dict):
        errs.append(f"{path}: ожидался объект")
        return
    for k in op:
        if k not in OP_KEYS:
            errs.append(f"{path}: неизвестное поле «{k}»")
    name = op.get("op")
    spec = OP_SPECS.get(name)
    if spec is None:
        errs.append(f"{path}.op: неизвестная операция {name!r}")
        return
    if origin == "llm" and not spec.llm:
        errs.append(f"{path}: операция «{name}» недоступна внешней модели")
        return
    path = f"{path}({name})"
    _check_obj(f"{path}.target", spec.target, op.get("target"), errs)
    if spec.time is None:
        if op.get("time") not in (None, {}):
            errs.append(f"{path}.time: операция не принимает время")
    else:
        if op.get("time") is None:
            if spec.time_required:
                errs.append(f"{path}.time: обязательное поле отсутствует")
        else:
            _check_obj(f"{path}.time", spec.time, op.get("time"), errs)
    _check_obj(f"{path}.params", spec.params, op.get("params"), errs)
    cons = op.get("constraints", [])
    if not isinstance(cons, list):
        errs.append(f"{path}.constraints: ожидался список")
    else:
        for j, c in enumerate(cons):
            if not isinstance(c, dict) or c.get("kind") not in CONSTRAINT_KINDS:
                errs.append(f"{path}.constraints[{j}]: неизвестное ограничение")
                continue
            if c["kind"] not in spec.constraints:
                errs.append(f"{path}.constraints[{j}]: «{c['kind']}» не применимо к операции")
                continue
            sub = dict(c)
            sub.pop("kind")
            _check_obj(f"{path}.constraints[{j}]", CONSTRAINT_KINDS[c["kind"]], sub, errs)
    pre = op.get("preconditions", [])
    if not isinstance(pre, list):
        errs.append(f"{path}.preconditions: ожидался список")
    else:
        for j, c in enumerate(pre):
            if not isinstance(c, dict) or c.get("kind") not in PRECONDITION_KINDS:
                errs.append(f"{path}.preconditions[{j}]: неизвестное предусловие")
                continue
            sub = dict(c)
            sub.pop("kind")
            _check_obj(f"{path}.preconditions[{j}]", PRECONDITION_KINDS[c["kind"]], sub, errs)


def validate_plan_schema(plan: Any) -> list[str]:
    """Строгая проверка формы плана. Возвращает список ошибок (пустой — форма верна)."""
    errs: list[str] = []
    if not isinstance(plan, dict):
        return ["план должен быть JSON-объектом"]
    for k in plan:
        if k not in PLAN_KEYS:
            errs.append(f"неизвестное поле плана «{k}»")
    if plan.get("schema") != PLAN_SCHEMA:
        errs.append(f"schema: ожидалось «{PLAN_SCHEMA}», получено {plan.get('schema')!r}")
    br = plan.get("base_revision")
    if isinstance(br, bool) or not isinstance(br, int) or br < 0:
        errs.append("base_revision: ожидалось неотрицательное целое число")
    origin = plan.get("origin", "ui")
    if origin not in ORIGINS:
        errs.append(f"origin: недопустимое значение {origin!r}")
    kind = plan.get("kind", "edit")
    if kind not in PLAN_KINDS:
        errs.append(f"kind: недопустимое значение {kind!r}")
    if "artistic" in plan and not isinstance(plan["artistic"], bool):
        errs.append("artistic: ожидалось true/false")
    for k in ("text", "message", "summary"):
        if k in plan and not isinstance(plan[k], str):
            errs.append(f"{k}: ожидалась строка")
    ops = plan.get("operations", [])
    if not isinstance(ops, list):
        errs.append("operations: ожидался список")
        ops = []
    if len(ops) > MAX_OPERATIONS:
        errs.append(f"operations: слишком много операций ({len(ops)} > {MAX_OPERATIONS})")
    if kind in ("edit", "proposal") and not ops:
        errs.append("operations: план изменения не содержит операций")
    for i, op in enumerate(ops):
        validate_operation(i, op, origin, errs)
    q = plan.get("question")
    if q is not None:
        if not isinstance(q, dict) or not isinstance(q.get("text"), str):
            errs.append("question: ожидалось {text, options}")
        else:
            opts = q.get("options", [])
            if not isinstance(opts, list) or len(opts) > 6:
                errs.append("question.options: ожидался список (до 6)")
            else:
                for j, o in enumerate(opts):
                    if not isinstance(o, dict) or not isinstance(o.get("label"), str):
                        errs.append(f"question.options[{j}]: ожидалось {{label, plan}}")
                    elif "plan" in o and o["plan"] is not None:
                        sub = validate_plan_schema(o["plan"])
                        errs.extend(f"question.options[{j}].plan: {e}" for e in sub)
    return errs


def make_plan(operations: list[dict], base_revision: int, origin: str = "ui", kind: str = "edit",
              text: str = "", artistic: bool = False, message: str = "",
              question: dict | None = None) -> dict:
    p = {"schema": PLAN_SCHEMA, "base_revision": int(base_revision), "origin": origin,
         "kind": kind, "artistic": artistic, "text": text, "operations": operations}
    if message:
        p["message"] = message
    if question:
        p["question"] = question
    return p


def op(name: str, target: dict | None = None, time: dict | None = None, params: dict | None = None,
       constraints: list | None = None, preconditions: list | None = None) -> dict:
    d: dict = {"op": name}
    if target:
        d["target"] = target
    if time:
        d["time"] = time
    if params:
        d["params"] = params
    if constraints:
        d["constraints"] = constraints
    if preconditions:
        d["preconditions"] = preconditions
    return d


# --------------------------------------------------------------------------- JSON Schema


def _field_schema(f: F) -> dict:
    t = f.type
    if t in ("id", "str"):
        s: dict = {"type": "string"}
    elif t == "id_or_null":
        s = {"type": ["string", "null"]}
    elif t == "bool":
        s = {"type": "boolean"}
    elif t == "enum":
        s = {"enum": list(f.enum or ())}
    elif t == "num":
        s = {"type": "number"}
    elif t == "num_or_null":
        s = {"type": ["number", "null"]}
    elif t == "int":
        s = {"type": "integer"}
        if f.enum:
            s["enum"] = list(f.enum)
    elif t == "values":
        s = {"type": "object", "properties": {p: {"type": "number", "minimum": PARAM_INFO[p]["min"],
                                                  "maximum": PARAM_INFO[p]["max"]} for p in SPACE_PARAMS},
             "additionalProperties": False}
    elif t == "curves":
        s = {"type": "object", "additionalProperties": False, "properties": {
            p: {"type": "object", "additionalProperties": False, "properties": {
                "start": {"type": "number", "minimum": 0, "maximum": 1},
                "end": {"type": "number", "minimum": 0, "maximum": 1},
                "shape": {"enum": list(CURVE_SHAPES)}}} for p in SPACE_PARAMS}}
    elif t == "list_str":
        s = {"type": "array", "items": {"type": "string"}}
    elif t == "list_num":
        s = {"type": "array", "items": {"type": "number"}}
    else:
        s = {"type": "object"}
    if t in ("num", "num_or_null", "int") and not f.enum:
        if f.min is not None:
            s["minimum"] = f.min
        if f.max is not None:
            s["maximum"] = f.max
    if f.desc:
        s["description"] = f.desc
    return s


def _obj_schema(specs: dict[str, F]) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": {k: _field_schema(v) for k, v in specs.items()},
            "required": [k for k, v in specs.items() if v.required]}


def operations_json_schema(llm_only: bool = True) -> list[dict]:
    out = []
    for name, spec in OP_SPECS.items():
        if llm_only and not spec.llm:
            continue
        props: dict = {"op": {"const": name}, "target": _obj_schema(spec.target),
                       "params": _obj_schema(spec.params)}
        if spec.time is not None:
            props["time"] = _obj_schema(spec.time)
        if spec.constraints:
            props["constraints"] = {"type": "array", "items": {"type": "object", "properties": {
                "kind": {"enum": list(spec.constraints)}}, "required": ["kind"]}}
        props["preconditions"] = {"type": "array", "items": {"type": "object", "properties": {
            "kind": {"enum": list(PRECONDITION_KINDS)}}, "required": ["kind"]}}
        req = ["op"]
        if spec.target and any(f.required for f in spec.target.values()):
            req.append("target")
        if spec.time_required:
            req.append("time")
        out.append({"type": "object", "description": spec.desc, "properties": props,
                    "required": req, "additionalProperties": False})
    return out


def plan_json_schema(llm_only: bool = True) -> dict:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": PLAN_SCHEMA,
        "type": "object",
        "additionalProperties": False,
        "required": ["schema", "base_revision", "kind", "operations"],
        "properties": {
            "schema": {"const": PLAN_SCHEMA},
            "base_revision": {"type": "integer", "minimum": 0},
            "origin": {"enum": list(ORIGINS)},
            "kind": {"enum": list(PLAN_KINDS)},
            "artistic": {"type": "boolean"},
            "text": {"type": "string"},
            "summary": {"type": "string"},
            "message": {"type": "string"},
            "question": {"type": "object", "properties": {
                "text": {"type": "string"},
                "options": {"type": "array", "items": {"type": "object", "properties": {
                    "label": {"type": "string"}, "plan": {"type": "object"}}}}}},
            "operations": {"type": "array", "maxItems": MAX_OPERATIONS,
                           "items": {"anyOf": operations_json_schema(llm_only)}},
        },
    }
