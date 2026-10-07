"""Локальный обработчик типовых команд (без сети и без модели).

ВАЖНО: это набор шаблонов, а не понимание произвольного русского языка.
Поддерживаемые шаблоны перечислены в SUPPORTED_TEMPLATES и показываются в интерфейсе.
Результат — план asiko.plan/1, который проходит ту же проверку, что и план от модели.
"""
from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field

from ..commands.schema import make_plan, op
from ..model.presets import GROUP_LABELS
from ..model.project import Project

SUPPORTED_TEMPLATES = [
    "Сделай переход с 12 до 16 секунды",
    "До 12-й секунды музыка звучит из радио справа в комнате. С 12-й по 16-ю переходит в закадровую. После 16-й не менять",
    "Продли выбранный переход на 2 секунды / Продли переход до 18 секунды",
    "Сдвинь выбранный переход на 1 секунду вправо (влево)",
    "Перенеси раскрытие частот в конец / В выбранном переходе эффект радио убирай только в самом конце",
    "Ширину меняй в начале / Комнату убирай на последних 2 секундах / … по всей длине",
    "Уменьши (увеличь) громкость выбранной дорожки на 3 дБ",
    "Перейти с A на B примерно на 40-й секунде. B должен войти на сильную долю. Хвост A сохранить",
    "Не меняй темп и высоту тона",
    "Выбери длинный вариант / вариант 3",
    "Сдвинь вход B на одну долю позже",
    "Защити участок после 16 секунды",
    "Заглуши (включи) выбранную дорожку / соло",
    "Отмени последнее изменение / Повтори отменённое",
    "Сделай драматичнее (художественное предложение, применяется только по подтверждению)",
]

_NUM_WORDS = {
    "ноль": 0, "нуля": 0, "один": 1, "одну": 1, "одна": 1, "одной": 1, "одного": 1, "два": 2, "две": 2,
    "двух": 2, "три": 3, "трех": 3, "четыре": 4, "четырех": 4, "пять": 5, "пяти": 5, "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7, "восемь": 8, "восьми": 8, "девять": 9, "девяти": 9, "десять": 10, "десяти": 10,
    "пятнадцать": 15, "двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50, "шестьдесят": 60,
    "полторы": 1.5, "полтора": 1.5, "полсекунды": 0.5, "пол": 0.5,
}
_ORD = {
    "перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8,
    "девят": 9, "десят": 10, "одиннадцат": 11, "двенадцат": 12, "тринадцат": 13, "четырнадцат": 14,
    "пятнадцат": 15, "шестнадцат": 16, "семнадцат": 17, "восемнадцат": 18, "девятнадцат": 19,
    "двадцат": 20, "тридцат": 30, "сороков": 40, "пятидесят": 50, "шестидесят": 60,
}
_TENS = {"двадцать": 20, "тридцать": 30, "сорок": 40, "пятьдесят": 50}
_UNIT_ORD = {"перв": 1, "втор": 2, "трет": 3, "четверт": 4, "пят": 5, "шест": 6, "седьм": 7, "восьм": 8, "девят": 9}

NUM = r"(\d+(?:[.,]\d+)?)"
SUF = r"(?:\s*-?\s*(?:й|ой|ей|ю|ую|я|го|ого|ему|ой|м))?"
SEC = r"(?:\s*(?:секунд\w*|сек\.?|с\b))?"


def _num(s: str) -> float:
    return float(s.replace(",", "."))


def normalize(text: str) -> str:
    t = text.lower().replace("ё", "е")
    t = re.sub(r"[«»\"“”„]", " ", t)
    t = t.replace("→", " на ").replace("->", " на ")
    # время мм:сс → секунды
    t = re.sub(r"\b(\d{1,2}):(\d{2}(?:[.,]\d+)?)\b", lambda m: f"{int(m.group(1)) * 60 + _num(m.group(2)):g}", t)
    words = t.split()
    out = []
    i = 0
    while i < len(words):
        w = words[i]
        core = re.sub(r"[^\w-]", "", w)
        tail = w[len(core):] if w.startswith(core) else ""
        val = None
        if core in _TENS and i + 1 < len(words):
            nxt = re.sub(r"[^\w-]", "", words[i + 1])
            unit = next((v for k, v in _UNIT_ORD.items() if nxt.startswith(k)), None)
            if unit is None and nxt in _NUM_WORDS and _NUM_WORDS[nxt] < 10:
                unit = _NUM_WORDS[nxt]
            if unit is not None:
                out.append(str(_TENS[core] + unit) + re.sub(r"[\w-]", "", words[i + 1]))
                i += 2
                continue
        if core in _NUM_WORDS:
            val = _NUM_WORDS[core]
        else:
            for k, v in sorted(_ORD.items(), key=lambda kv: -len(kv[0])):
                if core.startswith(k) and len(core) - len(k) <= 4 and re.fullmatch(k + r"(ой|ая|ую|ые|ых|ым|ого|ому|ий|ей|ья|ье|ою|ий)", core):
                    val = v
                    break
        if val is not None:
            out.append(f"{val:g}" + tail)
        else:
            out.append(w)
        i += 1
    t = " ".join(out)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Selection:
    track_id: str | None = None
    clip_ids: list[str] = field(default_factory=list)
    transition_id: str | None = None
    time_range: tuple[float, float] | None = None
    playhead: float = 0.0


@dataclass
class ParseContext:
    project: Project
    selection: Selection = field(default_factory=Selection)


class _Need(Exception):
    """Требуется уточнение: содержит вопрос."""

    def __init__(self, question: dict):
        super().__init__(question.get("text", ""))
        self.question = question


# --------------------------------------------------------------------------- helpers


def _plan(ctx: ParseContext, ops: list[dict], text: str, **kw) -> dict:
    return make_plan(ops, ctx.project.revision, origin="local", text=text, **kw)


def _info(ctx: ParseContext, text: str, message: str) -> dict:
    return make_plan([], ctx.project.revision, origin="local", kind="info", text=text, message=message)


def _question(ctx: ParseContext, text: str, q: str, options: list[tuple[str, dict]]) -> dict:
    return make_plan([], ctx.project.revision, origin="local", kind="question", text=text,
                     question={"text": q, "options": [{"label": lab, "plan": pl} for lab, pl in options]})


def _target_track(ctx: ParseContext, text: str, build) -> str:
    p = ctx.project
    if ctx.selection.track_id and p.track(ctx.selection.track_id):
        return ctx.selection.track_id
    m = re.search(r"дорожк\w*\s+(\d+)", text) or re.search(r"(\d+)\s*(?:-?\s*\w*)?\s+дорожк", text)
    if m:
        idx = int(float(m.group(1))) - 1
        if 0 <= idx < len(p.tracks):
            return p.tracks[idx].id
    for t in p.tracks:
        if t.name and t.name.lower() in text:
            return t.id
    with_clips = [t for t in p.tracks if t.clips]
    if len(with_clips) == 1:
        return with_clips[0].id
    if not p.tracks:
        raise _NoTarget("В проекте нет дорожек — сначала импортируйте аудио.")
    raise _Need({"text": "К какой дорожке применить?",
                 "options": [{"label": t.name, "plan": build(t.id)} for t in p.tracks[:6]]})


class _NoTarget(Exception):
    pass


def _target_transition(ctx: ParseContext, kind: str | None, build) -> str:
    p = ctx.project
    sel = ctx.selection.transition_id
    if sel and p.any_transition(sel):
        if kind == "space" and not p.space_transition(sel):
            pass
        elif kind == "music" and not p.music_transition(sel):
            pass
        else:
            return sel
    cands = []
    if kind in (None, "space"):
        cands += [(x.id, f"{x.name} ({x.start:g}–{x.end:g} с)") for x in p.space_transitions]
    if kind in (None, "music"):
        cands += [(x.id, f"A→B {x.name} (≈{x.params.switch_time:.1f} с)") for x in p.music_transitions]
    if len(cands) == 1:
        return cands[0][0]
    if not cands:
        raise _NoTarget("В проекте нет подходящего перехода. Сначала создайте переход.")
    raise _Need({"text": "Какой переход изменить?",
                 "options": [{"label": lab, "plan": build(xid)} for xid, lab in cands[:6]]})


def _range(text: str) -> tuple[float, float] | None:
    m = re.search(r"(?:^|\s)с\s*" + NUM + SUF + SEC + r"\s*(?:до|по)\s*" + NUM + SUF, text)
    if m:
        return _num(m.group(1)), _num(m.group(2))
    m = re.search(NUM + r"\s*[-–—]\s*" + NUM + r"\s*(?:секунд\w*|сек|с)\b", text)
    if m:
        return _num(m.group(1)), _num(m.group(2))
    return None


def _amount(text: str, unit: str = SEC) -> float | None:
    m = re.search(r"\bна\s*" + NUM + unit, text)
    if m:
        return _num(m.group(1))
    if re.search(r"\bна\s+секунду\b", text):
        return 1.0
    if re.search(r"\bна\s+полсекунды\b", text):
        return 0.5
    return None


GROUP_WORDS = [
    ("filter", r"раскрыти\w*\s+част\w*|частот\w*|фильтр\w*|спектр\w*|диапазон\w*|глух\w*"),
    ("color", r"эффект\w*\s+радио|радио\w*|телефон\w*|окраск\w*|окрас\w*"),
    ("width", r"ширин\w*|стерео\w*"),
    ("room", r"комнат\w*|ревер\w*|помещени\w*|пространств\w*|\bэхо\b"),
    ("pan", r"панорам\w*"),
    ("level", r"громкост\w*"),
]


def _groups(text: str) -> list[str]:
    out = []
    for g, rx in GROUP_WORDS:
        if re.search(rx, text):
            out.append(g)
    if "color" in out and "filter" in out and re.search(r"эффект\w*\s+радио", text) and not re.search(r"частот", text):
        out.remove("filter")
    return out


def _curve_window(text: str, length: float) -> dict | None:
    m = re.search(r"последн\w*\s+" + NUM + SEC, text)
    if m:
        v = min(_num(m.group(1)), length)
        return {"coord": "transition_end", "unit": "s", "start": -v, "end": 0.0}
    m = re.search(r"перв\w*\s+" + NUM + SEC, text)
    if m:
        v = min(_num(m.group(1)), length)
        return {"coord": "transition", "unit": "s", "start": 0.0, "end": v}
    rng = _range(text)
    if rng:
        return {"coord": "timeline", "unit": "s", "start": rng[0], "end": rng[1]}
    if re.search(r"в\s+самом\s+конце|в\s+самый\s+конец|в\s+последний\s+момент", text):
        v = 1.0 if length >= 2.0 else length * 0.4
        return {"coord": "transition_end", "unit": "s", "start": -v, "end": 0.0}
    if re.search(r"\bв\s+конц\w*|\bв\s+конец\b|\bк\s+концу\b|ближе\s+к\s+концу", text):
        v = 2.0 if length >= 3.0 else length * 0.5
        return {"coord": "transition_end", "unit": "s", "start": -v, "end": 0.0}
    if re.search(r"в\s+самом\s+начале", text):
        v = 1.0 if length >= 2.0 else length * 0.4
        return {"coord": "transition", "unit": "s", "start": 0.0, "end": v}
    if re.search(r"\bв\s+начал\w*|\bв\s+начало\b|\bсразу\b", text):
        v = 2.0 if length >= 3.0 else length * 0.5
        return {"coord": "transition", "unit": "s", "start": 0.0, "end": v}
    if re.search(r"в\s+середин\w*", text):
        return {"coord": "transition_fraction", "unit": "fraction", "start": 1 / 3, "end": 2 / 3}
    if re.search(r"по\s+всей\s+длин\w*|на\s+всю\s+длин\w*|равномерно|весь\s+переход", text):
        return {"coord": "transition_fraction", "unit": "fraction", "start": 0.0, "end": 1.0}
    if re.search(r"в\s+первой\s+половин\w*|1\s+половин\w*", text):
        return {"coord": "transition_fraction", "unit": "fraction", "start": 0.0, "end": 0.5}
    if re.search(r"во\s+второй\s+половин\w*|2\s+половин\w*", text):
        return {"coord": "transition_fraction", "unit": "fraction", "start": 0.5, "end": 1.0}
    return None


def _shape(text: str) -> str | None:
    if re.search(r"линейн\w*", text):
        return "linear"
    if re.search(r"плавн\w*|мягк\w*", text):
        return "smooth"
    if re.search(r"резк\w*", text):
        return "ease_in"
    return None


# --------------------------------------------------------------------------- intents


def _space_create(ctx: ParseContext, text: str, rng) -> dict:
    p = ctx.project
    start, end = rng
    if end <= start:
        return _info(ctx, text, f"Конец перехода ({end:g} с) должен быть позже начала ({start:g} с).")
    preset = "radio_room"
    if re.search(r"телефон", text):
        preset = "phone"
    elif re.search(r"издалек|удален\w*|вдалек|далек\w*", text):
        preset = "distant"
    elif re.search(r"за\s+стен|за\s+двер|за\s+окн|приглуш|препятств", text):
        preset = "muffled"
    elif re.search(r"радио|приемник", text):
        preset = "radio_room"
    reverse = bool(re.search(r"из\s+закадр\w*|обратно\s+в\s+сцен|в\s+сцену|становится\s+(?:звуком\s+)?в\s+сцене", text))
    vals = {}
    if re.search(r"крайне\s+справа|справа\s+до\s+упора|полностью\s+справа", text):
        vals["pan"] = 1.0
    elif re.search(r"справа|правее|с\s+правой|в\s+правой", text):
        vals["pan"] = 0.6
    elif re.search(r"крайне\s+слева|полностью\s+слева", text):
        vals["pan"] = -1.0
    elif re.search(r"слева|левее|с\s+левой|в\s+левой", text):
        vals["pan"] = -0.6
    elif re.search(r"по\s+центру|в\s+центре", text):
        vals["pan"] = 0.0
    params: dict = {}
    room = None
    if re.search(r"в\s+зале|больш\w+\s+помещени", text):
        room = "hall"
    elif re.search(r"без\s+комнат|без\s+ревер", text):
        room = "none"
    elif re.search(r"в\s+комнате", text):
        room = "room"
    scene_key = "from" if not reverse else "to"
    params[f"{scene_key}_preset"] = preset
    params[f"{'to' if not reverse else 'from'}_preset"] = "offscreen"
    if vals:
        params[f"{scene_key}_values"] = vals
    if room:
        params[f"{scene_key}_room"] = room
    if re.search(r"без\s+(?:резк\w+\s+)?скачк\w*\s+громкост|плавн\w*\s+по\s+громкост|громкост\w*\s+плавн", text):
        params["curves"] = {"gain_db": {"start": 0.0, "end": 1.0, "shape": "smooth"}}
    cons = []
    m = re.search(r"после\s*" + NUM + SUF + SEC + r"\s*(?:\S+\s+){0,3}?(?:не\s+меня\w*|не\s+измен\w*|не\s+трога\w*|без\s+изменени\w*|оставить|оставь)", text)
    if m:
        cons.append({"kind": "protect_after", "time": _num(m.group(1))})
    elif re.search(r"(?:дальше|потом|после\s+этого)\s+не\s+меня", text):
        cons.append({"kind": "protect_after", "time": end})

    def build(tid):
        return _plan(ctx, [op("space.create", target={"track_id": tid},
                              time={"coord": "timeline", "unit": "s", "start": start, "end": end},
                              params=params, constraints=cons or None)], text)

    tid = _target_track(ctx, text, build)
    return build(tid)


def _resolve_ab(ctx: ParseContext, text: str, build) -> tuple[str, str]:
    p = ctx.project
    sel = [c for c in ctx.selection.clip_ids if p.clip(c)]
    if len(sel) == 2:
        a, b = sorted(sel, key=lambda c: p.clip(c).start)
        return a, b
    clips = sorted(p.all_clips(), key=lambda c: c.start)
    named_a = [c for c in clips if re.search(r"(?:^|[\s_\-(])a(?:$|[\s_\-)])", (c.name or "").lower())]
    named_b = [c for c in clips if re.search(r"(?:^|[\s_\-(])b(?:$|[\s_\-)])", (c.name or "").lower())]
    if len(named_a) == 1 and len(named_b) == 1 and named_a[0].id != named_b[0].id:
        return named_a[0].id, named_b[0].id
    if len(clips) == 2:
        return clips[0].id, clips[1].id
    if len(clips) < 2:
        raise _NoTarget("Для перехода A → B нужны два клипа на шкале. Импортируйте второй трек.")
    pairs = list(itertools.permutations(clips[:4], 2))[:6]
    raise _Need({"text": "Какие клипы — A и B?",
                 "options": [{"label": f"A: {a.name} → B: {b.name}", "plan": build(a.id, b.id)} for a, b in pairs]})


def _music_create(ctx: ParseContext, text: str) -> dict:
    m = re.search(r"(?:примерно|около|приблизительно|в\s+районе|ровно|точно)?\s*(?:на|в|к)\s*" + NUM + SUF + r"\s*(?:секунд\w*|сек|с\b)", text)
    if not m:
        m = re.search(r"(?:в\s+районе|около|примерно)\s*" + NUM, text)
    if not m:
        return _info(ctx, text, "Укажите время перехода, например: «Перейти с A на B примерно на 40-й секунде».")
    at = _num(m.group(1))
    params = {
        "b_on_downbeat": not bool(re.search(r"не\s+на\s+сильн", text)),
        "keep_a_tail": not bool(re.search(r"без\s+хвост|обреж\w*\s+хвост|хвост\w*\s+(?:a|а)?\s*(?:не\s+нуж|убер|обреж)", text)),
        "exact": bool(re.search(r"\b(?:ровно|точно)\b", text)),
    }
    if re.search(r"(?:убер\w*|убрать|срез\w*)\s+(?:бас\w*|низ\w*)\s+(?:у\s+)?(?:a|а)\b", text):
        params["overlap_filter"] = "hp_a"
    cons = []
    if re.search(r"темп", text) and re.search(r"не\s+(?:меня|измен|трога)", text):
        cons.append({"kind": "keep_tempo"})
    if re.search(r"высот\w*|тональн\w*", text) and re.search(r"не\s+(?:меня|измен|трога)", text):
        cons.append({"kind": "keep_pitch"})

    def build(a, b):
        return _plan(ctx, [op("music.create", target={"clip_a_id": a, "clip_b_id": b},
                              time={"coord": "timeline", "unit": "s", "at": at}, params=params,
                              constraints=cons or None)], text)

    a, b = _resolve_ab(ctx, text, build)
    return build(a, b)


def _clause_ops(ctx: ParseContext, text: str, unknown: list[str]) -> list[dict]:
    p = ctx.project
    ops: list[dict] = []
    t = text.strip(" .,;")
    if not t:
        return ops
    # громкость дорожки / клипа
    m = re.search(r"(уменьш\w*|убав\w*|сниз\w*|пониз\w*|тише|увелич\w*|прибав\w*|подним\w*|повыс\w*|громче)", t)
    if m and re.search(r"громкост|уровен|тише|громче", t) and not _groups_curve_context(t):
        sign = -1.0 if re.match(r"(уменьш|убав|сниз|пониз|тише)", m.group(1)) else 1.0
        md = re.search(NUM + r"\s*(?:дб|db|децибел\w*)", t)
        amount = _num(md.group(1)) if md else 3.0
        if re.search(r"клип", t) and ctx.selection.clip_ids:
            ops.append(op("clip.set", target={"clip_id": ctx.selection.clip_ids[0]}, params={"delta_db": sign * amount}))
            return ops

        def build(tid):
            return _plan(ctx, [op("track.set", target={"track_id": tid}, params={"delta_db": sign * amount})], text)

        tid = _target_track(ctx, t, build)
        ops.append(op("track.set", target={"track_id": tid}, params={"delta_db": sign * amount}))
        return ops
    # ограничения темпа/высоты
    if re.search(r"не\s+(?:меня\w*|измен\w*|трога\w*)", t) and re.search(r"темп|высот|тональн", t):
        params = {}
        if re.search(r"темп", t):
            params["keep_tempo"] = True
        if re.search(r"высот|тональн", t):
            params["keep_pitch"] = True
        ops.append(op("constraint.set", params=params))
        return ops
    # продление/сокращение
    m = re.search(r"(продл\w*|удлин\w*|растян\w*|сократ\w*|укорот\w*)", t)
    if m and re.search(r"переход|наложени|свед", t) or (m and ctx.selection.transition_id):
        shorten = bool(re.match(r"(сократ|укорот)", m.group(1)))
        mt = re.search(r"\bдо\s*" + NUM + SUF + SEC, t)
        amount = _amount(t)

        def build_x(xid):
            return _plan(ctx, _extend_ops(p, xid, mt, amount, shorten), text)

        xid = _target_transition(ctx, None, build_x)
        ops += _extend_ops(p, xid, mt, amount, shorten)
        return ops
    # сдвиг перехода
    m = re.search(r"(сдвин\w*|передвин\w*|перемест\w*|подвин\w*|перенес\w*)", t)
    if m and re.search(r"переход", t) and not _groups(t):
        amount = _amount(t)
        if amount is None:
            unknown.append(text + " (не указано, на сколько секунд)")
            return ops
        sign = -1.0 if re.search(r"влево|левее|раньше|назад", t) else 1.0

        def build_s(xid):
            return _plan(ctx, _shift_ops(p, xid, sign * amount), text)

        xid = _target_transition(ctx, None, build_s)
        ops += _shift_ops(p, xid, sign * amount)
        return ops
    # сдвиг точки входа B
    if re.search(r"вход\w*\s*(?:b|б|в)?\b", t) and re.search(r"сдвин\w*|перенес\w*|передвин\w*", t):
        sign = -1.0 if re.search(r"раньше|назад|влево", t) else 1.0
        md = re.search(NUM + r"?\s*(дол\w*|такт\w*)", t)
        if md:
            n = _num(md.group(1)) if md.group(1) else 1.0
            if md.group(2).startswith("такт"):
                n *= 4
            params = {"b_entry_shift_beats": sign * n}
        else:
            amount = _amount(t)
            if amount is None:
                unknown.append(text)
                return ops
            params = {"b_entry_shift": sign * amount}

        def build_b(xid):
            return _plan(ctx, [op("music.set", target={"transition_id": xid}, params=params)], text)

        xid = _target_transition(ctx, "music", build_b)
        ops.append(op("music.set", target={"transition_id": xid}, params=params))
        return ops
    # выбор варианта
    m = re.search(r"вариант\w*\s*(\d)|(коротк\w*|длинн\w*|по\s+фраз\w*|фраз\w*)\s*(?:вариант|сведени)?", t)
    if m and re.search(r"выбер\w*|возьм\w*|вариант|используй|включи", t):
        kinds = ["short", "long", "phrase"]
        if m.group(1):
            i = int(m.group(1)) - 1
            if not 0 <= i < 3:
                unknown.append(text)
                return ops
            kind = kinds[i]
        else:
            w = m.group(2)
            kind = "short" if w.startswith("коротк") else "long" if w.startswith("длинн") else "phrase"

        def build_v(xid):
            return _plan(ctx, [op("music.select_variant", target={"transition_id": xid}, params={"variant": kind})], text)

        xid = _target_transition(ctx, "music", build_v)
        ops.append(op("music.select_variant", target={"transition_id": xid}, params={"variant": kind}))
        return ops
    # защита участка
    if re.search(r"защит\w*|не\s+трога\w*\s+участ|после\s*\d", t) and re.search(r"участ\w*|после|секунд|\d", t):
        rng = _range(t)
        mafter = re.search(r"после\s*" + NUM, t)
        if rng:
            start, end = rng
        elif mafter:
            start, end = _num(mafter.group(1)), None
        else:
            sel = ctx.selection.time_range
            if not sel:
                unknown.append(text + " (не указан участок)")
                return ops
            start, end = sel
        params = {"start": start, "end": end}
        if ctx.selection.track_id and re.search(r"дорожк", t):
            params["track_id"] = ctx.selection.track_id
        ops.append(op("protect.add", params=params))
        return ops
    # mute / solo
    if re.search(r"заглуш\w*|выключ\w*\s+дорожк|mute", t):
        def build_m(tid):
            return _plan(ctx, [op("track.set", target={"track_id": tid}, params={"mute": True})], text)
        tid = _target_track(ctx, t, build_m)
        ops.append(op("track.set", target={"track_id": tid}, params={"mute": True}))
        return ops
    if re.search(r"включ\w*\s+(?:звук\s+)?дорожк|сним\w*\s+mute|разглуш", t):
        def build_u(tid):
            return _plan(ctx, [op("track.set", target={"track_id": tid}, params={"mute": False})], text)
        tid = _target_track(ctx, t, build_u)
        ops.append(op("track.set", target={"track_id": tid}, params={"mute": False}))
        return ops
    if re.search(r"\bсоло\b|\bsolo\b", t):
        val = not re.search(r"сним\w*|выключ\w*|убер\w*", t)

        def build_so(tid):
            return _plan(ctx, [op("track.set", target={"track_id": tid}, params={"solo": val})], text)
        tid = _target_track(ctx, t, build_so)
        ops.append(op("track.set", target={"track_id": tid}, params={"solo": val}))
        return ops
    # кривые параметров внутри перехода
    groups = _groups(t)
    if groups:
        def build_c(xid):
            return _plan(ctx, _curve_ops(p, xid, groups, t) or [], text)

        xid = _target_transition(ctx, "space", build_c)
        cops = _curve_ops(p, xid, groups, t)
        if cops is None:
            unknown.append(text + " (не указано, когда менять параметр)")
            return ops
        ops += cops
        return ops
    unknown.append(text)
    return ops


def _groups_curve_context(t: str) -> bool:
    return bool(_curve_window(t, 4.0)) and not re.search(r"дорожк|трек|клип", t)


def _extend_ops(p: Project, xid: str, m_abs, amount, shorten) -> list[dict]:
    x = p.space_transition(xid)
    if x is not None:
        if m_abs:
            return [op("space.set_timing", target={"transition_id": xid},
                       time={"coord": "timeline", "unit": "s", "end": _num(m_abs.group(1))})]
        a = amount if amount is not None else 1.0
        return [op("space.set_timing", target={"transition_id": xid},
                   time={"coord": "relative", "unit": "s", "end_delta": -a if shorten else a})]
    a = amount if amount is not None else 1.0
    return [op("music.set", target={"transition_id": xid}, params={"extend": -a if shorten else a})]


def _shift_ops(p: Project, xid: str, d: float) -> list[dict]:
    if p.space_transition(xid):
        return [op("space.set_timing", target={"transition_id": xid},
                   time={"coord": "relative", "unit": "s", "start_delta": d, "end_delta": d})]
    return [op("music.set", target={"transition_id": xid}, params={"shift": d})]


def _curve_ops(p: Project, xid: str, groups: list[str], t: str) -> list[dict] | None:
    x = p.space_transition(xid)
    if x is None:
        return None
    win = _curve_window(t, x.length)
    shape = _shape(t)
    if win is None and shape is None:
        return None
    out = []
    for g in groups:
        params = {"group": g}
        if shape:
            params["shape"] = shape
        out.append(op("space.set_curve", target={"transition_id": xid}, time=win, params=params))
    return out


def _artistic(ctx: ParseContext, text: str) -> dict:
    p = ctx.project
    sel = ctx.selection.transition_id
    x = p.space_transition(sel) if sel else (p.space_transitions[0] if len(p.space_transitions) == 1 else None)
    m = p.music_transition(sel) if sel else (p.music_transitions[0] if len(p.music_transitions) == 1 and x is None else None)
    if x is not None:
        ops = []
        if "color_mix" not in x.locked_params and "hp_hz" not in x.locked_params:
            ops.append(op("space.set_curve", target={"transition_id": x.id},
                          time={"coord": "transition_fraction", "unit": "fraction", "start": 0.6, "end": 1.0},
                          params={"group": "color", "shape": "ease_in"}))
        if "reverb_db" not in x.locked_params and x.from_state.room != "none":
            ops.append(op("space.set_state", target={"transition_id": x.id},
                          params={"state": "from", "values": {"reverb_db": min(6.0, x.from_state.values["reverb_db"] + 3.0)}}))
        if not ops:
            return _info(ctx, text, "Все параметры, которые можно сделать контрастнее, зафиксированы.")
        return make_plan(ops, p.revision, origin="local", kind="proposal", artistic=True, text=text,
                         message="Художественное предложение (не гарантирует результат): раскрытие окраски "
                                 "перенесено на последние 40% перехода с медленным началом, комната в начальном "
                                 "состоянии громче на 3 дБ. Прослушайте и примите или отмените.")
    if m is not None:
        kind = "phrase" if (m.variant(m.selected_variant or "") or m.variants[0]).kind != "phrase" else "short"
        v = m.variant(kind)
        return make_plan([op("music.select_variant", target={"transition_id": m.id}, params={"variant": kind})],
                         p.revision, origin="local", kind="proposal", artistic=True, text=text,
                         message=f"Художественное предложение: вариант «{v.name if v else kind}» — более резкая "
                                 f"смена материала. Прослушайте и примите или отмените.")
    return _info(ctx, text, "Выберите переход, для которого нужно художественное предложение.")


def parse(text: str, ctx: ParseContext) -> dict:
    """Текст задания → план asiko.plan/1 (или вопрос/сообщение)."""
    raw = text.strip()
    if not raw:
        return _info(ctx, raw, "Введите задание.")
    t = normalize(raw)
    try:
        if re.match(r"^(?:отмени\w*|откат\w*|undo)\b", t) or re.search(r"верни\s+как\s+было", t):
            return make_plan([], ctx.project.revision, origin="local", kind="undo", text=raw)
        if re.match(r"^(?:повтори\w*|redo)\b", t) or re.search(r"верни\w*\s+отмен", t):
            return make_plan([], ctx.project.revision, origin="local", kind="redo", text=raw)
        if re.search(r"(?:убер\w*|удал\w*|вырез\w*|убрать|без)\s+(?:\w+\s+){0,2}(?:ударн\w*|барабан\w*|вокал\w*|голос\w*|гитар\w*|бас-?гитар\w*)", t):
            return _info(ctx, raw, "Чтобы убрать отдельный инструмент (например, ударные), нужна отдельная дорожка "
                                   "(стем). Из стереомикса точно выделить его нельзя, а разделение источников "
                                   "в этой версии не поддерживается. Можно ослабить диапазон фильтром, но это "
                                   "затронет и другие инструменты.")
        if re.search(r"драматичн|эффектн\w*е|интересн\w*е|кинематографичн|эмоциональн\w*е|красив\w*е|ярче|мощн\w*е", t):
            return _artistic(ctx, raw)
        is_music = bool(re.search(r"(?:^|\s)с\s+(?:трек\w*\s+|клип\w*\s+)?(?:a|а)\s+на\s+(?:трек\w*\s+|клип\w*\s+)?(?:b|б|в)\b|между\s+(?:a|а)\s+и\s+(?:b|б)|с\s+перв\w+\s+(?:трек\w*\s+)?на\s+втор", t))
        rng = _range(t)
        if is_music:
            return _music_create(ctx, t)
        space_words = re.search(r"закадр\w*|в\s+сцене|радио|телефон|издалек|удален\w*\s+источник|за\s+стен|приглуш", t)
        create_words = re.search(r"(?:сделай|создай|добавь|нужен|сделать|создать)\s+переход|переходит|превращ|становится|звучит", t)
        is_edit = re.search(r"продл|удлин|сдвин|перенес|передвин|сократ|укорот|убирай|меняй|раскрыв", t)
        if rng and (create_words or (space_words and not is_edit)):
            return _space_create(ctx, t, rng)
        clauses = [c for c in re.split(r"[.;!\n]+|,?\s+(?:а\s+)?(?:затем|потом|после\s+чего)\s+|\s+и\s+(?=(?:продл|сдвин|перенес|уменьш|увелич|не\s+меня|выбер|защит))", t) if c.strip()]
        unknown: list[str] = []
        ops: list[dict] = []
        for c in clauses:
            ops += _clause_ops(ctx, c, unknown)
        if not ops:
            return _info(ctx, raw, "Локальный обработчик понимает только типовые шаблоны и не распознал задание. "
                                   "Примеры: «" + "», «".join(SUPPORTED_TEMPLATES[:4]) + "». Для свободного текста "
                                   "подключите внешнюю модель в настройках.")
        plan = _plan(ctx, ops, raw)
        if unknown:
            plan["message"] = "Не распознано и не будет выполнено: «" + "», «".join(unknown) + "»."
            plan["kind"] = "proposal"
        return plan
    except _Need as e:
        return make_plan([], ctx.project.revision, origin="local", kind="question", text=raw, question=e.question)
    except _NoTarget as e:
        return _info(ctx, raw, str(e))


def describe_groups() -> str:
    return ", ".join(GROUP_LABELS.values())
