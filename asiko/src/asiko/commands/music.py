"""Переход A → B: варианты, раскладка клипов, оценка конфликтов.

Варианты — детерминированные предложения по разметке долей; они не являются
гарантированно художественно правильными (это явно сказано в интерфейсе).
Темп и высота тона никогда не изменяются: если без этого доли не совпадают,
конфликт показывается, а не устраняется молча.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass

from ..model.automation import seconds
from ..model.project import Clip, MusicParams, MusicTransition, MusicVariant, Project, Source, new_id

GRID_MIN_CONFIDENCE = 0.35
DRIFT_TOLERANCE_S = 0.03
ANTI_CLICK_S = 0.010


@dataclass
class GridView:
    bpm: float | None
    period: float | None
    beats: list[float]        # время шкалы
    downbeats: list[float]    # время шкалы
    phrases: list[float]      # время шкалы
    confidence: float
    phrases_manual: bool
    usable: bool
    src_beats: list[float]
    src_downbeats: list[float]


def grid_view(src: Source | None, clip: Clip) -> GridView:
    if src is None or src.beats is None:
        return GridView(None, None, [], [], [], 0.0, False, False, [], [])
    g = src.beats
    to_tl = lambda s: clip.start + (s - clip.src_in)  # noqa: E731
    beats = [to_tl(b) for b in g.beats]
    downs = [to_tl(b) for b in g.downbeats]
    if src.phrases:
        phrases = [to_tl(b) for b in src.phrases]
        manual = True
    else:
        phrases = [to_tl(b) for b in g.phrase_starts()]
        manual = False
    period = 60.0 / g.bpm if g.bpm > 0 else None
    return GridView(g.bpm, period, beats, downs, phrases, float(g.confidence), manual,
                    g.confidence >= GRID_MIN_CONFIDENCE, list(g.beats), list(g.downbeats))


def _nearest(values: list[float], t: float, lo: float | None = None, hi: float | None = None) -> float | None:
    best = None
    for v in values:
        if lo is not None and v < lo:
            continue
        if hi is not None and v > hi:
            continue
        if best is None or abs(v - t) < abs(best - t):
            best = v
    return best


def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _drift(ga: GridView, gb: GridView, window: float) -> float | None:
    if not (ga.usable and gb.usable and ga.period and gb.period):
        return None
    ratio = ga.period / gb.period
    # учитываем кратные темпы (половинный/двойной)
    for k in (1.0, 2.0, 0.5):
        if abs(ratio * k - 1.0) < 0.04:
            pb = gb.period * k
            n = window / ga.period
            return abs(n * (pb - ga.period))
    n = window / ga.period
    return abs(n * (gb.period - ga.period))


def generate_variants(project: Project, a: Clip, b: Clip, target: float, *, b_on_downbeat: bool = True,
                      keep_a_tail: bool = True, exact: bool = False,
                      overlap_filter: str = "none") -> tuple[list[MusicVariant], list[str]]:
    src_a = project.source(a.source_id)
    src_b = project.source(b.source_id)
    ga = grid_view(src_a, a)
    gb_src = src_b.beats if src_b else None
    common: list[str] = []
    a_src_dur = src_a.duration if src_a else a.src_out
    a_end_available = a.start + (a_src_dur - a.src_in)  # до конца исходника A
    b_src_dur = src_b.duration if src_b else b.src_out

    if target < a.start or target > a_end_available:
        common.append(f"Заданное время {seconds(target)} вне материала A "
                      f"({seconds(a.start)}–{seconds(a_end_available)}).")
    period_a = ga.period if ga.usable and ga.period else 0.5
    if not ga.usable:
        common.append("Разметка долей A отсутствует или ненадёжна — точка перехода выбрана по времени. "
                      "Запустите анализ или задайте сетку вручную.")

    # Точка входа B (время исходника B)
    b_entry = b.src_in
    b_grid_usable = gb_src is not None and gb_src.confidence >= GRID_MIN_CONFIDENCE
    if b_on_downbeat:
        if gb_src is not None and gb_src.downbeats:
            cands = [d for d in gb_src.downbeats if d >= b.src_in - 1e-6]
            b_entry = cands[0] if cands else gb_src.downbeats[0]
            if not b_grid_usable:
                common.append(f"Сильная доля B найдена с низкой уверенностью ({gb_src.confidence:.0%}) — "
                              f"проверьте точку входа вручную.")
        else:
            common.append("Для B нет разметки долей: вход по началу клипа. Запустите анализ, "
                          "чтобы B вошёл на сильную долю.")
    gb = grid_view(src_b, Clip(id="tmp", source_id=b.source_id, start=0.0, src_in=0.0, src_out=b_src_dur))
    b_pre_avail = b_entry  # материал B до точки входа (с)

    # Точка переключения для сведений: ближайшая сильная доля A
    if exact or not ga.usable:
        t_s = target
        if exact and ga.usable:
            nd = _nearest(ga.downbeats, target)
            if nd is not None and abs(nd - target) > DRIFT_TOLERANCE_S:
                common.append(f"Время задано точно: {seconds(target)}. Ближайшая сильная доля A — "
                              f"{seconds(nd)} (расхождение {abs(nd - target):.2f} с).")
    else:
        bar = period_a * 4
        nd = _nearest(ga.downbeats, target, target - bar, target + bar)
        t_s = nd if nd is not None else (_nearest(ga.beats, target) or target)

    variants: list[MusicVariant] = []
    conf = min(ga.confidence if ga.usable else 0.0, gb.confidence if gb.usable else 0.0) if b_on_downbeat else (ga.confidence if ga.usable else 0.0)

    def tail_len(want: float, start: float) -> tuple[float, list[str]]:
        avail = a_end_available - start
        notes = []
        if avail < want - 1e-6:
            notes.append(f"Хвост A: в исходнике после {seconds(start)} есть только {max(avail, 0):.2f} с — "
                         f"продолжение не выдумывается.")
        return max(min(want, avail), ANTI_CLICK_S), notes

    # ---------------- 1. Короткое сведение
    d = _clamp(period_a, 0.3, 1.0)
    notes, conflicts = [], []
    pre = min(d / 2, b_pre_avail)
    b_fade_start = t_s - pre
    b_fade_len = (t_s + d / 2) - b_fade_start
    if pre < d / 2 - 1e-6:
        notes.append("У B нет материала до сильной доли — нарастание начинается с самой доли.")
    a_fade_start = t_s - d / 2
    a_len = d
    if keep_a_tail:
        a_len, tn = tail_len(max(d, 2.0), a_fade_start)
        notes += tn
        notes.append(f"Хвост A сохранён: затухание {a_len:.2f} с из исходной записи.")
    dr = _drift(ga, gb, max(d, a_len))
    if dr is not None and dr > DRIFT_TOLERANCE_S:
        msg = (f"Темп A {ga.bpm:.1f} BPM, B {gb.bpm:.1f} BPM; изменение темпа запрещено — "
               f"за время наложения доли разойдутся на {dr:.2f} с.")
        (conflicts if project.constraints.get("keep_tempo") else notes).append(msg)
    variants.append(MusicVariant(
        id=new_id("var"), kind="short", name="Короткое сведение",
        params=MusicParams(switch_time=t_s, b_entry_src=b_entry, a_fade_start=a_fade_start, a_fade_len=a_len,
                           b_fade_start=b_fade_start, b_fade_len=b_fade_len, curve="equal_power",
                           a_tail="source" if keep_a_tail else "cut", overlap_filter=overlap_filter),
        description=f"B входит в {seconds(t_s)}; наложение ≈{d:.2f} с, equal-power.",
        notes=notes, conflicts=conflicts, confidence=conf))

    # ---------------- 2. Длинное сведение
    L = _clamp(period_a * 8, 3.0, 12.0)
    notes, conflicts = [], []
    a_len, tn = tail_len(L, t_s)
    notes += tn
    b_len = L
    if b_pre_avail >= L / 2 - 1e-6:
        b_start = t_s - L / 2
        a_start = t_s - L / 2
        a_len, tn = tail_len(L, a_start)
        notes = tn
        desc = f"Наложение {L:.1f} с, центр — сильная доля B в {seconds(t_s)}."
    else:
        b_start = t_s
        a_start = t_s
        desc = f"B входит на сильную долю в {seconds(t_s)} и нарастает {L:.1f} с; A затухает за то же время."
    dr = _drift(ga, gb, L)
    if dr is not None and dr > DRIFT_TOLERANCE_S:
        msg = (f"Темп A {ga.bpm:.1f} BPM, B {gb.bpm:.1f} BPM; при запрете изменения темпа "
               f"за {L:.1f} с наложения доли разойдутся на {dr:.2f} с. Возможны варианты: "
               f"короче наложение или «завершение фразы».")
        (conflicts if project.constraints.get("keep_tempo") else notes).append(msg)
    if dr is None:
        notes.append("Совпадение долей в длинном наложении не проверено: нет надёжной разметки обоих треков.")
    variants.append(MusicVariant(
        id=new_id("var"), kind="long", name="Длинное сведение",
        params=MusicParams(switch_time=t_s, b_entry_src=b_entry, a_fade_start=a_start, a_fade_len=a_len,
                           b_fade_start=b_start, b_fade_len=b_len, curve="equal_power", a_tail="source",
                           overlap_filter=overlap_filter if overlap_filter != "none" else "hp_a"),
        description=desc + " В зоне наложения у A убирается низ (можно отключить).",
        notes=notes, conflicts=conflicts, confidence=conf))

    # ---------------- 3. Завершение фразы и новый вход
    notes, conflicts = [], []
    if ga.phrases and not exact:
        p_t = _nearest([p for p in ga.phrases if p > a.start + 0.5], target)
    else:
        p_t = None
    half = False
    if not ga.phrases_manual and ga.usable and ga.downbeats and not exact:
        bar = period_a * 4
        if p_t is None or abs(p_t - target) > 2 * bar:
            p4 = _nearest([d for d in ga.downbeats[::4] if d > a.start + 0.5], target)
            if p4 is not None and (p_t is None or abs(p4 - target) < abs(p_t - target)):
                p_t = p4
                half = True
    if p_t is None:
        p_t = t_s
        notes.append("Границы фраз A не размечены — использована точка переключения сведений.")
    elif half:
        notes.append("Граница 4-тактовой фразы оценена по сетке — проверьте и при необходимости разметьте "
                     "фразы вручную.")
    elif not ga.phrases_manual:
        notes.append(f"Граница фразы оценена по сетке ({src_a.beats.phrase_bars if src_a and src_a.beats else 8} "
                     f"тактов) — проверьте и при необходимости разметьте фразы вручную.")
    if abs(p_t - target) > 0.05:
        notes.append(f"Вход смещён от заданных {seconds(target)} на {p_t - target:+.2f} с (граница фразы).")
    if keep_a_tail:
        a_len, tn = tail_len(1.5, p_t)
        notes += tn
        a_tail = "source"
        notes.append("Хвост A — продолжение исходной записи после конца фразы с затуханием. "
                     "Можно заменить на хвост реверберации (добавленный эффект).")
    else:
        a_len = 0.03
        a_tail = "cut"
    variants.append(MusicVariant(
        id=new_id("var"), kind="phrase", name="Завершение фразы и новый вход",
        params=MusicParams(switch_time=p_t, b_entry_src=b_entry, a_fade_start=p_t, a_fade_len=a_len,
                           b_fade_start=p_t, b_fade_len=ANTI_CLICK_S, curve="equal_power", a_tail=a_tail,
                           overlap_filter="none"),
        description=f"A доигрывает фразу до {seconds(p_t)}, B входит сразу с сильной доли.",
        notes=notes, conflicts=conflicts, confidence=conf))

    return variants, common


def apply_layout(project: Project, mt: MusicTransition, allow_locked: set | None = None) -> list[str]:
    """Расставляет клипы A и B согласно активным параметрам. Возвращает заметки.

    Бросает ValueError, если требуется изменить зафиксированный клип.
    """
    allow_locked = allow_locked or set()
    a = project.clip(mt.clip_a_id)
    b = project.clip(mt.clip_b_id)
    if a is None or b is None:
        raise ValueError("Клип A или B не найден")
    src_a = project.source(a.source_id)
    src_b = project.source(b.source_id)
    p = mt.params
    notes = []
    # B
    b_src_dur = src_b.duration if src_b else b.src_out
    want_in = p.b_entry_src - (p.switch_time - p.b_fade_start)
    new_b_in = max(0.0, want_in)
    if want_in < -1e-6:
        notes.append("У B нет материала до начала нарастания: B начинается позже, с начала файла.")
    new_b_start = p.switch_time - (p.b_entry_src - new_b_in)
    new_b_out = mt.orig_b.get("src_out", b.src_out)
    new_b_out = min(max(new_b_out, new_b_in + 0.05), b_src_dur)
    # A
    a_src_dur = src_a.duration if src_a else a.src_out
    need_out = a.to_src(p.a_fade_end) + ANTI_CLICK_S
    new_a_out = min(a_src_dur, max(need_out, a.src_in + 0.05))
    if need_out > a_src_dur + 1e-6:
        notes.append(f"A заканчивается раньше конца затухания (не хватает {need_out - a_src_dur:.2f} с исходника).")
    changes_b = (abs(new_b_start - b.start) > 1e-9 or abs(new_b_in - b.src_in) > 1e-9
                 or abs(new_b_out - b.src_out) > 1e-9)
    changes_a = abs(new_a_out - a.src_out) > 1e-9
    if changes_b and b.locked and b.id not in allow_locked:
        raise ValueError(f"Клип B «{b.name}» зафиксирован, а выбранный вариант требует сдвинуть его. "
                         f"Снимите фиксацию или выберите другой вариант.")
    if changes_a and a.locked and a.id not in allow_locked:
        raise ValueError(f"Клип A «{a.name}» зафиксирован, а выбранный вариант требует изменить его конец.")
    if new_b_start < 0:
        raise ValueError("Клип B оказался бы до начала проекта — сдвиньте точку перехода.")
    b.start, b.src_in, b.src_out = new_b_start, new_b_in, new_b_out
    a.src_out = new_a_out
    return notes


def evaluate(project: Project, mt: MusicTransition) -> tuple[list[str], list[str]]:
    """Конфликты и заметки для активных параметров (после ручных правок тоже)."""
    a = project.clip(mt.clip_a_id)
    b = project.clip(mt.clip_b_id)
    conflicts: list[str] = []
    notes: list[str] = []
    if a is None or b is None:
        return ["Клип A или B удалён"], []
    ga = grid_view(project.source(a.source_id), a)
    gb = grid_view(project.source(b.source_id), b)
    p = mt.params
    keep_tempo = project.constraints.get("keep_tempo", False)
    if mt.b_on_downbeat and gb.usable and gb.src_downbeats:
        nd = min(gb.src_downbeats, key=lambda d: abs(d - p.b_entry_src))
        if abs(nd - p.b_entry_src) > DRIFT_TOLERANCE_S:
            conflicts.append(f"Точка входа B ({seconds(p.b_entry_src)} в исходнике) не на сильной доле: "
                             f"ближайшая — {seconds(nd)}.")
    if ga.usable and ga.downbeats and mt.b_on_downbeat:
        nd = min(ga.downbeats, key=lambda d: abs(d - p.switch_time))
        if abs(nd - p.switch_time) > DRIFT_TOLERANCE_S:
            notes.append(f"Вход B не совпадает с сильной долей A (расхождение {abs(nd - p.switch_time):.2f} с).")
    dr = _drift(ga, gb, max(0.0, min(p.a_fade_end, b.end) - p.switch_time))
    if dr is not None and dr > DRIFT_TOLERANCE_S and p.a_fade_end - p.switch_time > 0.25:
        msg = (f"Темп A {ga.bpm:.1f} и B {gb.bpm:.1f} BPM различаются: к концу наложения доли разойдутся на "
               f"{dr:.2f} с (темп не изменяется).")
        (conflicts if keep_tempo else notes).append(msg)
    if p.a_tail == "effect":
        notes.append("Хвост A — добавленная реверберация (эффект), а не продолжение записи.")
    src_a = project.source(a.source_id)
    if src_a and a.to_src(p.a_fade_end) > src_a.duration + 1e-3:
        notes.append("Затухание A длиннее доступного материала исходника — хвост обрезан концом файла.")
    if project.constraints.get("keep_pitch"):
        notes.append("Высота тона не изменяется.")
    return conflicts, notes


def variant_summary(v: MusicVariant) -> str:
    p = v.params
    return (f"{v.name}: вход B {seconds(p.switch_time)}, A затухает {seconds(p.a_fade_start)}–"
            f"{seconds(p.a_fade_end)}, B нарастает {seconds(p.b_fade_start)}–{seconds(p.b_fade_end)}")


def copy_params(p: MusicParams) -> MusicParams:
    return copy.deepcopy(p)
