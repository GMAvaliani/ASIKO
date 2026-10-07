"""Обработчики операций плана. Работают на копии проекта внутри транзакции."""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field

from ..model.automation import seconds
from ..model.presets import (
    BUILTIN_PRESETS,
    GROUP_LABELS,
    PARAM_GROUPS,
    PARAM_INFO,
    ROOMS,
    SPACE_PARAMS,
    format_param,
    get_preset,
)
from ..model.project import (
    BeatGrid,
    Clip,
    FrozenRegion,
    MusicTransition,
    ParamCurve,
    Project,
    Protection,
    Source,
    SpaceState,
    SpaceTransition,
    Track,
    VideoRef,
    default_curves,
    new_id,
)
from . import music as music_mod

MIN_TRANSITION_S = 0.05


class OpError(Exception):
    pass


@dataclass
class Ctx:
    before: Project
    work: Project
    origin: str = "ui"
    user_presets: dict = field(default_factory=dict)
    summary: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    created: list[str] = field(default_factory=list)
    select: str | None = None
    unlocked_params: dict[str, set] = field(default_factory=dict)
    unapproved: set = field(default_factory=set)
    unlocked_clips: set = field(default_factory=set)
    removed_protections: set = field(default_factory=set)
    preserved_note: bool = False


def _fmt_range(a: float, b: float | None) -> str:
    return f"{seconds(a)}–{seconds(b) if b is not None else 'конец'}"


def _space(ctx: Ctx, op: dict) -> SpaceTransition:
    xid = op["target"]["transition_id"]
    x = ctx.work.space_transition(xid)
    if x is None:
        if ctx.work.music_transition(xid):
            raise OpError("Операция относится к переходу «в сцене → закадровый», а выбран переход A → B.")
        raise OpError(f"Переход {xid} не найден (возможно, удалён после формирования команды).")
    return x


def _music(ctx: Ctx, op: dict) -> MusicTransition:
    xid = op["target"]["transition_id"]
    m = ctx.work.music_transition(xid)
    if m is None:
        if ctx.work.space_transition(xid):
            raise OpError("Операция относится к переходу A → B, а выбран переход «в сцене → закадровый».")
        raise OpError(f"Переход {xid} не найден (возможно, удалён после формирования команды).")
    return m


def _track(ctx: Ctx, tid: str) -> Track:
    t = ctx.work.track(tid)
    if t is None:
        raise OpError(f"Дорожка {tid} не найдена.")
    return t


def _clip(ctx: Ctx, cid: str) -> Clip:
    c = ctx.work.clip(cid)
    if c is None:
        raise OpError(f"Клип {cid} не найден.")
    return c


def _check_overlap(ctx: Ctx, x_id: str | None, track_id: str, start: float, end: float) -> None:
    for o in ctx.work.transitions_on_track(track_id):
        if o.id == x_id:
            continue
        if o.start < end - 1e-9 and start < o.end - 1e-9:
            raise OpError(f"Переход пересекается с «{o.name}» ({_fmt_range(o.start, o.end)}) на той же дорожке.")


def _apply_protect_constraints(ctx: Ctx, op: dict, track_id: str | None, end: float | None) -> None:
    for c in op.get("constraints", []) or []:
        if c["kind"] == "protect_after":
            t = float(c["time"])
            if end is not None and t < end - 1e-6:
                raise OpError(f"Участок после {seconds(t)} должен остаться без изменений, а переход "
                              f"заканчивается в {seconds(end)}. Завершите переход к {seconds(t)}.")
            p = Protection(id=new_id("prt"), start=t, end=None, track_id=track_id,
                           note=f"После {seconds(t)} не менять")
            ctx.work.protections.append(p)
            ctx.summary.append(f"Участок после {seconds(t)} защищён от изменения настроек.")
        elif c["kind"] == "protect_range":
            s, e = float(c["start"]), c.get("end")
            if e is not None and e <= s:
                raise OpError("Конец защищаемого участка должен быть позже начала.")
            p = Protection(id=new_id("prt"), start=s, end=e, track_id=track_id, note="")
            ctx.work.protections.append(p)
            ctx.summary.append(f"Участок {_fmt_range(s, e)} защищён от изменения настроек.")


# --------------------------------------------------------------------------- space


def _neutral_preset(key: str) -> bool:
    return key == "offscreen"


def h_space_create(ctx: Ctx, op: dict) -> None:
    w = ctx.work
    tid = op["target"]["track_id"]
    track = _track(ctx, tid)
    start, end = float(op["time"]["start"]), float(op["time"]["end"])
    if end - start < MIN_TRANSITION_S:
        raise OpError("Конец перехода должен быть позже начала (минимум 50 мс).")
    p = op.get("params", {}) or {}
    fp = p.get("from_preset", "radio_room")
    tp = p.get("to_preset", "offscreen")
    for key in (fp, tp):
        if get_preset(key, ctx.user_presets) is None:
            raise OpError(f"Неизвестный пресет «{key}».")
    fs = SpaceState.from_preset(fp, ctx.user_presets, p.get("from_values"))
    ts = SpaceState.from_preset(tp, ctx.user_presets, p.get("to_values"))
    if "from_room" in p:
        fs.room = p["from_room"]
    if "to_room" in p:
        ts.room = p["to_room"]
    _check_overlap(ctx, None, tid, start, end)
    prev = [o for o in w.transitions_on_track(tid) if o.end <= start + 1e-9]
    if prev:
        pv = prev[-1]
        if pv.to_state.values != fs.values or pv.to_state.room != fs.room:
            ctx.warnings.append(f"Начальное состояние не совпадает с конечным состоянием перехода «{pv.name}»: "
                                f"в {seconds(start)} будет сглаженный (10 мс) скачок параметров.")
    curves = default_curves()
    for k, c in (p.get("curves") or {}).items():
        curves[k] = ParamCurve(start=float(c.get("start", 0.0)), end=float(c.get("end", 1.0)),
                               shape=c.get("shape", "smooth"))
        if curves[k].end <= curves[k].start:
            raise OpError(f"Кривая «{PARAM_INFO[k]['label']}»: конец должен быть позже начала.")
    slope = p.get("filter_slope")
    if slope is None:
        src_key = fp if not _neutral_preset(fp) else tp
        slope = (get_preset(src_key, ctx.user_presets) or {}).get("slope", 12)
    anchor = None
    for c in sorted(track.clips, key=lambda c: c.start):
        if c.start - 1e-9 <= start < c.end:
            anchor = c.id
            break
    if anchor is None and track.clips:
        anchor = min(track.clips, key=lambda c: abs(c.start - start)).id
    x = SpaceTransition(id=new_id("spc"), track_id=tid, start=start, end=end, from_state=fs, to_state=ts,
                        curves=curves, filter_slope=int(slope), anchor_clip_id=anchor,
                        name=p.get("name") or f"{fs.name} → {ts.name}")
    w.space_transitions.append(x)
    ctx.created.append(x.id)
    ctx.select = x.id
    ctx.summary.append(f"Переход «{x.name}» на дорожке «{track.name}»: {_fmt_range(start, end)}.")
    if not any(c.start - 1e-9 <= start and c.end + 1e-9 >= end for c in track.clips):
        ctx.warnings.append("Переход выходит за границы клипов дорожки — проверьте, что звук есть на всём участке.")
    _apply_protect_constraints(ctx, op, tid, end)


def _locked(ctx: Ctx, x: SpaceTransition) -> set:
    return set(x.locked_params) - ctx.unlocked_params.get(x.id, set())


def _not_approved(ctx: Ctx, x) -> None:
    if x.approved and x.id not in ctx.unapproved:
        raise OpError(f"Переход «{x.name}» утверждён. Снимите утверждение, чтобы его изменить.")


def h_space_set_timing(ctx: Ctx, op: dict) -> None:
    x = _space(ctx, op)
    _not_approved(ctx, x)
    tm = op["time"]
    old = (x.start, x.end)
    if tm["coord"] == "timeline":
        ns = float(tm.get("start", x.start))
        ne = float(tm.get("end", x.end))
    else:
        ns = x.start + float(tm.get("start_delta", 0.0))
        ne = x.end + float(tm.get("end_delta", 0.0))
    if ns < 0:
        raise OpError("Переход не может начинаться раньше начала проекта.")
    if ne - ns < MIN_TRANSITION_S:
        raise OpError("Конец перехода должен быть позже начала (минимум 50 мс).")
    _check_overlap(ctx, x.id, x.track_id, ns, ne)
    locked = _locked(ctx, x)
    keep_abs = {p: x.curve_abs(p) for p in locked}
    x.start, x.end = ns, ne
    for p, (a0, a1) in keep_abs.items():
        f0 = (a0 - ns) / (ne - ns)
        f1 = (a1 - ns) / (ne - ns)
        if f0 < -1e-9 or f1 > 1 + 1e-9:
            raise OpError(f"Параметр «{PARAM_INFO[p]['label']}» зафиксирован на {_fmt_range(a0, a1)}; "
                          f"переход {_fmt_range(ns, ne)} его не вмещает.")
        x.curves[p].start = min(max(f0, 0.0), 1.0)
        x.curves[p].end = min(max(f1, 0.0), 1.0)
    d_start, d_end = ns - old[0], ne - old[1]
    if abs(d_start - d_end) < 1e-9 and abs(d_start) > 1e-9:
        how = f"сдвинут на {d_start:+.2f} с"
    elif abs(d_start) < 1e-9 and d_end > 0:
        how = f"продлён на {d_end:.2f} с"
    elif abs(d_start) < 1e-9 and d_end < 0:
        how = f"сокращён на {-d_end:.2f} с"
    else:
        how = "границы изменены"
    ctx.select = x.id
    ctx.summary.append(f"Переход: {_fmt_range(*old)} → {_fmt_range(ns, ne)} ({how}).")
    if locked:
        names = ", ".join(PARAM_INFO[p]["label"].lower() for p in sorted(locked))
        ctx.summary.append(f"Зафиксированные параметры ({names}) сохранили своё положение во времени.")
    ctx.summary.append("Моменты изменения остальных параметров масштабированы пропорционально длине перехода.")
    _apply_protect_constraints(ctx, op, x.track_id, ne)


def _curve_params(op: dict) -> tuple[list[str], str | None]:
    p = op.get("params", {}) or {}
    if "params" in p:
        names = list(p["params"])
        for n in names:
            if n not in SPACE_PARAMS:
                raise OpError(f"Неизвестный параметр «{n}».")
        return names, None
    if "group" in p:
        return list(PARAM_GROUPS[p["group"]]), p["group"]
    raise OpError("Не указано, какие параметры менять (params или group).")


def _describe_window(x: SpaceTransition, f0: float, f1: float) -> str:
    L = x.length
    a0, a1 = x.start + f0 * L, x.start + f1 * L
    if f1 >= 1 - 1e-6 and f0 > 1e-6:
        rel = f"последние {(1 - f0) * L:.2f}".rstrip("0").rstrip(".") + " с перехода"
    elif f0 <= 1e-6 and f1 < 1 - 1e-6:
        rel = f"первые {f1 * L:.2f}".rstrip("0").rstrip(".") + " с перехода"
    elif f0 <= 1e-6 and f1 >= 1 - 1e-6:
        rel = "вся длина перехода"
    else:
        rel = f"{f0 * L:.2f}–{f1 * L:.2f} с от начала перехода"
    return f"{_fmt_range(a0, a1)} ({rel})"


def h_space_set_curve(ctx: Ctx, op: dict) -> None:
    x = _space(ctx, op)
    _not_approved(ctx, x)
    names, group = _curve_params(op)
    locked = _locked(ctx, x) & set(names)
    if locked:
        raise OpError("Зафиксированы параметры: " + ", ".join(PARAM_INFO[p]["label"] for p in sorted(locked))
                      + ". Снимите фиксацию, чтобы менять их кривые.")
    p = op.get("params", {}) or {}
    tm = op.get("time")
    if tm is None and "shape" not in p:
        raise OpError("Не указано, что изменить: время (time) или форма кривой (shape).")
    L = x.length
    f0 = f1 = None
    if tm is not None:
        coord = tm["coord"]
        unit = tm.get("unit", "fraction" if coord == "transition_fraction" else "s")

        def conv(v):
            if v is None:
                return None
            v = float(v)
            if coord == "transition_fraction" or unit == "fraction":
                return v
            if coord == "transition":
                return v / L
            if coord == "transition_end":
                return 1.0 + v / L
            return (v - x.start) / L  # timeline

        f0, f1 = conv(tm.get("start")), conv(tm.get("end"))
    changed = []
    for n in names:
        c = x.curves[n]
        s = c.start if f0 is None else f0
        e = c.end if f1 is None else f1
        if s < -1e-6 or e > 1 + 1e-6:
            raise OpError(f"Интервал изменения выходит за границы перехода {_fmt_range(x.start, x.end)}.")
        s, e = min(max(s, 0.0), 1.0), min(max(e, 0.0), 1.0)
        if (e - s) * L < 0.01 - 1e-9:
            raise OpError("Интервал изменения параметра слишком короткий (минимум 10 мс).")
        c.start, c.end = s, e
        if "shape" in p:
            c.shape = p["shape"]
        changed.append(n)
    label = GROUP_LABELS.get(group) if group else ", ".join(PARAM_INFO[n]["label"] for n in changed)
    c0 = x.curves[changed[0]]
    ctx.select = x.id
    ctx.summary.append(f"Переход: {_fmt_range(x.start, x.end)}. {label}: {_describe_window(x, c0.start, c0.end)}"
                       + (f", форма «{p['shape']}»" if "shape" in p else "") + ".")
    ctx.summary.append("Длительность перехода, положение клипов и остальные параметры сохранены.")


def h_space_set_state(ctx: Ctx, op: dict) -> None:
    x = _space(ctx, op)
    _not_approved(ctx, x)
    p = op.get("params", {}) or {}
    which = p["state"]
    st = x.from_state if which == "from" else x.to_state
    locked = _locked(ctx, x)
    before_vals = dict(st.values)
    before_room = st.room
    if "preset" in p:
        if get_preset(p["preset"], ctx.user_presets) is None:
            raise OpError(f"Неизвестный пресет «{p['preset']}».")
        new = SpaceState.from_preset(p["preset"], ctx.user_presets)
        for k in locked:
            new.values[k] = st.values[k]
        if "reverb_db" in locked:
            new.room = st.room
        st.preset, st.name, st.values, st.room = new.preset, new.name, new.values, new.room
        if locked:
            ctx.warnings.append("Зафиксированные параметры не изменены при смене пресета.")
    for k, v in (p.get("values") or {}).items():
        if k in locked:
            raise OpError(f"Параметр «{PARAM_INFO[k]['label']}» зафиксирован.")
        st.values[k] = float(v)
    if "room" in p:
        if "reverb_db" in locked:
            raise OpError("Комнатная обработка зафиксирована.")
        st.room = p["room"]
    diffs = [f"{PARAM_INFO[k]['label'].lower()} {format_param(k, before_vals[k])} → {format_param(k, st.values[k])}"
             for k in SPACE_PARAMS if abs(before_vals[k] - st.values[k]) > 1e-9]
    if st.room != before_room:
        diffs.append(f"комната {ROOMS[before_room]} → {ROOMS[st.room]}")
    name = "Начальное" if which == "from" else "Конечное"
    ctx.select = x.id
    if diffs:
        ctx.summary.append(f"{name} состояние перехода «{x.name}»: " + "; ".join(diffs) + ".")
        ctx.summary.append("Время перехода и остальные параметры сохранены.")
    else:
        ctx.summary.append(f"{name} состояние перехода не изменилось.")
    x.name = f"{x.from_state.name} → {x.to_state.name}"


def h_space_set_slope(ctx: Ctx, op: dict) -> None:
    x = _space(ctx, op)
    _not_approved(ctx, x)
    if _locked(ctx, x) & {"hp_hz", "lp_hz"}:
        raise OpError("Параметры фильтра зафиксированы.")
    x.filter_slope = int(op["params"]["filter_slope"])
    ctx.summary.append(f"Крутизна фильтров перехода: {x.filter_slope} дБ/окт.")


def h_space_delete(ctx: Ctx, op: dict) -> None:
    x = _space(ctx, op)
    _not_approved(ctx, x)
    ctx.work.space_transitions = [o for o in ctx.work.space_transitions if o.id != x.id]
    ctx.summary.append(f"Переход «{x.name}» ({_fmt_range(x.start, x.end)}) удалён.")


def h_lock_params(ctx: Ctx, op: dict) -> None:
    xid = op["target"]["transition_id"]
    x = ctx.work.space_transition(xid)
    if x is None:
        raise OpError("Фиксация параметров доступна для переходов «в сцене → закадровый».")
    names = op["params"]["params"]
    for n in names:
        if n not in SPACE_PARAMS:
            if n in PARAM_GROUPS:
                continue
            raise OpError(f"Неизвестный параметр «{n}».")
    expanded = []
    for n in names:
        expanded += list(PARAM_GROUPS.get(n, (n,)))
    if op["params"]["locked"]:
        x.locked_params = sorted(set(x.locked_params) | set(expanded))
        ctx.summary.append("Зафиксированы параметры: " + ", ".join(PARAM_INFO[n]["label"] for n in expanded) + ".")
    else:
        x.locked_params = sorted(set(x.locked_params) - set(expanded))
        ctx.unlocked_params.setdefault(x.id, set()).update(expanded)
        ctx.summary.append("Снята фиксация параметров: "
                           + ", ".join(PARAM_INFO[n]["label"] for n in expanded) + ".")
    ctx.select = x.id


def h_approve(ctx: Ctx, op: dict) -> None:
    xid = op["target"]["transition_id"]
    x = ctx.work.any_transition(xid)
    if x is None:
        raise OpError(f"Переход {xid} не найден.")
    val = bool(op["params"]["approved"])
    x.approved = val
    if not val:
        ctx.unapproved.add(x.id)
    ctx.select = x.id
    ctx.summary.append(f"Переход «{x.name}» " + ("утверждён и защищён от изменений." if val else "снят с утверждения."))


# --------------------------------------------------------------------------- music


def _allow_locked(ctx: Ctx) -> set:
    return set(ctx.unlocked_clips)


def _refresh_music(ctx: Ctx, mt: MusicTransition) -> None:
    try:
        lay = music_mod.apply_layout(ctx.work, mt, _allow_locked(ctx))
    except ValueError as e:
        raise OpError(str(e))
    conflicts, notes = music_mod.evaluate(ctx.work, mt)
    mt.conflicts = conflicts
    mt.notes = lay + notes
    for c in conflicts:
        ctx.warnings.append("Конфликт: " + c)


def h_music_create(ctx: Ctx, op: dict) -> None:
    w = ctx.work
    a = _clip(ctx, op["target"]["clip_a_id"])
    b = _clip(ctx, op["target"]["clip_b_id"])
    if a.id == b.id:
        raise OpError("A и B должны быть разными клипами.")
    for m in w.music_transitions:
        if {m.clip_a_id, m.clip_b_id} == {a.id, b.id}:
            raise OpError(f"Переход между этими клипами уже есть («{m.name}») — измените его.")
    p = op.get("params", {}) or {}
    for c in op.get("constraints", []) or []:
        if c["kind"] in ("keep_tempo", "keep_pitch"):
            w.constraints[c["kind"]] = True
    ta, tb = w.clip_track(a.id), w.clip_track(b.id)
    if ta.id == tb.id:
        if b.locked and b.id not in ctx.unlocked_clips:
            raise OpError("A и B на одной дорожке, а клип B зафиксирован — его нельзя перенести на новую дорожку.")
        nt = Track(id=new_id("trk"), name=f"{tb.name} (B)")
        tb.clips = [c for c in tb.clips if c.id != b.id]
        nt.clips.append(b)
        idx = w.tracks.index(tb)
        w.tracks.insert(idx + 1, nt)
        ctx.warnings.append(f"Клип B перенесён на новую дорожку «{nt.name}», чтобы A и B могли звучать одновременно.")
    at = float(op["time"]["at"])
    variants, common = music_mod.generate_variants(
        w, a, b, at, b_on_downbeat=p.get("b_on_downbeat", True), keep_a_tail=p.get("keep_a_tail", True),
        exact=p.get("exact", False), overlap_filter=p.get("overlap_filter", "none"))
    sel = p.get("select", "short")
    v = next(vv for vv in variants if vv.kind == sel)
    mt = MusicTransition(id=new_id("mus"), clip_a_id=a.id, clip_b_id=b.id, target_time=at,
                         params=copy.deepcopy(v.params), variants=variants, selected_variant=v.id,
                         b_on_downbeat=p.get("b_on_downbeat", True), keep_a_tail=p.get("keep_a_tail", True),
                         exact_time=p.get("exact", False),
                         name=p.get("name") or f"{a.name or 'A'} → {b.name or 'B'}",
                         orig_a={"src_out": a.src_out}, orig_b={"start": b.start, "src_in": b.src_in, "src_out": b.src_out})
    w.music_transitions.append(mt)
    _refresh_music(ctx, mt)
    mt.notes = common + mt.notes
    ctx.created.append(mt.id)
    ctx.select = mt.id
    ctx.summary.append(f"Переход A → B около {seconds(at)}: предложено {len(variants)} варианта "
                       f"(не гарантируют художественно верный результат).")
    ctx.summary.append(f"Выбран «{v.name}»: {v.description}")
    if w.constraints.get("keep_tempo") or w.constraints.get("keep_pitch"):
        ctx.summary.append("Темп и высота тона не изменяются.")
    for n in common:
        ctx.warnings.append(n)


def h_music_select(ctx: Ctx, op: dict) -> None:
    mt = _music(ctx, op)
    _not_approved(ctx, mt)
    v = mt.variant(op["params"]["variant"])
    if v is None:
        raise OpError(f"Вариант «{op['params']['variant']}» не найден.")
    mt.params = copy.deepcopy(v.params)
    mt.selected_variant = v.id
    mt.modified = False
    _refresh_music(ctx, mt)
    ctx.select = mt.id
    ctx.summary.append(f"Выбран вариант «{v.name}»: {v.description}")


def h_music_set(ctx: Ctx, op: dict) -> None:
    mt = _music(ctx, op)
    _not_approved(ctx, mt)
    p = op.get("params", {}) or {}
    if not p:
        raise OpError("Не указано, что изменить в переходе.")
    mp = mt.params
    old = copy.deepcopy(mp)
    for k in ("switch_time", "b_entry_src", "a_fade_start", "a_fade_len", "b_fade_start", "b_fade_len",
              "curve", "a_tail", "overlap_filter"):
        if k in p:
            setattr(mp, k, p[k])
    if "shift" in p:
        d = float(p["shift"])
        mp.switch_time += d
        mp.a_fade_start += d
        mp.b_fade_start += d
    if "extend" in p:
        d = float(p["extend"])
        pre_roll = mp.b_fade_end <= mp.switch_time + 1e-6
        mp.a_fade_len += d
        mp.b_fade_len += d
        if pre_roll:
            mp.b_fade_start -= d
    if "b_entry_shift" in p:
        mp.b_entry_src += float(p["b_entry_shift"])
    if "b_entry_shift_beats" in p:
        b = ctx.work.clip(mt.clip_b_id)
        src = ctx.work.source(b.source_id) if b else None
        if not src or not src.beats or src.beats.bpm <= 0:
            raise OpError("Для сдвига по долям нужна разметка долей B.")
        mp.b_entry_src += float(p["b_entry_shift_beats"]) * 60.0 / src.beats.bpm
    if mp.a_fade_len < 0.005 or mp.b_fade_len < 0.005:
        raise OpError("Длительность затухания/нарастания должна быть не меньше 5 мс.")
    b = ctx.work.clip(mt.clip_b_id)
    src_b = ctx.work.source(b.source_id) if b else None
    if mp.b_entry_src < 0 or (src_b and mp.b_entry_src >= src_b.duration):
        raise OpError("Точка входа B вне исходника B.")
    if mp.switch_time < 0 or mp.a_fade_start < 0 or mp.b_fade_start < 0:
        raise OpError("Переход не может начинаться раньше начала проекта.")
    mt.modified = True
    _refresh_music(ctx, mt)
    ctx.select = mt.id
    parts = []
    if abs(old.switch_time - mp.switch_time) > 1e-9:
        parts.append(f"вход B {seconds(old.switch_time)} → {seconds(mp.switch_time)}")
    if abs(old.b_entry_src - mp.b_entry_src) > 1e-9:
        parts.append(f"точка входа в исходнике B {seconds(old.b_entry_src)} → {seconds(mp.b_entry_src)}")
    if abs(old.a_fade_len - mp.a_fade_len) > 1e-9 or abs(old.a_fade_start - mp.a_fade_start) > 1e-9:
        parts.append(f"затухание A {_fmt_range(old.a_fade_start, old.a_fade_end)} → {_fmt_range(mp.a_fade_start, mp.a_fade_end)}")
    if abs(old.b_fade_len - mp.b_fade_len) > 1e-9 or abs(old.b_fade_start - mp.b_fade_start) > 1e-9:
        parts.append(f"нарастание B {_fmt_range(old.b_fade_start, old.b_fade_end)} → {_fmt_range(mp.b_fade_start, mp.b_fade_end)}")
    for k, lab in (("curve", "кривая"), ("a_tail", "хвост A"), ("overlap_filter", "фильтр наложения")):
        if getattr(old, k) != getattr(mp, k):
            parts.append(f"{lab}: {getattr(old, k)} → {getattr(mp, k)}")
    ctx.summary.append("Переход A → B: " + ("; ".join(parts) if parts else "без изменений") + ".")
    ctx.summary.append("Темп и высота тона не изменены; остальные параметры сохранены.")


def h_music_delete(ctx: Ctx, op: dict) -> None:
    mt = _music(ctx, op)
    _not_approved(ctx, mt)
    a = ctx.work.clip(mt.clip_a_id)
    b = ctx.work.clip(mt.clip_b_id)
    if a is not None and "src_out" in mt.orig_a:
        src = ctx.work.source(a.source_id)
        a.src_out = min(mt.orig_a["src_out"], src.duration if src else mt.orig_a["src_out"])
    if b is not None and mt.orig_b:
        b.start = mt.orig_b.get("start", b.start)
        b.src_in = mt.orig_b.get("src_in", b.src_in)
        b.src_out = mt.orig_b.get("src_out", b.src_out)
    ctx.work.music_transitions = [m for m in ctx.work.music_transitions if m.id != mt.id]
    ctx.summary.append(f"Переход A → B «{mt.name}» удалён; клипы возвращены в исходное положение.")


# --------------------------------------------------------------------------- tracks & clips


def h_track_add(ctx: Ctx, op: dict) -> None:
    name = (op.get("params") or {}).get("name") or f"Дорожка {len(ctx.work.tracks) + 1}"
    t = Track(id=new_id("trk"), name=name)
    ctx.work.tracks.append(t)
    ctx.created.append(t.id)
    ctx.select = t.id
    ctx.summary.append(f"Добавлена дорожка «{name}».")


def h_track_set(ctx: Ctx, op: dict) -> None:
    t = _track(ctx, op["target"]["track_id"])
    p = op.get("params", {}) or {}
    if not p:
        raise OpError("Не указано, что изменить на дорожке.")
    parts = []
    if "gain_db" in p or "delta_db" in p:
        old = t.gain_db
        new = float(p["gain_db"]) if "gain_db" in p else old + float(p["delta_db"])
        if new < -60 - 1e-9 or new > 12 + 1e-9:
            raise OpError(f"Уровень дорожки {new:+.1f} дБ вне диапазона −60…+12 дБ.")
        t.gain_db = new
        parts.append(f"уровень {old:+.1f} → {new:+.1f} дБ")
    if "mute" in p:
        t.mute = bool(p["mute"])
        parts.append("mute " + ("вкл" if t.mute else "выкл"))
    if "solo" in p:
        t.solo = bool(p["solo"])
        parts.append("solo " + ("вкл" if t.solo else "выкл"))
    if "name" in p:
        parts.append(f"имя «{t.name}» → «{p['name']}»")
        t.name = p["name"]
    ctx.select = t.id
    ctx.summary.append(f"Дорожка «{t.name}»: " + "; ".join(parts) + ".")


def _remove_music_for_clip(ctx: Ctx, cid: str) -> None:
    for mt in list(ctx.work.music_for_clip(cid)):
        _not_approved(ctx, mt)
        other_id = mt.clip_b_id if cid == mt.clip_a_id else mt.clip_a_id
        other = ctx.work.clip(other_id)
        if other is not None:
            if other_id == mt.clip_b_id and mt.orig_b:
                other.start, other.src_in = mt.orig_b.get("start", other.start), mt.orig_b.get("src_in", other.src_in)
            if other_id == mt.clip_a_id and "src_out" in mt.orig_a:
                other.src_out = mt.orig_a["src_out"]
        ctx.work.music_transitions = [m for m in ctx.work.music_transitions if m.id != mt.id]
        ctx.warnings.append(f"Переход A → B «{mt.name}» удалён вместе с клипом.")


def h_track_delete(ctx: Ctx, op: dict) -> None:
    t = _track(ctx, op["target"]["track_id"])
    for c in list(t.clips):
        if c.locked and c.id not in ctx.unlocked_clips:
            raise OpError(f"На дорожке есть зафиксированный клип «{c.name}».")
        _remove_music_for_clip(ctx, c.id)
    for x in ctx.work.transitions_on_track(t.id):
        _not_approved(ctx, x)
    ctx.work.space_transitions = [x for x in ctx.work.space_transitions if x.track_id != t.id]
    ctx.work.tracks = [tt for tt in ctx.work.tracks if tt.id != t.id]
    ctx.work.protections = [p for p in ctx.work.protections if p.track_id != t.id]
    ctx.summary.append(f"Дорожка «{t.name}» удалена вместе с клипами и переходами.")


def h_clip_add(ctx: Ctx, op: dict) -> None:
    w = ctx.work
    sid = op["target"]["source_id"]
    src = w.source(sid)
    if src is None:
        raise OpError(f"Исходник {sid} не найден в библиотеке.")
    tid = op["target"].get("track_id")
    if tid:
        track = _track(ctx, tid)
    else:
        track = Track(id=new_id("trk"), name=src.name)
        w.tracks.append(track)
        ctx.created.append(track.id)
    p = op.get("params", {}) or {}
    src_in = float(p.get("src_in", 0.0))
    src_out = float(p.get("src_out", src.duration))
    src_out = min(src_out, src.duration)
    if src_out - src_in < 0.01:
        raise OpError("Клип слишком короткий или границы вне исходника.")
    tm = op.get("time") or {}
    if "start" in tm:
        start = float(tm["start"])
    else:
        start = max((c.end for c in track.clips), default=0.0)
    c = Clip(id=new_id("clp"), source_id=sid, start=start, src_in=src_in, src_out=src_out,
             name=p.get("name") or src.name)
    track.clips.append(c)
    ctx.created.append(c.id)
    ctx.select = c.id
    ctx.summary.append(f"Клип «{c.name}» на дорожке «{track.name}» с {seconds(start)} "
                       f"(длина {seconds(c.length)}).")


def h_clip_move(ctx: Ctx, op: dict) -> None:
    w = ctx.work
    c = _clip(ctx, op["target"]["clip_id"])
    if c.locked and c.id not in ctx.unlocked_clips:
        raise OpError(f"Клип «{c.name}» зафиксирован.")
    tm = op.get("time") or {}
    p = op.get("params", {}) or {}
    old_start = c.start
    if tm.get("coord") == "relative":
        new_start = c.start + float(tm.get("delta", 0.0))
    elif "start" in tm:
        new_start = float(tm["start"])
    else:
        new_start = c.start
    if new_start < -1e-9:
        raise OpError("Клип не может начинаться раньше начала проекта.")
    delta = new_start - old_start
    old_track = w.clip_track(c.id)
    new_track = old_track
    if p.get("track_id") and p["track_id"] != old_track.id:
        new_track = _track(ctx, p["track_id"])
    if abs(delta) < 1e-12 and new_track is old_track:
        raise OpError("Положение клипа не изменилось.")
    linked = []
    # связанные переходы A → B
    for mt in w.music_for_clip(c.id):
        _not_approved(ctx, mt)
        mp = mt.params
        if c.id == mt.clip_a_id:
            mp.switch_time += delta
            mp.a_fade_start += delta
            mp.b_fade_start += delta
            b = w.clip(mt.clip_b_id)
            if b is not None:
                if b.locked and b.id not in ctx.unlocked_clips and abs(delta) > 1e-12:
                    raise OpError(f"Клип B «{b.name}» зафиксирован и не может переместиться вместе с A.")
                b.start += delta
                if mt.orig_b:
                    mt.orig_b["start"] = mt.orig_b.get("start", b.start) + delta
            linked.append(f"переход «{mt.name}» и клип B")
        else:
            mp.switch_time += delta
            mp.a_fade_start += delta
            mp.b_fade_start += delta
            linked.append(f"переход «{mt.name}» (момент входа B)")
    c.start = new_start
    # связанные пространственные переходы
    for x in w.space_transitions:
        if x.anchor_clip_id == c.id:
            _not_approved(ctx, x)
            x.start += delta
            x.end += delta
            if new_track is not old_track:
                x.track_id = new_track.id
            linked.append(f"переход «{x.name}»")
    if new_track is not old_track:
        old_track.clips = [cc for cc in old_track.clips if cc.id != c.id]
        new_track.clips.append(c)
        for x in w.transitions_on_track(new_track.id):
            _check_overlap(ctx, x.id, new_track.id, x.start, x.end)
    for mt in w.music_for_clip(c.id):
        _refresh_music(ctx, mt)
    ctx.select = c.id
    msg = f"Клип «{c.name}»: {seconds(old_start)} → {seconds(new_start)}"
    if new_track is not old_track:
        msg += f", дорожка «{new_track.name}»"
    ctx.summary.append(msg + ".")
    if linked:
        ctx.summary.append("Вместе с клипом перенесены: " + ", ".join(dict.fromkeys(linked)) + ".")


def h_clip_set(ctx: Ctx, op: dict) -> None:
    c = _clip(ctx, op["target"]["clip_id"])
    p = op.get("params", {}) or {}
    if not p:
        raise OpError("Не указано, что изменить в клипе.")
    if c.locked and c.id not in ctx.unlocked_clips:
        if p.get("locked") is False:
            ctx.unlocked_clips.add(c.id)
        else:
            raise OpError(f"Клип «{c.name}» зафиксирован. Снимите фиксацию, чтобы его изменить.")
    src = ctx.work.source(c.source_id)
    parts = []
    if "gain_db" in p or "delta_db" in p:
        old = c.gain_db
        new = float(p["gain_db"]) if "gain_db" in p else old + float(p["delta_db"])
        if new < -60 or new > 24:
            raise OpError("Усиление клипа вне диапазона −60…+24 дБ.")
        c.gain_db = new
        parts.append(f"усиление {old:+.1f} → {new:+.1f} дБ")
    if "src_in" in p or "src_out" in p:
        si = float(p.get("src_in", c.src_in))
        so = float(p.get("src_out", c.src_out))
        dur = src.duration if src else so
        if si < 0 or so > dur + 1e-6 or so - si < 0.01:
            raise OpError("Границы клипа вне исходника.")
        if "src_in" in p:
            c.start += si - c.src_in  # обрезка начала не сдвигает материал на шкале
        c.src_in, c.src_out = si, min(so, dur)
        parts.append(f"границы в исходнике {seconds(si)}–{seconds(c.src_out)}")
    for k, lab in (("fade_in", "нарастание"), ("fade_out", "затухание")):
        if k in p:
            v = float(p[k])
            if v > c.length / 2 + 1e-9:
                raise OpError(f"Фейд ({lab}) длиннее половины клипа.")
            setattr(c, k, v)
            parts.append(f"{lab} {v * 1000:.0f} мс")
    if "fade_shape" in p:
        c.fade_shape = p["fade_shape"]
        parts.append(f"форма фейдов {p['fade_shape']}")
    if "name" in p:
        c.name = p["name"]
        parts.append(f"имя «{c.name}»")
    if "locked" in p:
        c.locked = bool(p["locked"])
        parts.append("зафиксирован" if c.locked else "фиксация снята")
    for mt in ctx.work.music_for_clip(c.id):
        if "src_in" in p or "src_out" in p:
            _refresh_music(ctx, mt)
    ctx.select = c.id
    ctx.summary.append(f"Клип «{c.name}»: " + "; ".join(parts) + ".")


def h_clip_delete(ctx: Ctx, op: dict) -> None:
    c = _clip(ctx, op["target"]["clip_id"])
    if c.locked and c.id not in ctx.unlocked_clips:
        raise OpError(f"Клип «{c.name}» зафиксирован.")
    _remove_music_for_clip(ctx, c.id)
    t = ctx.work.clip_track(c.id)
    t.clips = [cc for cc in t.clips if cc.id != c.id]
    for x in ctx.work.space_transitions:
        if x.anchor_clip_id == c.id:
            x.anchor_clip_id = None
    ctx.summary.append(f"Клип «{c.name}» удалён с дорожки «{t.name}» (исходник остаётся в библиотеке).")


# --------------------------------------------------------------------------- constraints/protection


def h_constraint_set(ctx: Ctx, op: dict) -> None:
    p = op.get("params", {}) or {}
    if not p:
        raise OpError("Не указано ограничение.")
    parts = []
    for k, lab in (("keep_tempo", "темп"), ("keep_pitch", "высоту тона")):
        if k in p:
            ctx.work.constraints[k] = bool(p[k])
            parts.append(("не менять " if p[k] else "разрешено менять ") + lab)
    for mt in ctx.work.music_transitions:
        mt.conflicts, notes = music_mod.evaluate(ctx.work, mt)
    ctx.summary.append("Ограничения проекта: " + ", ".join(parts) + ".")
    if p.get("keep_tempo") or p.get("keep_pitch"):
        ctx.summary.append("В этой версии ASIKO темп и высота тона материала никогда не изменяются; "
                           "ограничение учитывается при подборе вариантов перехода и проверке команд.")


def h_protect_add(ctx: Ctx, op: dict) -> None:
    p = op["params"]
    s = float(p["start"])
    e = p.get("end")
    if e is not None and float(e) <= s:
        raise OpError("Конец защищаемого участка должен быть позже начала.")
    tid = p.get("track_id")
    if tid:
        _track(ctx, tid)
    pr = Protection(id=new_id("prt"), start=s, end=None if e is None else float(e), track_id=tid,
                    note=p.get("note", ""))
    ctx.work.protections.append(pr)
    ctx.created.append(pr.id)
    where = f"дорожка «{ctx.work.track(tid).name}»" if tid else "все дорожки"
    ctx.summary.append(f"Защищён участок {_fmt_range(s, pr.end)} ({where}): настройки в нём не будут меняться. "
                       f"Хвосты эффектов из соседних участков могут звучать внутри — для побитовой "
                       f"неизменности используйте фиксацию в аудио.")


def h_protect_remove(ctx: Ctx, op: dict) -> None:
    pid = op["target"]["protection_id"]
    pr = ctx.work.protection(pid)
    if pr is None:
        raise OpError("Защищённый участок не найден.")
    ctx.work.protections = [p for p in ctx.work.protections if p.id != pid]
    ctx.removed_protections.add(pid)
    ctx.summary.append(f"Снята защита участка {_fmt_range(pr.start, pr.end)}.")


# --------------------------------------------------------------------------- beats


def _source(ctx: Ctx, sid: str) -> Source:
    s = ctx.work.source(sid)
    if s is None:
        raise OpError(f"Исходник {sid} не найден.")
    return s


def _reeval_music_for_source(ctx: Ctx, sid: str) -> None:
    for mt in ctx.work.music_transitions:
        clips = [ctx.work.clip(mt.clip_a_id), ctx.work.clip(mt.clip_b_id)]
        if any(c is not None and c.source_id == sid for c in clips):
            mt.conflicts, notes = music_mod.evaluate(ctx.work, mt)
            ctx.warnings.append(f"Разметка изменена: варианты перехода «{mt.name}» не пересчитаны автоматически — "
                                f"при необходимости пересоздайте переход.")


def h_beats_set(ctx: Ctx, op: dict) -> None:
    s = _source(ctx, op["target"]["source_id"])
    p = op.get("params", {}) or {}
    if "grid" in p:
        g = p["grid"]
        try:
            grid = BeatGrid.from_dict(g)
        except TypeError as e:
            raise OpError(f"Неверная сетка долей: {e}")
        if not (20 <= grid.bpm <= 400):
            raise OpError("Темп сетки вне диапазона 20–400 BPM.")
        for arr in (grid.beats, grid.downbeats):
            if any((not isinstance(v, (int, float))) or v < -1e-6 or v > s.duration + 1 for v in arr):
                raise OpError("Доли сетки вне исходника.")
            if any(b2 < b1 for b1, b2 in zip(arr, arr[1:])):
                raise OpError("Доли сетки должны идти по возрастанию.")
        s.beats = grid
        ctx.summary.append(f"Сетка долей «{s.name}»: {grid.bpm:.1f} BPM, уверенность {grid.confidence:.0%} "
                           f"({'анализ' if grid.method == 'auto' else 'вручную'}).")
    elif "bpm" in p:
        bpb = int(p.get("beats_per_bar", s.beats.beats_per_bar if s.beats else 4))
        fd = float(p.get("first_downbeat", s.beats.downbeats[0] if s.beats and s.beats.downbeats else 0.0))
        pb = int(p.get("phrase_bars", s.beats.phrase_bars if s.beats else 8))
        s.beats = BeatGrid.manual(float(p["bpm"]), fd, s.duration, bpb, pb)
        ctx.summary.append(f"Сетка долей «{s.name}» задана вручную: {p['bpm']:.1f} BPM, первая сильная доля "
                           f"{seconds(fd)}, {bpb} доли в такте, фраза {pb} тактов.")
    elif "phrase_bars" in p and s.beats:
        s.beats.phrase_bars = int(p["phrase_bars"])
        ctx.summary.append(f"Длина фразы «{s.name}»: {s.beats.phrase_bars} тактов.")
    else:
        raise OpError("Не указана сетка долей (grid или bpm).")
    _reeval_music_for_source(ctx, s.id)


def h_beats_shift(ctx: Ctx, op: dict) -> None:
    s = _source(ctx, op["target"]["source_id"])
    if s.beats is None:
        raise OpError("У исходника нет сетки долей.")
    d = float(op["params"]["delta"])
    s.beats.beats = [b + d for b in s.beats.beats if 0 <= b + d <= s.duration]
    s.beats.downbeats = [b + d for b in s.beats.downbeats if 0 <= b + d <= s.duration]
    s.beats.method = "edited"
    ctx.summary.append(f"Сетка долей «{s.name}» сдвинута на {d * 1000:+.0f} мс.")
    _reeval_music_for_source(ctx, s.id)


def h_phrases_set(ctx: Ctx, op: dict) -> None:
    s = _source(ctx, op["target"]["source_id"])
    ph = sorted(float(v) for v in op["params"]["phrases"])
    if any(v < 0 or v > s.duration for v in ph):
        raise OpError("Границы фраз вне исходника.")
    s.phrases = ph
    ctx.summary.append(f"Границы фраз «{s.name}»: " + (", ".join(seconds(v) for v in ph) if ph else "нет") + ".")
    _reeval_music_for_source(ctx, s.id)


# --------------------------------------------------------------------------- freeze, video, project


def h_freeze_add(ctx: Ctx, op: dict) -> None:
    p = op["params"]
    s, e = float(p["start"]), float(p["end"])
    if e <= s:
        raise OpError("Конец участка должен быть позже начала.")
    for f in ctx.work.frozen:
        if f.start < e and s < f.end:
            raise OpError("Участок пересекается с уже зафиксированным.")
    fr = FrozenRegion(id=p.get("id") or new_id("frz"), start=s, end=e, file=p["file"], sha256=p["sha256"],
                      sample_rate=int(p["sample_rate"]), margin=float(p.get("margin", 0.02)), note=p.get("note", ""))
    ctx.work.frozen.append(fr)
    ctx.created.append(fr.id)
    ctx.summary.append(f"Участок {_fmt_range(s, e)} зафиксирован в PCM ({fr.sample_rate} Гц, float32): "
                       f"при экспорте в WAV {fr.sample_rate // 1000} кГц эти сэмплы не изменятся. "
                       f"Побитовая идентичность не гарантируется при другой частоте или в lossy-формате.")


def h_freeze_remove(ctx: Ctx, op: dict) -> None:
    fid = op["target"]["frozen_id"]
    fr = next((f for f in ctx.work.frozen if f.id == fid), None)
    if fr is None:
        raise OpError("Зафиксированный участок не найден.")
    ctx.work.frozen = [f for f in ctx.work.frozen if f.id != fid]
    ctx.summary.append(f"Снята фиксация PCM участка {_fmt_range(fr.start, fr.end)}.")


def h_video_set(ctx: Ctx, op: dict) -> None:
    p = op["params"]
    import os

    ctx.work.video = VideoRef(path=p["path"], offset=float(p.get("offset", 0.0)),
                              name=p.get("name") or os.path.basename(p["path"]))
    ctx.summary.append(f"Подключено видео «{ctx.work.video.name}» (начало на {seconds(ctx.work.video.offset)}).")


def h_video_offset(ctx: Ctx, op: dict) -> None:
    if ctx.work.video is None:
        raise OpError("Видео не подключено.")
    ctx.work.video.offset = float(op["params"]["offset"])
    ctx.summary.append(f"Смещение видео: начало на {seconds(ctx.work.video.offset)}.")


def h_video_clear(ctx: Ctx, op: dict) -> None:
    if ctx.work.video is None:
        raise OpError("Видео не подключено.")
    ctx.work.video = None
    ctx.summary.append("Видео отключено.")


def h_project_rename(ctx: Ctx, op: dict) -> None:
    ctx.work.name = op["params"]["name"]
    ctx.summary.append(f"Проект переименован: «{ctx.work.name}».")


def h_source_add(ctx: Ctx, op: dict) -> None:
    d = op["params"]["source"]
    try:
        s = Source.from_dict(d)
    except (TypeError, KeyError) as e:
        raise OpError(f"Неверное описание исходника: {e}")
    if ctx.work.source(s.id):
        raise OpError("Исходник с таким идентификатором уже есть.")
    if s.sample_rate <= 0 or s.frames <= 0 or s.channels not in (1, 2):
        raise OpError("Поддерживаются моно и стерео исходники с ненулевой длительностью.")
    ctx.work.sources.append(s)
    ctx.created.append(s.id)
    ctx.summary.append(f"Импортирован «{s.name}»: {s.sample_rate} Гц, "
                       f"{'стерео' if s.channels == 2 else 'моно'}, {seconds(s.duration)}.")


def h_source_relink(ctx: Ctx, op: dict) -> None:
    s = _source(ctx, op["target"]["source_id"])
    s.path = op["params"]["path"]
    ctx.summary.append(f"Исходник «{s.name}» найден: {s.path}.")


def h_source_remove(ctx: Ctx, op: dict) -> None:
    s = _source(ctx, op["target"]["source_id"])
    if any(c.source_id == s.id for c in ctx.work.all_clips()):
        raise OpError("Исходник используется клипами — сначала удалите клипы.")
    ctx.work.sources = [x for x in ctx.work.sources if x.id != s.id]
    ctx.summary.append(f"Исходник «{s.name}» удалён из библиотеки (файл на диске не тронут).")


HANDLERS = {
    "space.create": h_space_create,
    "space.set_timing": h_space_set_timing,
    "space.set_curve": h_space_set_curve,
    "space.set_state": h_space_set_state,
    "space.set_slope": h_space_set_slope,
    "space.delete": h_space_delete,
    "transition.lock_params": h_lock_params,
    "transition.approve": h_approve,
    "music.create": h_music_create,
    "music.select_variant": h_music_select,
    "music.set": h_music_set,
    "music.delete": h_music_delete,
    "track.add": h_track_add,
    "track.set": h_track_set,
    "track.delete": h_track_delete,
    "clip.add": h_clip_add,
    "clip.move": h_clip_move,
    "clip.set": h_clip_set,
    "clip.delete": h_clip_delete,
    "constraint.set": h_constraint_set,
    "protect.add": h_protect_add,
    "protect.remove": h_protect_remove,
    "beats.set": h_beats_set,
    "beats.shift": h_beats_shift,
    "phrases.set": h_phrases_set,
    "freeze.add": h_freeze_add,
    "freeze.remove": h_freeze_remove,
    "video.set": h_video_set,
    "video.offset": h_video_offset,
    "video.clear": h_video_clear,
    "project.rename": h_project_rename,
    "source.add": h_source_add,
    "source.relink": h_source_relink,
    "source.remove": h_source_remove,
}


def check_invariants(p: Project) -> list[str]:
    errs = []
    ids = set()
    for s in p.sources:
        ids.add(s.id)
    for t in p.tracks:
        for c in t.clips:
            src = p.source(c.source_id)
            if src is None:
                errs.append(f"Клип «{c.name}» ссылается на отсутствующий исходник.")
                continue
            if c.start < -1e-9:
                errs.append(f"Клип «{c.name}» начинается раньше начала проекта.")
            if c.src_in < -1e-9 or c.src_out > src.duration + 1e-6 or c.length < 0.005:
                errs.append(f"Границы клипа «{c.name}» вне исходника.")
    for x in p.space_transitions:
        if p.track(x.track_id) is None:
            errs.append(f"Переход «{x.name}» ссылается на отсутствующую дорожку.")
        if x.end - x.start < MIN_TRANSITION_S - 1e-9:
            errs.append(f"Переход «{x.name}»: неверные границы.")
        for k, c in x.curves.items():
            if not (0 <= c.start < c.end <= 1):
                errs.append(f"Переход «{x.name}»: неверная кривая параметра {k}.")
    for tid in {x.track_id for x in p.space_transitions}:
        xs = p.transitions_on_track(tid)
        for a, b in zip(xs, xs[1:]):
            if b.start < a.end - 1e-9:
                errs.append(f"Переходы «{a.name}» и «{b.name}» пересекаются.")
    for m in p.music_transitions:
        if p.clip(m.clip_a_id) is None or p.clip(m.clip_b_id) is None:
            errs.append(f"Переход A → B «{m.name}» ссылается на отсутствующий клип.")
    for pr in p.protections:
        if pr.track_id and p.track(pr.track_id) is None:
            errs.append("Защищённый участок ссылается на отсутствующую дорожку.")
    return errs
