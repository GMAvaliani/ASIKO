"""Проверка блокировок и защищённых участков (защита настроек).

Защита настроек сравнивает «отпечаток» эффективных настроек на защищённом участке до и
после транзакции: значения всех автоматизируемых параметров на сетке 1 мс, состав
звучащих клипов, их привязку к исходнику, огибающие и уровень дорожки. Любое
отличие — нарушение. Изменения вне участка разрешены, но если они могут прозвучать
внутри участка через хвост эффекта, выдаётся предупреждение (защита сэмплов
обеспечивается только фиксацией участка в PCM).
"""
from __future__ import annotations

import math
from dataclasses import asdict

import numpy as np

from ..model.automation import (
    all_space_params,
    clip_effect_tail,
    clip_envelope,
    clip_filter,
    seconds,
)
from ..model.presets import PARAM_INFO
from ..model.project import Clip, Project, Protection, SpaceTransition

GRID_S = 0.001
TOL = 1e-7
EFFECT_MEMORY_S = 3.0  # максимальная длина хвоста реверберации


def _grid(t0: float, t1: float) -> np.ndarray:
    n = max(2, int(math.ceil((t1 - t0) / GRID_S)) + 1)
    return t0 + np.arange(n) * GRID_S


def _track_fingerprint(project: Project, tid: str, t: np.ndarray) -> dict:
    out: dict = {}
    tr = project.track(tid)
    if tr is None:
        out["__track__"] = None
        return out
    out["__track__"] = (round(tr.gain_db, 9), tr.mute)
    xs = project.transitions_on_track(tid)
    for k, v in all_space_params(xs, t).items():
        out[f"param:{k}"] = v
    out["slope"] = tuple(sorted((round(x.start, 9), round(x.end, 9), x.filter_slope) for x in xs
                                if x.start <= t[-1] and x.end >= t[0]))
    sr = project.sample_rate
    n = np.round(t * sr).astype(np.int64)
    audible = []
    for c in tr.clips:
        music = project.music_for_clip(c.id)
        env = clip_envelope(c, sr, n, music)
        if not np.any(env > 0):
            continue
        audible.append(c.id)
        offset = int(round(c.start * sr)) - int(round(c.src_in * sr))
        out[f"clip:{c.id}:map"] = (c.source_id, offset)
        out[f"clip:{c.id}:env"] = env
        f = clip_filter(c, n, sr, music)
        out[f"clip:{c.id}:filter"] = None if f is None else (f[0], f[1])
        tail = clip_effect_tail(c, n, sr, music)
        out[f"clip:{c.id}:tail"] = None if tail is None else (tail[0], tail[1])
    out["clips"] = tuple(sorted(audible))
    return out


def _equal(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)) or a.shape != b.shape:
            return False
        return bool(np.allclose(a, b, rtol=0, atol=TOL))
    if isinstance(a, tuple) and isinstance(b, tuple):
        if len(a) != len(b):
            return False
        return all(_equal(x, y) for x, y in zip(a, b))
    return a == b


def _first_diff_time(a, b, t: np.ndarray) -> float | None:
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.shape == b.shape:
        idx = np.nonzero(np.abs(a - b) > TOL)[0]
        if idx.size:
            return float(t[idx[0]])
    return None


_KEY_LABELS = {
    "__track__": "уровень/mute дорожки",
    "clips": "состав звучащих клипов",
    "slope": "крутизна фильтров",
}


def _label_for_key(key: str) -> str:
    if key in _KEY_LABELS:
        return _KEY_LABELS[key]
    if key.startswith("param:"):
        p = key[6:]
        if p.startswith("reverb_db@"):
            return "комнатная обработка"
        lab = PARAM_INFO.get(p, {}).get("label", p)
        return lab[:1].lower() + lab[1:]
    if key.startswith("clip:"):
        part = key.rsplit(":", 1)[-1]
        return {"map": "положение клипа", "env": "огибающая клипа", "filter": "фильтр клипа",
                "tail": "хвост эффекта"}.get(part, "клип")
    return key


def check_range(before: Project, after: Project, prot: Protection) -> list[str]:
    end_candidates = [before.content_end(), after.content_end(), prot.start + 1.0]
    t1 = prot.end if prot.end is not None else max(end_candidates) + EFFECT_MEMORY_S
    t0 = prot.start
    if t1 <= t0:
        return []
    t = _grid(t0, t1)
    if prot.track_id:
        tids = [prot.track_id]
    else:
        tids = sorted({tr.id for tr in before.tracks} | {tr.id for tr in after.tracks})
    errors = []
    rng = f"{seconds(t0)}–{seconds(prot.end) if prot.end is not None else 'конец'}"
    for tid in tids:
        fb = _track_fingerprint(before, tid, t)
        fa = _track_fingerprint(after, tid, t)
        tr = before.track(tid) or after.track(tid)
        tname = tr.name if tr else tid
        changed = []
        when = None
        for k in sorted(set(fb) | set(fa)):
            if not _equal(fb.get(k), fa.get(k)):
                changed.append(_label_for_key(k))
                w = _first_diff_time(fb.get(k), fa.get(k), t)
                if w is not None:
                    when = w if when is None else min(when, w)
        if changed:
            uniq = list(dict.fromkeys(changed))
            at = f" (с {seconds(when)})" if when is not None else ""
            errors.append(
                f"Защищённый участок {rng} на дорожке «{tname}»: изменение затрагивает "
                f"{', '.join(uniq)}{at}. Снимите защиту или измените только незащищённую часть.")
    return errors


def tail_warnings(before: Project, after: Project, prot: Protection) -> list[str]:
    """Изменения непосредственно перед защищённым участком могут звучать в нём хвостом."""
    t0 = max(0.0, prot.start - EFFECT_MEMORY_S)
    t1 = prot.start
    if t1 - t0 < GRID_S * 2:
        return []
    t = _grid(t0, t1)
    tids = [prot.track_id] if prot.track_id else sorted({tr.id for tr in before.tracks} | {tr.id for tr in after.tracks})
    for tid in tids:
        fb = _track_fingerprint(before, tid, t)
        fa = _track_fingerprint(after, tid, t)
        if any(not _equal(fb.get(k), fa.get(k)) for k in set(fb) | set(fa)):
            return [f"Изменение перед защищённым участком (с {seconds(prot.start)}) может прозвучать в нём "
                    f"хвостом реверберации/фильтров. Настройки участка не изменены; для побитовой "
                    f"неизменности зафиксируйте участок в аудио."]
    return []


def _rel_curve(project: Project, x: SpaceTransition, p: str) -> tuple:
    a0, a1 = x.curve_abs(p)
    base = 0.0
    if x.anchor_clip_id:
        c = project.clip(x.anchor_clip_id)
        if c is not None:
            base = c.start
    return round(a0 - base, 7), round(a1 - base, 7), x.curves[p].shape


def check_locks(before: Project, after: Project, unlocked_params: dict[str, set],
                unapproved: set, unlocked_clips: set) -> list[str]:
    errors: list[str] = []
    # клипы
    for c in before.all_clips():
        if not c.locked or c.id in unlocked_clips:
            continue
        a = after.clip(c.id)
        tb = before.clip_track(c.id)
        ta = after.clip_track(c.id)
        if a is None:
            errors.append(f"Клип «{c.name or c.id}» зафиксирован: его нельзя удалить.")
            continue
        fields_b = (c.source_id, round(c.start, 9), round(c.src_in, 9), round(c.src_out, 9),
                    round(c.gain_db, 9), round(c.fade_in, 9), round(c.fade_out, 9), c.fade_shape,
                    tb.id if tb else None)
        fields_a = (a.source_id, round(a.start, 9), round(a.src_in, 9), round(a.src_out, 9),
                    round(a.gain_db, 9), round(a.fade_in, 9), round(a.fade_out, 9), a.fade_shape,
                    ta.id if ta else None)
        if fields_b != fields_a:
            errors.append(f"Клип «{c.name or c.id}» зафиксирован (положение и настройки). "
                          f"Снимите фиксацию клипа, чтобы изменить его.")
    # отдельные параметры
    for x in before.space_transitions:
        locked = set(x.locked_params) - unlocked_params.get(x.id, set())
        if not locked:
            continue
        y = after.space_transition(x.id)
        if y is None:
            errors.append(f"У перехода «{x.name}» есть зафиксированные параметры — его нельзя удалить.")
            continue
        for p in sorted(locked):
            same = (abs(x.from_state.values[p] - y.from_state.values[p]) < 1e-9
                    and abs(x.to_state.values[p] - y.to_state.values[p]) < 1e-9
                    and _rel_curve(before, x, p) == _rel_curve(after, y, p))
            if p == "reverb_db":
                same = same and x.from_state.room == y.from_state.room and x.to_state.room == y.to_state.room
            if not same:
                errors.append(f"Параметр «{PARAM_INFO[p]['label']}» перехода «{x.name}» зафиксирован.")
    # утверждённые переходы
    for x in before.space_transitions:
        if not x.approved or x.id in unapproved:
            continue
        y = after.space_transition(x.id)
        if y is None or asdict(x) != asdict(y):
            errors.append(f"Переход «{x.name}» утверждён и защищён от изменений. Снимите утверждение, чтобы править.")
    for m in before.music_transitions:
        if not m.approved or m.id in unapproved:
            continue
        y = after.music_transition(m.id)
        same = y is not None and asdict(m) == asdict(y)
        if same:
            for cid in (m.clip_a_id, m.clip_b_id):
                cb, ca = before.clip(cid), after.clip(cid)
                if (cb is None) != (ca is None) or (cb and asdict(cb) != asdict(ca)):
                    same = False
        if not same:
            errors.append(f"Переход A→B «{m.name}» утверждён: его параметры и клипы защищены.")
    return errors


def freeze_warnings(before: Project, after: Project) -> list[str]:
    warns = []
    for fr in after.frozen:
        prot = Protection(id="tmp", start=fr.start, end=fr.end)
        if check_range(before, after, prot):
            warns.append(f"Участок {seconds(fr.start)}–{seconds(fr.end)} зафиксирован в аудио: изменение "
                         f"настроек в нём не будет слышно в рендере (звучит сохранённый PCM). На границах "
                         f"возможен разрыв — проверьте отчёт о границах.")
    return warns
