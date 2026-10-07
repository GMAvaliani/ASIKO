"""Движок команд: проверка плана, транзакция, история (undo/redo).

Путь любого изменения: план → проверка схемы → проверка версии проекта →
применение на копии → инварианты → блокировки/защищённые участки → фиксация.
Ошибка на любом шаге — проект не меняется (атомарность транзакции).
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field

from ..model.project import Project
from . import protection
from .handlers import HANDLERS, Ctx, OpError, check_invariants
from .schema import PRECONDITION_KINDS, validate_plan_schema

HISTORY_LIMIT = 500


@dataclass
class Result:
    ok: bool
    applied: bool = False
    kind: str = "edit"
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summary: list[str] = field(default_factory=list)
    stale: bool = False
    needs_confirmation: bool = False
    artistic: bool = False
    message: str = ""
    question: dict | None = None
    created: list[str] = field(default_factory=list)
    select: str | None = None
    revision: int = 0
    preview: Project | None = None

    @property
    def text(self) -> str:
        parts = []
        if self.summary:
            parts.append(" ".join(self.summary))
        if self.message:
            parts.append(self.message)
        return "\n".join(parts)


@dataclass
class HistoryEntry:
    description: str
    origin: str
    before: dict
    after: dict
    plan: dict
    revision: int
    timestamp: float


def _strip_rev(d: dict) -> dict:
    d = dict(d)
    d.pop("revision", None)
    return d


class CommandEngine:
    def __init__(self, project: Project | None = None, user_presets: dict | None = None):
        self.project = project or Project()
        self.user_presets = user_presets or {}
        self.undo_stack: list[HistoryEntry] = []
        self.redo_stack: list[HistoryEntry] = []
        self._listeners: list = []
        self.dirty = False

    # ---------------------------------------------------------------- listeners
    def subscribe(self, fn) -> None:
        self._listeners.append(fn)

    def _notify(self, result: Result) -> None:
        for fn in list(self._listeners):
            fn(result)

    def reset(self, project: Project) -> None:
        self.project = project
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False
        self._notify(Result(ok=True, applied=True, kind="reset", revision=project.revision))

    # ---------------------------------------------------------------- core
    def _transaction(self, plan: dict) -> tuple[Project | None, Ctx | None, list[str]]:
        before = self.project
        work = before.clone()
        ctx = Ctx(before=before, work=work, origin=plan.get("origin", "ui"), user_presets=self.user_presets)
        errors: list[str] = []
        for i, op in enumerate(plan.get("operations", [])):
            try:
                self._check_preconditions(op, work, plan)
                HANDLERS[op["op"]](ctx, op)
            except OpError as e:
                errors.append(f"{op['op']}: {e}" if len(plan.get("operations", [])) > 1 else str(e))
                return None, ctx, errors
            except (KeyError, TypeError, ValueError) as e:  # защитный барьер: неожиданные данные
                errors.append(f"{op.get('op')}: некорректные данные ({e})")
                return None, ctx, errors
        errors += check_invariants(work)
        if errors:
            return None, ctx, errors
        errors += protection.check_locks(before, work, ctx.unlocked_params, ctx.unapproved, ctx.unlocked_clips)
        for prot in before.protections:
            if prot.id in ctx.removed_protections:
                continue
            errors += protection.check_range(before, work, prot)
        if errors:
            return None, ctx, errors
        for prot in before.protections:
            if prot.id not in ctx.removed_protections:
                ctx.warnings += protection.tail_warnings(before, work, prot)
        ctx.warnings += protection.freeze_warnings(before, work)
        return work, ctx, []

    def _check_preconditions(self, op: dict, work: Project, plan: dict) -> None:
        for pc in op.get("preconditions", []) or []:
            k = pc["kind"]
            if k == "exists":
                oid = pc["id"]
                if not (work.track(oid) or work.clip(oid) or work.any_transition(oid) or work.source(oid)
                        or work.protection(oid)):
                    raise OpError(f"Предусловие не выполнено: объект {oid} не существует.")
            elif k == "not_approved":
                x = work.any_transition(pc["id"])
                if x is not None and x.approved:
                    raise OpError(f"Предусловие не выполнено: переход «{x.name}» утверждён.")
            elif k == "revision":
                if self.project.revision != pc["value"]:
                    raise OpError(f"Предусловие не выполнено: версия проекта {self.project.revision}, "
                                  f"ожидалась {pc['value']}.")
            elif k == "duration_at_least":
                x = work.any_transition(pc["id"])
                if x is None or getattr(x, "length", 0.0) < pc["seconds"] - 1e-9:
                    raise OpError(f"Предусловие не выполнено: длительность перехода меньше {pc['seconds']} с.")

    def check(self, plan: dict) -> Result:
        """Полная проверка без изменения проекта."""
        errs = validate_plan_schema(plan)
        kind = plan.get("kind", "edit") if isinstance(plan, dict) else "edit"
        if errs:
            return Result(ok=False, kind=kind, errors=errs)
        res = Result(ok=True, kind=kind, artistic=bool(plan.get("artistic")), message=plan.get("message", ""),
                     question=plan.get("question"), revision=self.project.revision)
        if kind in ("info", "question"):
            return res
        if kind == "undo":
            res.ok = bool(self.undo_stack)
            if not res.ok:
                res.errors.append("Нечего отменять.")
            else:
                res.summary.append(f"Будет отменено: {self.undo_stack[-1].description}")
            return res
        if kind == "redo":
            res.ok = bool(self.redo_stack)
            if not res.ok:
                res.errors.append("Нечего повторять.")
            else:
                res.summary.append(f"Будет повторено: {self.redo_stack[-1].description}")
            return res
        if plan["base_revision"] != self.project.revision:
            res.stale = True
            res.needs_confirmation = True
            res.warnings.append(f"Команда сформирована для версии проекта {plan['base_revision']}, текущая — "
                                f"{self.project.revision}. План проверен заново на текущем проекте.")
        work, ctx, errors = self._transaction(plan)
        if errors:
            res.ok = False
            res.errors += errors
            return res
        res.summary = ctx.summary
        res.warnings += ctx.warnings
        res.created = ctx.created
        res.select = ctx.select
        res.preview = work
        if res.artistic or kind == "proposal":
            res.needs_confirmation = True
        return res

    def apply(self, plan: dict, confirm_stale: bool = False, accept_proposal: bool = False) -> Result:
        res = self.check(plan)
        if not res.ok:
            self._notify(res)
            return res
        kind = res.kind
        if kind == "undo":
            return self.undo()
        if kind == "redo":
            return self.redo()
        if kind in ("info", "question"):
            return res
        if res.stale and not confirm_stale:
            res.ok = True
            res.applied = False
            return res
        if (res.artistic or kind == "proposal") and not accept_proposal:
            res.applied = False
            return res
        work = res.preview
        before = self.project
        work.revision = before.revision + 1
        desc = " ".join(res.summary) if res.summary else plan.get("text", "Изменение")
        if res.artistic or kind == "proposal":
            desc = "[Художественное предложение] " + desc
        entry = HistoryEntry(description=desc, origin=plan.get("origin", "ui"), before=before.to_dict(),
                             after=work.to_dict(), plan=copy.deepcopy(plan), revision=work.revision,
                             timestamp=time.time())
        self.undo_stack.append(entry)
        if len(self.undo_stack) > HISTORY_LIMIT:
            self.undo_stack.pop(0)
        self.redo_stack.clear()
        self.project = work
        self.dirty = True
        res.applied = True
        res.preview = None
        res.revision = work.revision
        self._notify(res)
        return res

    def undo(self) -> Result:
        if not self.undo_stack:
            r = Result(ok=False, kind="undo", errors=["Нечего отменять."])
            self._notify(r)
            return r
        e = self.undo_stack.pop()
        p = Project.from_dict(e.before)
        p.revision = self.project.revision + 1
        self.project = p
        self.redo_stack.append(e)
        self.dirty = True
        r = Result(ok=True, applied=True, kind="undo", summary=[f"Отменено: {e.description}"], revision=p.revision)
        self._notify(r)
        return r

    def redo(self) -> Result:
        if not self.redo_stack:
            r = Result(ok=False, kind="redo", errors=["Нечего повторять."])
            self._notify(r)
            return r
        e = self.redo_stack.pop()
        p = Project.from_dict(e.after)
        p.revision = self.project.revision + 1
        self.project = p
        self.undo_stack.append(e)
        self.dirty = True
        r = Result(ok=True, applied=True, kind="redo", summary=[f"Повторено: {e.description}"], revision=p.revision)
        self._notify(r)
        return r

    def undo_to(self, index: int) -> Result:
        """Отменить всё после записи истории с индексом index (0 — самая старая)."""
        r = Result(ok=True, kind="undo")
        while len(self.undo_stack) > index + 1:
            r = self.undo()
        return r

    def history(self) -> list[HistoryEntry]:
        return list(self.undo_stack)

    def same_content(self, a: dict, b: dict) -> bool:
        return json.dumps(_strip_rev(a), sort_keys=True) == json.dumps(_strip_rev(b), sort_keys=True)
