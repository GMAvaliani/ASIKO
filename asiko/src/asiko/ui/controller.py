"""Контроллер сессии: связывает проект, команды, рендер, воспроизведение и хранение.

Не содержит виджетов — используется окном, сценарными тестами и самопроверкой.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QSettings, QTimer, Signal

from .. import app_paths
from ..audio import dsp
from ..audio.analysis import analyze_beats, peak_envelope
from ..audio.export import ExportOptions, export_wav
from ..audio.freeze import FrozenLoader, freeze_params
from ..audio.playback import PlaybackEngine
from ..audio.preview import PreviewRenderer, PreviewRequest, loudness_offsets
from ..audio.render import ChunkCache, frozen_boundary_report
from ..audio.sources import SourceStore
from ..commands.engine import CommandEngine, Result
from ..commands.schema import make_plan, op
from ..model.automation import seconds
from ..model.project import Project
from ..nlp.llm import DEFAULTS, LlmAdapter, LlmConfig
from ..nlp.local_parser import ParseContext, Selection, parse
from ..nlp.secrets import SecretStore
from ..storage.project_io import Autosaver, data_dir_for, load_project, save_project
from .workers import run_job


def settings() -> QSettings:
    return QSettings(str(app_paths.config_dir() / "settings.ini"), QSettings.Format.IniFormat)


class Controller(QObject):
    projectChanged = Signal(object)        # Result
    selectionChanged = Signal()
    previewProgress = Signal(str, float)
    previewDone = Signal(str, object)
    message = Signal(str, str)             # level (info|warn|error), text
    pathChanged = Signal()
    _bg_progress = Signal(str, float)
    _bg_done = Signal(str, object)

    def __init__(self, parent=None, open_audio: bool = True):
        super().__init__(parent)
        self.settings = settings()
        self.user_presets = self._load_presets()
        self.engine = CommandEngine(Project(), user_presets=self.user_presets)
        self.engine.subscribe(self._on_engine)
        self.store = SourceStore(app_paths.audio_cache_dir(), 48000)
        self.cache = ChunkCache(max_bytes=int(self.settings.value("cache_mb", 1024)) << 20)
        self.playback = PlaybackEngine(48000)
        self.frozen = FrozenLoader()
        self.preview = PreviewRenderer(self.store, self.cache, self.playback, frozen_audio=self.frozen)
        self.preview.on_progress = self._emit_progress
        self.preview.on_done = self._emit_done
        self._bg_progress.connect(self.previewProgress)
        self._bg_done.connect(self._preview_finished)
        self.path: Path | None = None
        self.selection = Selection()
        self.compare_mode = "result"
        self.compare_region: tuple[float, float] | None = None
        self.peak_env = np.zeros(0, np.float32)
        self.master_peak = 0.0
        self.boundary_report = []
        self.rendered_revision: dict[str, int | None] = {}
        self.secrets = SecretStore()
        self.llm_config = self._load_llm_config()
        self.use_llm = bool(self.settings.value("use_llm", "false") == "true")
        self.autosaver = Autosaver(app_paths.autosave_dir())
        self._autosaved_rev = -1
        self._debounce = QTimer(self)
        self._debounce.setSingleShot(True)
        self._debounce.setInterval(120)
        self._debounce.timeout.connect(self._do_preview)
        self.autosave_timer = QTimer(self)
        self.autosave_timer.setInterval(int(self.settings.value("autosave_s", 60)) * 1000)
        self.autosave_timer.timeout.connect(self.autosave)
        self.autosave_timer.start()
        self.last_result: Result | None = None
        if open_audio:
            self.open_audio()

    def _emit_progress(self, name, frac):
        try:
            self._bg_progress.emit(name, frac)
        except RuntimeError:  # окно уже закрыто
            pass

    def _emit_done(self, name, buf, info):
        try:
            self._bg_done.emit(name, (buf, info))
        except RuntimeError:
            pass

    # ---------------------------------------------------------------- basics
    @property
    def project(self) -> Project:
        return self.engine.project

    def open_audio(self) -> None:
        dev = self.settings.value("audio_device", None)
        try:
            dev = int(dev) if dev not in (None, "", "default") else None
        except (TypeError, ValueError):
            dev = None
        self.playback.open(device=dev)
        if self.playback.error:
            self.message.emit("warn", self.playback.error)

    def shutdown(self) -> None:
        self.preview.on_progress = None
        self.preview.on_done = None
        self.preview.shutdown()
        self.playback.close()

    def _load_presets(self) -> dict:
        p = app_paths.config_dir() / "presets.json"
        try:
            return json.loads(p.read_text("utf-8"))
        except (OSError, json.JSONDecodeError):
            return {}

    def save_user_preset(self, key: str, preset: dict) -> None:
        self.user_presets[key] = preset
        from ..audio.io import atomic_write_bytes

        atomic_write_bytes(app_paths.config_dir() / "presets.json",
                           json.dumps(self.user_presets, ensure_ascii=False, indent=1).encode("utf-8"))

    def delete_user_preset(self, key: str) -> None:
        self.user_presets.pop(key, None)
        from ..audio.io import atomic_write_bytes

        atomic_write_bytes(app_paths.config_dir() / "presets.json",
                           json.dumps(self.user_presets, ensure_ascii=False, indent=1).encode("utf-8"))

    def _load_llm_config(self) -> LlmConfig:
        s = self.settings
        prov = s.value("llm/provider", "none")
        d = DEFAULTS.get(prov, {"endpoint": "", "model": ""})
        return LlmConfig(provider=prov, endpoint=s.value("llm/endpoint", d["endpoint"]),
                         model=s.value("llm/model", d["model"]),
                         include_names=s.value("llm/include_names", "false") == "true",
                         timeout=float(s.value("llm/timeout", 120)))

    def save_llm_config(self, cfg: LlmConfig, key: str | None) -> bool | None:
        s = self.settings
        s.setValue("llm/provider", cfg.provider)
        s.setValue("llm/endpoint", cfg.endpoint)
        s.setValue("llm/model", cfg.model)
        s.setValue("llm/include_names", "true" if cfg.include_names else "false")
        s.sync()
        self.llm_config = cfg
        stored = None
        if key:
            stored = self.secrets.set(self.secrets.account(cfg.provider, cfg.endpoint), key)
        return stored

    def llm(self) -> LlmAdapter:
        return LlmAdapter(self.llm_config, self.secrets)

    # ---------------------------------------------------------------- engine
    def _on_engine(self, result: Result) -> None:
        self.last_result = result
        if result.applied:
            if result.select:
                self._select_object(result.select)
            self._prune_selection()
            self.schedule_preview()
        self.projectChanged.emit(result)

    def apply(self, plan: dict, confirm_stale: bool = False, accept_proposal: bool = False) -> Result:
        return self.engine.apply(plan, confirm_stale=confirm_stale, accept_proposal=accept_proposal)

    def ui_ops(self, ops: list[dict], text: str = "") -> Result:
        r = self.apply(make_plan(ops, self.project.revision, origin="ui", text=text))
        if not r.ok:
            self.message.emit("error", "\n".join(r.errors))
        return r

    def undo(self) -> Result:
        r = self.engine.undo()
        if not r.ok:
            self.message.emit("info", "Нечего отменять.")
        return r

    def redo(self) -> Result:
        r = self.engine.redo()
        if not r.ok:
            self.message.emit("info", "Нечего повторять.")
        return r

    def parse_local(self, text: str) -> dict:
        self.selection.playhead = self.playback.position
        return parse(text, ParseContext(self.project, self.selection))

    # ---------------------------------------------------------------- selection
    def _select_object(self, oid: str) -> None:
        p = self.project
        if p.any_transition(oid):
            self.selection.transition_id = oid
            x = p.space_transition(oid)
            if x:
                self.selection.track_id = x.track_id
        elif p.clip(oid):
            self.selection.clip_ids = [oid]
            t = p.clip_track(oid)
            self.selection.track_id = t.id if t else None
        elif p.track(oid):
            self.selection.track_id = oid
        self.selectionChanged.emit()

    def _prune_selection(self) -> None:
        p = self.project
        s = self.selection
        if s.track_id and not p.track(s.track_id):
            s.track_id = None
        s.clip_ids = [c for c in s.clip_ids if p.clip(c)]
        if s.transition_id and not p.any_transition(s.transition_id):
            s.transition_id = None
        if s.track_id is None and p.tracks:
            s.track_id = p.tracks[0].id

    def select(self, track_id=..., clip_ids=..., transition_id=..., time_range=...) -> None:
        s = self.selection
        if track_id is not ...:
            s.track_id = track_id
        if clip_ids is not ...:
            s.clip_ids = list(clip_ids or [])
        if transition_id is not ...:
            s.transition_id = transition_id
        if time_range is not ...:
            s.time_range = time_range
            if time_range:
                self.playback.set_loop(*time_range) if self.playback.loop is not None else None
        self.selectionChanged.emit()

    # ---------------------------------------------------------------- project files
    def _reset_audio_rate(self, sr: int) -> None:
        if self.store.sample_rate != sr:
            self.store.sample_rate = sr
            self.store.forget()
            self.cache.clear()
            self.playback.close()
            self.playback.sr = sr
            self.playback.buffers.clear()
            self.open_audio()

    def new_project(self, sample_rate: int = 48000, name: str = "Новый проект") -> None:
        self.playback.pause()
        self._reset_audio_rate(sample_rate)
        self.playback.buffers.clear()
        self.path = None
        self.frozen.data_dir = None
        self.store.project_dir = None
        self.selection = Selection()
        p = Project(name=name, sample_rate=sample_rate)
        self.engine.reset(p)
        self._autosaved_rev = -1
        self.pathChanged.emit()
        self.selectionChanged.emit()
        self.schedule_preview()

    def open_project(self, path: str | os.PathLike, recovered_from: str | None = None) -> Project:
        p = load_project(path)
        self.playback.pause()
        self._reset_audio_rate(p.sample_rate)
        self.playback.buffers.clear()
        real = Path(recovered_from) if recovered_from else Path(path)
        self.path = real if (recovered_from or Path(path).suffix == ".asiko") else None
        if recovered_from is None and Path(path).parent == app_paths.autosave_dir():
            self.path = None
        self.store.project_dir = self.path.parent if self.path else None
        self.frozen.data_dir = data_dir_for(self.path) if self.path else None
        self.selection = Selection()
        self.engine.reset(p)
        if recovered_from is not None or self.path is None:
            self.engine.dirty = True
        self._prune_selection()
        self.pathChanged.emit()
        self.selectionChanged.emit()
        self.schedule_preview()
        if self.path:
            self._add_recent(str(self.path))
        return p

    def save(self, path: str | os.PathLike | None = None) -> Path:
        target = Path(path) if path else self.path
        if target is None:
            raise ValueError("Не указан путь сохранения")
        if target.suffix.lower() != ".asiko":
            target = target.with_suffix(".asiko")
        old_data = data_dir_for(self.path) if self.path else None
        save_project(self.project, target)
        new_data = data_dir_for(target)
        if old_data and old_data != new_data and old_data.exists() and self.project.frozen:
            import shutil

            shutil.copytree(old_data, new_data, dirs_exist_ok=True)
        self.path = target
        self.store.project_dir = target.parent
        self.frozen.data_dir = new_data
        self.engine.dirty = False
        self.autosaver.clear(self.project.id)
        self._add_recent(str(target))
        self.pathChanged.emit()
        return target

    def autosave(self) -> Path | None:
        if not self.engine.dirty or self.project.revision == self._autosaved_rev:
            return None
        try:
            p = self.autosaver.write(self.project, str(self.path) if self.path else None)
            self._autosaved_rev = self.project.revision
            return p
        except OSError as e:
            self.message.emit("warn", f"Автосохранение не удалось: {e}")
            return None

    def recent(self) -> list[str]:
        v = self.settings.value("recent", [])
        if isinstance(v, str):
            v = [v]
        return [x for x in (v or []) if Path(x).exists()]

    def _add_recent(self, path: str) -> None:
        r = [x for x in self.recent() if x != path]
        r.insert(0, path)
        self.settings.setValue("recent", r[:10])

    # ---------------------------------------------------------------- import & analysis
    def import_files(self, parent, paths: list[str], track_id: str | None = None, at: float | None = None,
                     on_done=None) -> None:
        paths = [str(p) for p in paths]
        store = self.store

        def work(progress, cancel):
            out, errors = [], []
            for i, pth in enumerate(paths):
                if cancel():
                    break
                try:
                    src = store.import_file(pth, cancel=cancel,
                                            progress=lambda f, m="": progress((i + f) / len(paths), f"{Path(pth).name}: {m}"))
                    out.append(src)
                except Exception as e:
                    from ..audio.io import Cancelled

                    if isinstance(e, Cancelled):
                        raise
                    errors.append(f"{Path(pth).name}: {e}")
            return out, errors

        def done(res):
            sources, errors = res
            for e in errors:
                self.message.emit("error", e)
            if not sources:
                return
            ops = []
            existing = {s.sha256: s for s in self.project.sources}
            created_tracks = 0
            for k, src in enumerate(sources):
                sid = src.id
                if src.sha256 in existing:
                    sid = existing[src.sha256].id
                else:
                    ops.append(op("source.add", params={"source": src.to_dict()}))
                    existing[src.sha256] = src
                tgt = {"source_id": sid}
                if track_id and k == 0:
                    tgt["track_id"] = track_id
                time = {"coord": "timeline", "unit": "s", "start": float(at if at is not None else 0.0)}
                ops.append(op("clip.add", target=tgt, time=time))
                created_tracks += 1
            r = self.ui_ops(ops, text="Импорт: " + ", ".join(Path(p).name for p in paths))
            if r.ok and r.created:
                clips = [c for c in r.created if c.startswith("clp_")]
                if clips:
                    self.select(clip_ids=clips[-1:], track_id=self.project.clip_track(clips[-1]).id)
            if on_done:
                on_done(r)

        run_job(parent, "Импорт аудио…", work, on_done=done,
                on_error=lambda m: self.message.emit("error", f"Импорт не удался: {m}"),
                on_cancel=lambda: self.message.emit("info", "Импорт отменён."))

    def analyze_sources(self, parent, source_ids: list[str], on_done=None, force: bool = False) -> None:
        srcs = [self.project.source(s) for s in source_ids]
        srcs = [s for s in srcs if s is not None and (force or s.beats is None)]
        if not srcs:
            if on_done:
                on_done(True)
            return
        app_paths.setup_numba_cache()
        store = self.store

        def work(progress, cancel):
            res = {}
            for i, s in enumerate(srcs):
                audio = store.load(s, cancel=cancel)
                if audio is None:
                    raise FileNotFoundError(f"Исходник «{s.name}» недоступен.")
                g = analyze_beats(np.asarray(audio), store.sample_rate, cancel=cancel,
                                  progress=lambda f, m="": progress((i + f) / len(srcs), f"{s.name}: {m}"))
                res[s.id] = g
            return res

        def done(res):
            from dataclasses import asdict

            ops = [op("beats.set", target={"source_id": sid}, params={"grid": asdict(g)}) for sid, g in res.items()]
            r = self.ui_ops(ops, text="Анализ темпа и долей")
            if on_done:
                on_done(r.ok)

        run_job(parent, "Анализ темпа и долей (первый запуск дольше — компиляция анализатора)…", work,
                on_done=done, on_error=lambda m: (self.message.emit("error", f"Анализ не удался: {m}"),
                                                  on_done(False) if on_done else None),
                on_cancel=lambda: (self.message.emit("info", "Анализ отменён."), on_done(None) if on_done else None))

    # ---------------------------------------------------------------- preview & playback
    def schedule_preview(self) -> None:
        self._debounce.start()

    def preview_current(self, name: str = "result") -> bool:
        """Буфер прослушивания отрисован для текущей ревизии проекта."""
        return (not self._debounce.isActive() and self.rendered_revision.get(name) == self.project.revision
                and self.preview.status.get(name, 0) >= 1.0)

    def _do_preview(self) -> None:
        p = self.project
        self.preview.request(PreviewRequest("result", p))
        modes = self.compare_modes()
        for name in list(self.playback.buffers):
            if name not in ("result", "__swap__") and name not in [m for m, _ in modes]:
                self.preview.drop(name)
        if self.compare_mode != "result":
            self._request_compare(self.compare_mode)

    def _request_compare(self, mode: str) -> None:
        p = self.project
        if mode == "bypass":
            self.preview.request(PreviewRequest("bypass", p, bypass=True))
        elif mode.startswith("variant:"):
            _, mid, vid = mode.split(":", 2)
            mt = p.music_transition(mid)
            v = mt.variant(vid) if mt else None
            if v is not None:
                self.preview.request(PreviewRequest(mode, p, music_override={mid: v.params}))

    def compare_modes(self) -> list[tuple[str, str]]:
        out = [("result", "Результат"), ("bypass", "Исходник (без обработки переходов)")]
        sel = self.selection.transition_id
        mt = self.project.music_transition(sel) if sel else None
        if mt is None and len(self.project.music_transitions) == 1:
            mt = self.project.music_transitions[0]
        if mt is not None:
            for i, v in enumerate(mt.variants):
                out.append((f"variant:{mt.id}:{v.id}", f"Вариант {i + 1}: {v.name}"))
        return out

    def prepare_compare(self) -> None:
        """Заранее рендерит буферы сравнения (исходник и варианты), не переключая прослушивание."""
        for mode, _ in self.compare_modes():
            if mode != "result" and mode not in self.playback.buffers:
                self._request_compare(mode)

    def set_compare(self, mode: str) -> None:
        self.compare_mode = mode
        if mode != "result" and mode not in self.playback.buffers:
            self._request_compare(mode)
        self.playback.set_active(mode)
        self.update_loudness_match()

    def update_loudness_match(self) -> None:
        region = self.compare_region or self.selection.time_range
        if region is None:
            x = self.project.any_transition(self.selection.transition_id) if self.selection.transition_id else None
            if x is not None:
                a, b = self.transition_bounds(x)
                region = (max(0.0, a - 2), b + 2)
        if region is None:
            region = (0.0, max(1.0, self.project.content_end()))
        bufs = {k: v for k, v in self.playback.buffers.items() if k != "__swap__" and v.complete()}
        if len(bufs) >= 2:
            self.playback.monitor_gain_db = loudness_offsets(bufs, self.playback.sr, *region)

    def _preview_finished(self, name: str, payload) -> None:
        buf, info = payload
        self.rendered_revision[name] = info.get("revision")
        if name == "result":
            n = buf.n
            self.peak_env = peak_envelope(buf.data[:n], self.playback.sr, 0.05)
            self.master_peak = float(info.get("peak", 0.0))
            self.boundary_report = frozen_boundary_report(info["plan"], self.cache) if info["plan"].frozen else []
            if self.frozen.errors:
                for e in self.frozen.errors.values():
                    self.message.emit("warn", e)
        if self.playback.loudness_match:
            self.update_loudness_match()
        self.previewDone.emit(name, info)

    def transition_bounds(self, x) -> tuple[float, float]:
        if hasattr(x, "params"):
            return x.params.overlap
        return x.start, x.end

    def play(self) -> None:
        self.playback.play()

    def play_transition(self, xid: str | None = None, context: float = 3.0) -> bool:
        xid = xid or self.selection.transition_id
        x = self.project.any_transition(xid) if xid else None
        if x is None:
            self.message.emit("info", "Выберите переход для прослушивания.")
            return False
        a, b = self.transition_bounds(x)
        start = max(0.0, a - context)
        end = b + context
        self.compare_region = (start, end)
        self.playback.play(start=start, stop_at=None if self.playback.loop else end)
        return True

    def toggle_loop(self, on: bool) -> None:
        if on:
            rng = self.selection.time_range or self.compare_region
            if rng is None:
                x = self.project.any_transition(self.selection.transition_id) if self.selection.transition_id else None
                if x is not None:
                    a, b = self.transition_bounds(x)
                    rng = (max(0.0, a - 3), b + 3)
            if rng is None:
                self.message.emit("info", "Выделите участок на линейке, чтобы зациклить его.")
                self.playback.set_loop(None, None)
                return
            self.playback.set_loop(*rng)
            if not (rng[0] <= self.playback.position < rng[1]):
                self.playback.seek(rng[0])
        else:
            self.playback.set_loop(None, None)

    # ---------------------------------------------------------------- export & freeze
    def export(self, parent, opts: ExportOptions, on_done=None) -> None:
        project = self.project.clone()
        store, cache, frozen = self.store, self.cache, self.frozen

        def work(progress, cancel):
            return export_wav(project, store, opts, cache=cache, progress=progress, cancel=cancel, frozen_audio=frozen)

        def done(rep):
            if on_done:
                on_done(rep)

        run_job(parent, f"Экспорт: {Path(opts.path).name}", work, on_done=done,
                on_error=lambda m: self.message.emit("error", f"Экспорт не выполнен: {m}. Файл не создан."),
                on_cancel=lambda: self.message.emit("info", "Экспорт отменён; незавершённый файл удалён."))

    def freeze_selection(self, parent, on_done=None) -> None:
        rng = self.selection.time_range
        if not rng:
            self.message.emit("info", "Выделите участок на линейке.")
            return
        if self.path is None:
            self.message.emit("warn", "Сначала сохраните проект: зафиксированный PCM хранится рядом с файлом проекта.")
            return
        project = self.project.clone()
        data_dir = data_dir_for(self.path)

        def work(progress, cancel):
            return freeze_params(project, self.store, data_dir, rng[0], rng[1], self.cache,
                                 frozen_audio=self.frozen, cancel=cancel, progress=progress)

        def done(params):
            r = self.ui_ops([op("freeze.add", params=params)], text="Фиксация участка в PCM")
            if on_done:
                on_done(r)

        run_job(parent, "Фиксация участка в PCM…", work, on_done=done,
                on_error=lambda m: self.message.emit("error", f"Фиксация не выполнена: {m}"))

    # ---------------------------------------------------------------- info
    def status_text(self) -> str:
        p = self.project
        name = self.path.name if self.path else "не сохранён"
        dirty = " •" if self.engine.dirty else ""
        return f"{p.name} ({name}){dirty} · ревизия {p.revision} · {p.sample_rate // 1000 if p.sample_rate % 1000 == 0 else p.sample_rate / 1000} кГц"

    def content_seconds(self) -> float:
        return max(self.project.content_end(), 1.0)

    def latency_note(self) -> str:
        sr = self.project.sample_rate
        return (f"Задержка КИХ-фильтров {dsp.taps_half(sr) / sr * 1000:.1f} мс компенсирована; хвост "
                f"реверберации до {dsp.max_ir_length(sr) / sr:.1f} с учитывается при рендере и экспорте.")

    def describe_selection_time(self) -> str:
        r = self.selection.time_range
        return f"{seconds(r[0])}–{seconds(r[1])}" if r else ""
