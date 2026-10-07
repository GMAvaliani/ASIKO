"""Фоновое предварительное отрисовывание буферов прослушивания.

Использует ту же функцию рендера фрагмента, что и экспорт. Фрагменты, ключ которых
не изменился, копируются из прежнего буфера; изменившиеся рендерятся в порядке
близости к курсору воспроизведения. Новый запрос отменяет устаревший.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass

import numpy as np

from .analysis import integrated_lufs
from .playback import MonitorBuffer, PlaybackEngine
from .render import CHUNK, ChunkCache, compile_plan, master_chunk_key, render_master_chunk

EXTRA_TAIL_S = 0.5


@dataclass
class PreviewRequest:
    name: str
    project: object
    bypass: bool = False
    music_override: dict | None = None
    focus: float = 0.0


class PreviewRenderer:
    def __init__(self, store, cache: ChunkCache, engine: PlaybackEngine, frozen_audio=None):
        self.store = store
        self.cache = cache
        self.engine = engine
        self.frozen_audio = frozen_audio
        self._lock = threading.Lock()
        self._pending: dict[str, PreviewRequest] = {}
        self._wake = threading.Event()
        self._cancel_gen: dict[str, int] = {}
        self._stop = False
        self.status: dict[str, float] = {}
        self.errors: dict[str, str] = {}
        self.on_progress = None   # (name, fraction)
        self.on_done = None       # (name, buffer, info)
        self.thread = threading.Thread(target=self._run, name="asiko-preview", daemon=True)
        self.thread.start()

    def request(self, req: PreviewRequest) -> None:
        with self._lock:
            self._pending[req.name] = req
            self._cancel_gen[req.name] = self._cancel_gen.get(req.name, 0) + 1
        self._wake.set()

    def drop(self, name: str) -> None:
        with self._lock:
            self._pending.pop(name, None)
            self._cancel_gen[name] = self._cancel_gen.get(name, 0) + 1
            self.engine.buffers.pop(name, None)

    def busy(self) -> bool:
        with self._lock:
            return bool(self._pending)

    def shutdown(self) -> None:
        self._stop = True
        self._wake.set()

    def _run(self) -> None:
        while not self._stop:
            self._wake.wait(0.5)
            self._wake.clear()
            while True:
                with self._lock:
                    if not self._pending:
                        break
                    name = "result" if "result" in self._pending else next(iter(self._pending))
                    req = self._pending.pop(name)
                    gen = self._cancel_gen.get(name, 0)
                try:
                    self._render(req, gen)
                except Exception as e:  # ошибка рендера не должна ронять приложение
                    self.errors[name] = f"{e}"
                    traceback.print_exc()

    def _cancelled(self, name: str, gen: int) -> bool:
        return self._stop or self._cancel_gen.get(name, 0) != gen

    def _render(self, req: PreviewRequest, gen: int) -> None:
        plan = compile_plan(req.project, self.store, bypass=req.bypass, music_override=req.music_override,
                            frozen_audio=self.frozen_audio)
        sr = plan.sr
        n = max(plan.end_with_tails(), plan.content_end) + int(EXTRA_TAIL_S * sr)
        buf = MonitorBuffer(max(n, sr))
        old = self.engine.buffers.get(req.name)
        keys = [master_chunk_key(plan, k) for k in range(buf.nchunks)]
        if old is not None:
            for k in range(min(buf.nchunks, old.nchunks)):
                if old.ready[k] and old.keys[k] == keys[k]:
                    buf.put(k, old.data[k * CHUNK:(k + 1) * CHUNK], keys[k])
        focus_k = min(buf.nchunks - 1, max(0, int(self.engine.pos // CHUNK)))
        order = sorted(range(buf.nchunks), key=lambda k: (k < focus_k, abs(k - focus_k)))
        swapped = False
        if old is None or buf.ready[focus_k]:
            self.engine.swap_buffer(req.name, buf)
            swapped = True
        todo = [k for k in order if not buf.ready[k]]
        total = max(1, len(todo))
        t0 = time.perf_counter()
        for i, k in enumerate(todo):
            if self._cancelled(req.name, gen):
                return
            y = render_master_chunk(plan, k, self.cache)
            buf.put(k, y, keys[k])
            if not swapped and buf.ready[focus_k]:
                # как только готов фрагмент под курсором — переключаемся на новый буфер
                self.engine.swap_buffer(req.name, buf)
                swapped = True
            self.status[req.name] = (i + 1) / total
            if self.on_progress:
                self.on_progress(req.name, (i + 1) / total)
        if self._cancelled(req.name, gen):
            return
        if not swapped:
            self.engine.swap_buffer(req.name, buf)
        self.status[req.name] = 1.0
        info = {"seconds": time.perf_counter() - t0, "chunks": len(todo), "plan": plan,
                "revision": getattr(req.project, "revision", None),
                "peak": float(buf.peaks.max()) if buf.peaks.size else 0.0}
        if self.on_done:
            self.on_done(req.name, buf, info)


def loudness_offsets(buffers: dict[str, MonitorBuffer], sr: int, a: float, b: float, ref: str = "result") -> dict[str, float]:
    """Поправки громкости (дБ) для сравнения на участке [a, b] — только мониторинг."""
    vals = {}
    for name, buf in buffers.items():
        s0, s1 = int(a * sr), int(b * sr)
        y = buf.read(s0, max(0, s1 - s0))
        vals[name] = integrated_lufs(y, sr)
    base = vals.get(ref)
    out = {}
    for name, v in vals.items():
        if base is None or not np.isfinite(base) or not np.isfinite(v):
            out[name] = 0.0
        else:
            out[name] = float(np.clip(base - v, -24.0, 24.0))
    return out
