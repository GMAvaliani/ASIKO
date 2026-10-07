"""Воспроизведение отрендеренных буферов (sounddevice / PortAudio).

Буферы содержат результат того же рендера, что и экспорт. Здесь применяются только
средства мониторинга: громкость прослушивания и выравнивание громкости для
сравнения (не попадают в экспорт). Переключение сравниваемых буферов — с
перекрёстным затуханием 20 мс без потери позиции.

Если аудиовыход недоступен, работает «тихий» режим: позиция идёт в реальном
времени, интерфейс и видео синхронизируются, звука нет.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np

from .render import CHUNK

XFADE_S = 0.02
LOOP_DECLICK_S = 0.003


def import_sounddevice():
    """Импорт sounddevice; в упакованной Linux-сборке — с вложенной libportaudio."""
    import sys
    from pathlib import Path

    if "sounddevice" not in sys.modules and getattr(sys, "frozen", False) and sys.platform.startswith("linux"):
        import ctypes.util

        bundled = Path(getattr(sys, "_MEIPASS", "")) / "libportaudio.so.2"
        if bundled.exists():
            orig = ctypes.util.find_library
            ctypes.util.find_library = lambda name: str(bundled) if name == "portaudio" else orig(name)
            try:
                import sounddevice
            finally:
                ctypes.util.find_library = orig
            return sounddevice
    import sounddevice

    return sounddevice


class MonitorBuffer:
    """Буфер мастера на шкале проекта, заполняемый фрагментами по мере рендера."""

    def __init__(self, n: int):
        self.n = int(n)
        nch = max(1, (self.n + CHUNK - 1) // CHUNK)
        self.data = np.zeros((nch * CHUNK, 2), np.float32)
        self.ready = np.zeros(nch, dtype=bool)
        self.keys: list[str | None] = [None] * nch
        self.peaks = np.zeros(nch, np.float32)

    @property
    def nchunks(self) -> int:
        return self.ready.shape[0]

    def complete(self) -> bool:
        return bool(self.ready.all())

    def put(self, k: int, y: np.ndarray, key: str | None = None) -> None:
        self.data[k * CHUNK:(k + 1) * CHUNK] = y
        self.keys[k] = key
        self.peaks[k] = float(np.max(np.abs(y))) if y.size else 0.0
        self.ready[k] = True

    def read(self, pos: int, n: int) -> np.ndarray:
        out = np.zeros((n, 2), np.float32)
        if pos >= self.n or n <= 0:
            return out
        end = min(pos + n, self.n)
        k0, k1 = pos // CHUNK, (end - 1) // CHUNK
        for k in range(k0, k1 + 1):
            if not self.ready[k]:
                continue
            a = max(pos, k * CHUNK)
            b = min(end, (k + 1) * CHUNK)
            out[a - pos:b - pos] = self.data[a:b]
        return out

    def ready_fraction(self) -> float:
        return float(self.ready.mean()) if self.ready.size else 1.0


class PlaybackEngine:
    def __init__(self, sr: int = 48000):
        self.sr = sr
        self._lock = threading.RLock()
        self.buffers: dict[str, MonitorBuffer] = {}
        self.active = "result"
        self._prev: str | None = None
        self._xf = 0
        self.pos = 0
        self.playing = False
        self.loop: tuple[int, int] | None = None
        self.stop_at: int | None = None
        self.play_start = 0
        self.monitor_gain_db: dict[str, float] = {}
        self.loudness_match = False
        self.volume = 1.0
        self.peak = np.zeros(2, np.float32)
        self.clip = np.zeros(2, dtype=bool)
        self.correlation = 1.0
        self.backend = "none"
        self.device_name = ""
        self.error = ""
        self._stream = None
        self._null_thread = None
        self._null_stop = threading.Event()
        self.on_stop = None

    # ---------------------------------------------------------------- device
    def open(self, device=None, blocksize: int = 1024) -> str:
        self.close()
        try:
            sd = import_sounddevice()
            self._stream = sd.OutputStream(samplerate=self.sr, channels=2, dtype="float32",
                                           callback=self._callback, device=device, blocksize=blocksize,
                                           latency="high")
            self._stream.start()
            info = sd.query_devices(self._stream.device)
            self.device_name = info.get("name", "") if isinstance(info, dict) else str(info)
            self.backend = "device"
            self.error = ""
        except Exception as e:  # нет PortAudio, нет устройства, неподдерживаемая частота
            self._stream = None
            self.backend = "null"
            self.error = f"Аудиовыход недоступен ({e}). Воспроизведение идёт без звука."
            self._start_null(blocksize)
        return self.backend

    def _start_null(self, blocksize: int) -> None:
        self._null_stop.clear()

        def run():
            t_next = time.perf_counter()
            dt = blocksize / self.sr
            while not self._null_stop.is_set():
                self.pull(blocksize)
                t_next += dt
                delay = t_next - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.perf_counter()

        self._null_thread = threading.Thread(target=run, name="asiko-null-audio", daemon=True)
        self._null_thread.start()

    def close(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._null_thread is not None:
            self._null_stop.set()
            self._null_thread.join(timeout=1.0)
            self._null_thread = None
        self.backend = "none"

    def _callback(self, outdata, frames, _time, _status):
        outdata[:] = self.pull(frames)

    # ---------------------------------------------------------------- control
    def set_buffer(self, name: str, buf: MonitorBuffer) -> None:
        with self._lock:
            self.buffers[name] = buf

    def swap_buffer(self, name: str, buf: MonitorBuffer) -> None:
        """Замена буфера (после правки) с перекрёстным затуханием, если он сейчас звучит."""
        with self._lock:
            old = self.buffers.get(name)
            self.buffers[name] = buf
            if old is not None and name == self.active and self.playing:
                self.buffers["__swap__"] = old
                self._prev = "__swap__"
                self._xf = int(XFADE_S * self.sr)

    def set_active(self, name: str) -> None:
        with self._lock:
            if name == self.active:
                return
            self._prev = self.active
            self.active = name
            self._xf = int(XFADE_S * self.sr)

    def play(self, start: float | None = None, stop_at: float | None = None) -> None:
        with self._lock:
            if start is not None:
                self.pos = max(0, int(round(start * self.sr)))
            self.play_start = self.pos
            self.stop_at = None if stop_at is None else int(round(stop_at * self.sr))
            self.playing = True

    def pause(self) -> None:
        with self._lock:
            self.playing = False

    def stop(self) -> None:
        with self._lock:
            self.playing = False
            self.pos = self.loop[0] if self.loop else self.play_start

    def seek(self, t: float) -> None:
        with self._lock:
            self.pos = max(0, int(round(t * self.sr)))

    def set_loop(self, a: float | None, b: float | None) -> None:
        with self._lock:
            if a is None or b is None or b - a < 0.05:
                self.loop = None
            else:
                self.loop = (int(round(a * self.sr)), int(round(b * self.sr)))

    @property
    def position(self) -> float:
        return self.pos / self.sr

    def _gain(self, name: str) -> float:
        g = self.volume
        if self.loudness_match:
            g *= 10 ** (self.monitor_gain_db.get(name, 0.0) / 20)
        return g

    def _read(self, name: str | None, pos: int, n: int) -> np.ndarray:
        buf = self.buffers.get(name) if name else None
        if buf is None:
            return np.zeros((n, 2), np.float32)
        return buf.read(pos, n)

    # ---------------------------------------------------------------- core
    def pull(self, frames: int) -> np.ndarray:
        """Следующие frames сэмплов мониторинга (вызывается из аудиопотока)."""
        out = np.zeros((frames, 2), np.float32)
        raw_peak = np.zeros(2, np.float32)
        with self._lock:
            if not self.playing:
                self.peak *= 0.0
                return out
            o = 0
            while o < frames:
                buf = self.buffers.get(self.active)
                length = buf.n if buf is not None else 0
                if self.loop:
                    bound = self.loop[1]
                elif self.stop_at is not None:
                    bound = min(self.stop_at, length)
                else:
                    bound = length
                n = min(frames - o, bound - self.pos)
                if n <= 0:
                    if self.loop and self.loop[1] > self.loop[0]:
                        self.pos = self.loop[0]
                        continue
                    self.playing = False
                    cb = self.on_stop
                    if cb:
                        try:
                            cb()
                        except Exception:
                            pass
                    break
                seg = self._read(self.active, self.pos, n)
                raw_peak = np.maximum(raw_peak, np.max(np.abs(seg), axis=0) if n else raw_peak)
                seg = seg * self._gain(self.active)
                if self._xf > 0 and self._prev is not None:
                    total = int(XFADE_S * self.sr)
                    m = min(n, self._xf)
                    done = total - self._xf
                    w = (done + np.arange(m) + 1) / total
                    prev = self._read(self._prev, self.pos, m) * self._gain(self._prev)
                    seg[:m] = seg[:m] * w[:, None] + prev * (1 - w[:, None])
                    self._xf -= m
                if self.loop:
                    d = int(LOOP_DECLICK_S * self.sr)
                    a, b = self.loop
                    idx = self.pos + np.arange(n)
                    g = np.ones(n)
                    m1 = idx >= b - d
                    g[m1] = np.clip((b - idx[m1]) / d, 0, 1)
                    m2 = idx < a + d
                    g[m2] = np.minimum(g[m2], np.clip((idx[m2] - a + 1) / d, 0, 1))
                    seg *= g[:, None]
                out[o:o + n] = seg
                self.pos += n
                o += n
            self.peak = np.maximum(self.peak * 0.0, raw_peak)
            self.clip |= raw_peak >= 1.0
            l, r = out[:, 0].astype(np.float64), out[:, 1].astype(np.float64)
            den = math.sqrt(float(np.dot(l, l)) * float(np.dot(r, r)))
            if den > 1e-12:
                self.correlation = float(np.dot(l, r)) / den
        return out

    def reset_clip(self) -> None:
        with self._lock:
            self.clip[:] = False
