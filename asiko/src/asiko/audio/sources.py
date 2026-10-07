"""Хранилище исходников: импорт, явный ресемплинг (soxr VHQ), кэш, волновые формы.

Исходные файлы только читаются. Декодированный и приведённый к частоте проекта
материал хранится в кэше как float32 .npy и открывается через memmap.
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soxr

from ..model.project import Source, new_id
from .io import AudioImportError, Cancelled, probe, read_float, sha256_file

RESAMPLER = "soxr-VHQ"
RESAMPLER_VERSION = f"{RESAMPLER}-{soxr.__version__}"
PEAK_BLOCK = 256


def resample(data: np.ndarray, in_sr: int, out_sr: int, cancel=None, progress=None) -> np.ndarray:
    """Явный качественный ресемплинг (soxr VHQ) с точной итоговой длиной."""
    if in_sr == out_sr:
        return np.ascontiguousarray(data, dtype=np.float32)
    ch = data.shape[1]
    rs = soxr.ResampleStream(in_sr, out_sr, ch, dtype="float32", quality="VHQ")
    out = []
    step = 1 << 17
    n = data.shape[0]
    for i in range(0, n, step):
        if cancel is not None and cancel():
            raise Cancelled()
        chunk = np.ascontiguousarray(data[i:i + step], dtype=np.float32)
        last = i + step >= n
        out.append(rs.resample_chunk(chunk, last=last))
        if progress:
            progress(min(1.0, (i + step) / n))
    y = np.concatenate(out, axis=0) if out else np.zeros((0, ch), np.float32)
    expected = int(round(n * out_sr / in_sr))
    if y.shape[0] > expected:
        y = y[:expected]
    elif y.shape[0] < expected:
        y = np.concatenate([y, np.zeros((expected - y.shape[0], ch), np.float32)], axis=0)
    return y


@dataclass
class Peaks:
    """Пирамида min/max для отрисовки волны. levels[k]: (blocks, channels, 2), блок = 256·4^k сэмплов."""

    levels: list[np.ndarray]
    sample_rate: int
    frames: int

    def level_for(self, samples_per_px: float) -> tuple[int, np.ndarray]:
        k = 0
        block = PEAK_BLOCK
        while k + 1 < len(self.levels) and block * 4 <= samples_per_px:
            block *= 4
            k += 1
        return block, self.levels[k]


def compute_peaks(audio: np.ndarray, sr: int) -> Peaks:
    n, ch = audio.shape
    nb = max(1, (n + PEAK_BLOCK - 1) // PEAK_BLOCK)
    pad = nb * PEAK_BLOCK - n
    a = np.asarray(audio, dtype=np.float32)
    if pad:
        a = np.concatenate([a, np.zeros((pad, ch), np.float32)], axis=0)
    r = a.reshape(nb, PEAK_BLOCK, ch)
    lv = np.stack([r.min(axis=1), r.max(axis=1)], axis=-1)
    levels = [lv]
    while levels[-1].shape[0] > 4:
        cur = levels[-1]
        m = cur.shape[0] // 4
        if m == 0:
            break
        cur = cur[: m * 4].reshape(m, 4, ch, 2)
        levels.append(np.stack([cur[..., 0].min(axis=1), cur[..., 1].max(axis=1)], axis=-1))
    return Peaks(levels=levels, sample_rate=sr, frames=n)


class SourceStore:
    def __init__(self, cache_dir: str | os.PathLike, sample_rate: int = 48000):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self._audio: dict[str, np.ndarray] = {}
        self._peaks: dict[str, Peaks] = {}
        self._lock = threading.RLock()
        self.project_dir: Path | None = None
        self.missing: set[str] = set()

    # ---------------------------------------------------------------- paths
    def resolve_path(self, src: Source) -> str | None:
        cands = []
        if src.path:
            cands.append(Path(src.path))
        if src.rel_path and self.project_dir is not None:
            cands.append((self.project_dir / src.rel_path).resolve())
        for c in cands:
            try:
                if c.is_file():
                    return str(c)
            except OSError:
                continue
        return None

    def _key(self, sha: str, sr: int) -> str:
        return f"{sha[:24]}_{sr}_{RESAMPLER_VERSION}"

    def _cache_path(self, sha: str, sr: int) -> Path:
        return self.cache_dir / f"{self._key(sha, sr)}.f32.npy"

    def key_for(self, src: Source) -> str:
        return self._key(src.sha256, self.sample_rate)

    # ---------------------------------------------------------------- import
    def import_file(self, path: str | os.PathLike, cancel=None, progress=None) -> Source:
        info = probe(path)
        if progress:
            progress(0.02, "Контрольная сумма…")
        sha = sha256_file(info.path, cancel=cancel)
        src = Source(id=new_id("src"), name=info.name, path=os.path.abspath(info.path), sha256=sha, size=info.size,
                     sample_rate=info.sample_rate, channels=info.channels, frames=info.frames,
                     format=info.format, subtype=info.subtype)
        self.load(src, cancel=cancel, progress=(lambda f: progress(0.05 + 0.9 * f, "Декодирование и ресемплинг…"))
                  if progress else None)
        if progress:
            progress(1.0, "Готово")
        return src

    def load(self, src: Source, cancel=None, progress=None) -> np.ndarray | None:
        """Аудио исходника на частоте проекта (memmap float32), None — файл недоступен."""
        key = self.key_for(src)
        with self._lock:
            if key in self._audio:
                return self._audio[key]
        cp = self._cache_path(src.sha256, self.sample_rate)
        if cp.exists():
            try:
                arr = np.load(cp, mmap_mode="r")
                with self._lock:
                    self._audio[key] = arr
                    self.missing.discard(src.id)
                return arr
            except (OSError, ValueError):
                try:
                    cp.unlink()
                except OSError:
                    pass
        path = self.resolve_path(src)
        if path is None:
            with self._lock:
                self.missing.add(src.id)
            return None
        data, sr = read_float(path, cancel=cancel, progress=(lambda f: progress(0.5 * f)) if progress else None)
        if data.shape[1] != src.channels:
            raise AudioImportError(f"«{src.name}»: число каналов изменилось ({data.shape[1]} вместо {src.channels}).")
        data = resample(data, sr, self.sample_rate, cancel=cancel,
                        progress=(lambda f: progress(0.5 + 0.45 * f)) if progress else None)
        tmp = cp.with_suffix(".tmp.npy")
        np.save(tmp, data.astype(np.float32))
        os.replace(tmp, cp)
        arr = np.load(cp, mmap_mode="r")
        with self._lock:
            self._audio[key] = arr
            self.missing.discard(src.id)
        return arr

    def cached(self, src: Source) -> np.ndarray | None:
        with self._lock:
            return self._audio.get(self.key_for(src))

    def peaks(self, src: Source) -> Peaks | None:
        key = self.key_for(src)
        with self._lock:
            if key in self._peaks:
                return self._peaks[key]
        audio = self.cached(src)
        if audio is None:
            return None
        pk = compute_peaks(np.asarray(audio), self.sample_rate)
        with self._lock:
            self._peaks[key] = pk
        return pk

    def forget(self) -> None:
        with self._lock:
            self._audio.clear()
            self._peaks.clear()
            self.missing.clear()
