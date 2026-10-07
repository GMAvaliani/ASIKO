"""Чтение/запись аудио (libsndfile через soundfile), контрольные суммы, атомарная запись."""
from __future__ import annotations

import hashlib
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".aif", ".aiff", ".w64", ".rf64")
REQUIRED_FORMATS = ("WAV", "FLAC", "MP3")
VIDEO_EXTENSIONS = (".mp4", ".mov", ".m4v", ".mkv", ".avi", ".webm", ".wmv", ".mpg", ".mpeg")


class AudioImportError(Exception):
    pass


class Cancelled(Exception):
    """Операция отменена пользователем."""


@dataclass
class AudioInfo:
    path: str
    name: str
    sample_rate: int
    channels: int
    frames: int
    format: str
    subtype: str
    size: int

    @property
    def duration(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def sha256_file(path: str | os.PathLike, cancel=None, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            if cancel is not None and cancel():
                raise Cancelled()
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def probe(path: str | os.PathLike) -> AudioInfo:
    p = Path(path)
    if not p.exists():
        raise AudioImportError(f"Файл не найден: {p}")
    try:
        info = sf.info(str(p))
    except Exception as e:  # libsndfile: формат не распознан
        raise AudioImportError(f"Не удалось прочитать «{p.name}»: формат не поддерживается или файл повреждён ({e}).")
    if info.frames <= 0 or info.samplerate <= 0:
        raise AudioImportError(f"«{p.name}»: пустой файл или неизвестная длительность.")
    if info.channels not in (1, 2):
        raise AudioImportError(f"«{p.name}»: {info.channels} каналов. Поддерживаются моно и стерео.")
    frames = int(info.frames)
    if info.format == "MP3":
        # Для MP3 libsndfile сообщает оценку длины; уточняем фактическим декодированием.
        frames = _count_frames(p)
    return AudioInfo(path=str(p), name=p.stem, sample_rate=int(info.samplerate), channels=int(info.channels),
                     frames=frames, format=info.format, subtype=info.subtype, size=p.stat().st_size)


def _count_frames(p: Path) -> int:
    n = 0
    with sf.SoundFile(str(p)) as f:
        for block in f.blocks(blocksize=1 << 16, dtype="float32", always_2d=True):
            n += block.shape[0]
    return n


def read_float(path: str | os.PathLike, cancel=None, progress=None) -> tuple[np.ndarray, int]:
    """Декодирует весь файл в float32 (frames, channels)."""
    out = []
    with sf.SoundFile(str(path)) as f:
        sr = f.samplerate
        total = max(1, f.frames)
        done = 0
        for block in f.blocks(blocksize=1 << 17, dtype="float32", always_2d=True):
            if cancel is not None and cancel():
                raise Cancelled()
            out.append(block.copy())
            done += block.shape[0]
            if progress:
                progress(min(1.0, done / total))
    if not out:
        raise AudioImportError("Файл не содержит аудиоданных.")
    data = np.concatenate(out, axis=0)
    if not np.all(np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=1.0, neginf=-1.0)
    return data, sr


def atomic_write_bytes(path: str | os.PathLike, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_float_wav(path: str | os.PathLike, data: np.ndarray, sr: int) -> str:
    """Атомарно пишет float32 WAV и возвращает sha256 файла."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp.wav", dir=str(path.parent))
    os.close(fd)
    try:
        sf.write(tmp, np.asarray(data, dtype=np.float32), sr, subtype="FLOAT", format="WAV")
        digest = sha256_file(tmp)
        os.replace(tmp, path)
        return digest
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
