"""Фиксация участка в отрендеренный PCM (защита итоговых сэмплов)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from ..model.project import FrozenRegion, new_id
from .io import sha256_file, write_float_wav
from .render import ChunkCache, compile_plan, render_range

MARGIN_S = 0.02


def freeze_params(project, store, data_dir: str | Path, start: float, end: float, cache: ChunkCache | None = None,
                  frozen_audio=None, cancel=None, progress=None) -> dict:
    """Рендерит мастер участка (с полями для сшивки) и сохраняет float32 WAV.
    Возвращает параметры для операции freeze.add."""
    sr = project.sample_rate
    plan = compile_plan(project, store, frozen_audio=frozen_audio)
    m = int(round(MARGIN_S * sr))
    s0 = int(round(start * sr))
    s1 = int(round(end * sr))
    if s1 <= s0:
        raise ValueError("Пустой участок.")
    if s0 - m < 0:
        m = s0
    y = render_range(plan, s0 - m, s1 + m, cache, progress=progress, cancel=cancel)
    fid = new_id("frz")
    rel = f"frozen/{fid}.wav"
    digest = write_float_wav(Path(data_dir) / rel, y, sr)
    return {"id": fid, "start": start, "end": end, "file": rel, "sha256": digest, "sample_rate": sr,
            "margin": m / sr}


class FrozenLoader:
    """Загружает зафиксированные участки с проверкой контрольной суммы."""

    def __init__(self):
        self.data_dir: Path | None = None
        self._cache: dict[str, np.ndarray] = {}
        self.errors: dict[str, str] = {}

    def __call__(self, fr: FrozenRegion) -> np.ndarray | None:
        if fr.sha256 in self._cache:
            return self._cache[fr.sha256]
        if self.data_dir is None:
            self.errors[fr.id] = "Проект не сохранён — данные фиксации недоступны."
            return None
        p = self.data_dir / fr.file
        if not p.exists():
            self.errors[fr.id] = f"Файл фиксации не найден: {p}"
            return None
        if sha256_file(p) != fr.sha256:
            self.errors[fr.id] = "Файл фиксации изменён (контрольная сумма не совпадает) — участок не применяется."
            return None
        data, sr = sf.read(str(p), dtype="float32", always_2d=True)
        if sr != fr.sample_rate:
            self.errors[fr.id] = "Частота файла фиксации не совпадает."
            return None
        self._cache[fr.sha256] = data
        self.errors.pop(fr.id, None)
        return data
