"""Экспорт WAV через ту же модель рендера, что и предпрослушивание.

* Временный файл в той же папке → проверка заголовка и длины → атомарная замена.
  При отмене или ошибке временный файл удаляется: повреждённый файл не остаётся
  под видом готового.
* Квантование явное: округление, ограничение диапазона (с подсчётом перегрузок),
  TPDF-дизеринг — только по выбору пользователя (детерминированный, по номеру сэмпла).
* Нормализация и лимитер не применяются без явного выбора (лимитера в v1 нет).
* Выравнивание громкости для мониторинга в экспорт не попадает (здесь его нет).
"""
from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from .analysis import integrated_lufs, peak_dbfs, true_peak_dbfs
from .io import Cancelled, sha256_file
from .render import CHUNK, ChunkCache, compile_plan, render_range

SUBTYPES = {"PCM_24": "WAV PCM 24 бит", "PCM_16": "WAV PCM 16 бит", "FLOAT": "WAV 32 бит float"}
SAMPLE_RATES = (44100, 48000, 96000)


@dataclass
class ExportOptions:
    path: str
    start: float | None = None          # None — начало проекта
    end: float | None = None            # None — конец проекта
    tails: bool = True                  # включать хвосты эффектов
    sample_rate: int = 48000
    subtype: str = "PCM_24"
    dither: bool = False
    normalize_dbfs: float | None = None
    edge_fade_ms: float = 0.0           # короткие фейды на краях экспортируемого участка
    music_override: dict | None = None  # вариант перехода A→B для экспорта


@dataclass
class ExportReport:
    path: str
    frames: int
    sample_rate: int
    subtype: str
    duration: float
    peak_dbfs: float
    true_peak_dbfs: float
    lufs: float
    clipped_samples: int
    tail_seconds: float
    sha256: str
    normalize_gain_db: float = 0.0
    warnings: list[str] = field(default_factory=list)


def export_range_samples(project, plan, opts: ExportOptions) -> tuple[int, int, int]:
    """(s0, s1, хвост) в сэмплах частоты проекта."""
    sr = plan.sr
    s0 = int(round((opts.start or 0.0) * sr))
    if opts.end is None:
        base_end = plan.content_end
        s1 = plan.end_with_tails() if opts.tails else base_end
        s1 = max(s1, base_end)
        tail = max(0, s1 - base_end)
    else:
        base_end = int(round(opts.end * sr))
        tail = 0
        if opts.tails:
            from . import dsp

            for tp in plan.tracks:
                if not tp.audible:
                    continue
                t = tp.tail_samples(sr)
                for cp in tp.clips:
                    t = max(t, tp.tail_samples(sr) + cp.lookahead(sr)
                            + (dsp.ir_length(cp.tail_room, sr) if cp.tail_room else 0))
                tail = max(tail, t)
        s1 = base_end + tail
    return s0, s1, tail


def quantize(x: np.ndarray, subtype: str, dither: bool, first_index: int) -> tuple[np.ndarray, int]:
    """float → целые PCM (или float32). Возвращает (данные, число ограниченных сэмплов)."""
    if subtype == "FLOAT":
        return x.astype(np.float32), 0
    bits = 24 if subtype == "PCM_24" else 16
    scale = float(2 ** (bits - 1))
    y = np.asarray(x, dtype=np.float64) * scale
    if dither:
        rng = np.random.default_rng([20260, first_index])
        y = y + (rng.random(y.shape) - rng.random(y.shape))
    y = np.round(y)
    lo, hi = -scale, scale - 1
    clipped = int(np.count_nonzero((y < lo) | (y > hi)))
    y = np.clip(y, lo, hi)
    if bits == 24:
        return (y.astype(np.int32) << 8), clipped
    return y.astype(np.int16), clipped


def export_wav(project, store, opts: ExportOptions, cache: ChunkCache | None = None, progress=None,
               cancel=None, frozen_audio=None) -> ExportReport:
    if opts.subtype not in SUBTYPES:
        raise ValueError(f"Неподдерживаемый формат: {opts.subtype}")
    if opts.sample_rate not in SAMPLE_RATES:
        raise ValueError(f"Неподдерживаемая частота: {opts.sample_rate}")
    region = opts.end is not None
    plan = compile_plan(project, store, music_override=opts.music_override,
                        gate_end_s=opts.end if (region and opts.tails) else None, frozen_audio=frozen_audio)
    warnings = []
    if plan.missing_sources:
        warnings.append(f"Не найдено исходников: {len(plan.missing_sources)} — их клипы в экспорте молчат.")
    if plan.solo_active:
        warnings.append("Активен режим соло: экспортируются только дорожки соло (как при прослушивании).")
    sr = plan.sr
    s0, s1, tail = export_range_samples(project, plan, opts)
    if s1 <= s0:
        raise ValueError("Пустой участок экспорта.")
    n = s1 - s0
    out_sr = opts.sample_rate
    expected = n if out_sr == sr else int(round(n * out_sr / sr))
    # Нормализация (только по явному выбору): первый проход для пика.
    gain = 1.0
    norm_db = 0.0
    if opts.normalize_dbfs is not None:
        peak = 0.0
        for a in range(s0, s1, CHUNK):
            if cancel is not None and cancel():
                raise Cancelled()
            y = render_range(plan, a, min(s1, a + CHUNK), cache)
            peak = max(peak, float(np.max(np.abs(y))) if y.size else 0.0)
            if progress:
                progress(0.3 * (a - s0 + CHUNK) / n, "Анализ пика для нормализации…")
        if peak > 0:
            gain = 10 ** (opts.normalize_dbfs / 20) / peak
            norm_db = 20 * math.log10(gain)
    path = Path(opts.path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.stem}.asiko-partial-{uuid.uuid4().hex[:6]}.wav"
    rs = soxr.ResampleStream(sr, out_sr, 2, dtype="float64", quality="VHQ") if out_sr != sr else None
    written = 0
    clipped = 0
    abs0 = s0 if out_sr == sr else int(round(s0 * out_sr / sr))  # дизеринг зависит от абсолютной позиции
    keep = [] if n <= sr * 1200 else None
    fade = int(round(opts.edge_fade_ms / 1000.0 * sr))
    base = 0.3 if opts.normalize_dbfs is not None else 0.0
    try:
        with sf.SoundFile(str(tmp), "w", samplerate=out_sr, channels=2, subtype=opts.subtype, format="WAV") as f:
            a = s0
            while a < s1:
                if cancel is not None and cancel():
                    raise Cancelled()
                b = min(s1, a + CHUNK)
                y = render_range(plan, a, b, cache).astype(np.float64)
                if gain != 1.0:
                    y *= gain
                if fade > 0:
                    idx = np.arange(a, b)
                    w = np.ones(b - a)
                    m = idx < s0 + fade
                    w[m] = 0.5 - 0.5 * np.cos(np.pi * (idx[m] - s0 + 0.5) / fade)
                    m = idx >= s1 - fade
                    w[m] = 0.5 - 0.5 * np.cos(np.pi * (s1 - idx[m] - 0.5) / fade)
                    y *= w[:, None]
                last = b >= s1
                if rs is not None:
                    y = rs.resample_chunk(y, last=last)
                    if written + y.shape[0] > expected:
                        y = y[: expected - written]
                    if last and written + y.shape[0] < expected:
                        y = np.concatenate([y, np.zeros((expected - written - y.shape[0], 2))], axis=0)
                if keep is not None:
                    keep.append(y.astype(np.float32))
                q, c = quantize(y, opts.subtype, opts.dither, abs0 + written)
                clipped += c
                f.write(q)
                written += y.shape[0]
                a = b
                if progress:
                    progress(base + (1 - base) * 0.95 * (a - s0) / n, "Рендер и запись…")
        info = sf.info(str(tmp))
        if info.frames != expected or info.samplerate != out_sr or info.channels != 2 or info.subtype != opts.subtype:
            raise IOError(f"Проверка файла не пройдена: {info.frames} кадров вместо {expected}.")
        digest = sha256_file(tmp)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    full = np.concatenate(keep, axis=0) if keep else None
    if progress:
        progress(1.0, "Готово")
    pk = peak_dbfs(full) if full is not None else float("nan")
    tp = true_peak_dbfs(full, out_sr) if full is not None else float("nan")
    lufs = integrated_lufs(full, out_sr) if full is not None else float("nan")
    if clipped:
        warnings.append(f"Перегрузка: {clipped} сэмплов выше 0 дБFS ограничены при записи в {SUBTYPES[opts.subtype]}. "
                        f"Уменьшите уровень, включите нормализацию или экспортируйте 32 бит float.")
    if out_sr != sr:
        warnings.append(f"Частота {out_sr} Гц: выполнен ресемплинг soxr VHQ из {sr} Гц — побитовое совпадение "
                        f"с зафиксированными участками не гарантируется.")
    return ExportReport(path=str(path), frames=expected, sample_rate=out_sr, subtype=opts.subtype,
                        duration=expected / out_sr, peak_dbfs=pk, true_peak_dbfs=tp, lufs=lufs,
                        clipped_samples=clipped, tail_seconds=tail / sr, sha256=digest,
                        normalize_gain_db=norm_db, warnings=warnings)
