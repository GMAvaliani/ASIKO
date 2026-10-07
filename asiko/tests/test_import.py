"""1. Импорт файлов с разными частотами дискретизации, разрядностью и числом каналов (+ кириллица)."""
from __future__ import annotations

import numpy as np
import pytest

from asiko import testsignals as ts
from asiko.audio.io import AudioImportError, probe
from conftest import write

CASES = [
    ("моно 44.1 кГц 16 бит.wav", 44100, 1, "PCM_16", "WAV"),
    ("стерео 48 кГц 24 бит.wav", 48000, 2, "PCM_24", "WAV"),
    ("стерео 96 кГц float.wav", 96000, 2, "FLOAT", "WAV"),
    ("моно 22.05 кГц.wav", 22050, 1, "PCM_16", "WAV"),
    ("стерео 44.1 кГц.flac", 44100, 2, "PCM_24", "FLAC"),
    ("моно 48 кГц.flac", 48000, 1, "PCM_16", "FLAC"),
    ("стерео 44.1 кГц.mp3", 44100, 2, "MPEG_LAYER_III", "MP3"),
    ("моно 48 кГц.mp3", 48000, 1, "MPEG_LAYER_III", "MP3"),
    ("стерео 48 кГц.ogg", 48000, 2, "VORBIS", "OGG"),
    ("стерео 48 кГц.aiff", 48000, 2, "PCM_24", "AIFF"),
]


@pytest.mark.parametrize("name,sr,ch,subtype,fmt", CASES)
def test_import_formats(work, store, name, sr, ch, subtype, fmt):
    dur = 2.5
    data = ts.sine(1000.0, dur, sr, 0.5, ch)
    p = write(work / name, data, sr, subtype, fmt)
    src = store.import_file(p)
    assert src.sample_rate == sr
    assert src.channels == ch
    if fmt == "MP3":
        # MP3: libsndfile учитывает задержку кодера; допускаем расхождение в 1 кадр MP3
        assert abs(src.duration - dur) < 1152 / sr + 1e-3
    else:
        assert src.frames == int(round(dur * sr))
    audio = store.load(src)
    assert audio.dtype == np.float32
    assert audio.shape == (int(round(src.frames * 48000 / sr)), ch)
    assert np.all(np.isfinite(audio))
    # после явного ресемплинга (soxr VHQ) частота и уровень тона сохранены
    mid = np.asarray(audio[int(0.5 * 48000):int(2.0 * 48000), 0], dtype=np.float64)
    spec = np.abs(np.fft.rfft(mid * np.hanning(mid.size)))
    f = np.fft.rfftfreq(mid.size, 1 / 48000)
    assert abs(f[np.argmax(spec)] - 1000.0) < 2.0
    if fmt not in ("MP3", "OGG"):
        rms = np.sqrt(np.mean(mid ** 2))
        assert abs(20 * np.log10(rms / (0.5 / np.sqrt(2)))) < 0.05


def test_source_never_modified(work, store):
    p = write(work / "исходник.wav", ts.noise(1.0, 48000, seed=5), 48000)
    before = p.read_bytes()
    from asiko.audio.io import sha256_file

    src = store.import_file(p)
    assert src.sha256 == sha256_file(p)
    store.load(src)
    assert p.read_bytes() == before


def test_resample_exact_length(store, work):
    from asiko.audio.sources import resample

    x = ts.noise(1.2345, 44100, seed=1, channels=2)
    y = resample(x, 44100, 48000)
    assert y.shape[0] == int(round(x.shape[0] * 48000 / 44100))


def test_unsupported_file_clear_error(work):
    bad = work / "не аудио.wav"
    bad.write_bytes(b"this is not audio at all" * 10)
    with pytest.raises(AudioImportError) as e:
        probe(bad)
    assert "не поддерживается" in str(e.value) or "повреждён" in str(e.value)


def test_multichannel_rejected(work):
    p = write(work / "5.1.wav", np.zeros((4800, 6), np.float32), 48000)
    with pytest.raises(AudioImportError) as e:
        probe(p)
    assert "каналов" in str(e.value)
