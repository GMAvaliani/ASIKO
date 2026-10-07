"""Демонстрационные материалы и проекты (синтезированная музыка, без чужих записей)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from . import app_paths, testsignals as ts
from .audio.analysis import analyze_beats
from .audio.sources import SourceStore
from .commands.engine import CommandEngine
from .commands.schema import make_plan, op
from .model.project import Project
from .storage.project_io import save_project

DEMO_VERSION = "v1"


def _write(path: Path, data: np.ndarray, sr: int, subtype: str = "PCM_24", fmt: str | None = None) -> None:
    if path.exists():
        return
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    sf.write(str(tmp), data, sr, subtype=subtype, format=fmt)
    tmp.replace(path)


def demo_audio(folder: Path) -> dict[str, Path]:
    folder.mkdir(parents=True, exist_ok=True)
    clean = folder / "Музыка — чистая.wav"
    a = folder / "Трек A (120 BPM).flac"
    b = folder / "Трек B (100 BPM).flac"
    _write(clean, ts.music_piece(bpm=110, bars=16, sr=48000, root="D", minor=False, seed=21), 48000)
    _write(a, ts.music_piece(bpm=120, bars=24, sr=44100, root="A", minor=True, seed=7, final_ring=3.0), 44100,
           subtype="PCM_24", fmt="FLAC")
    _write(b, ts.music_piece(bpm=100, bars=24, sr=48000, root="C", minor=False, seed=11), 48000,
           subtype="PCM_24", fmt="FLAC")
    return {"clean": clean, "a": a, "b": b}


def build_space_project(folder: Path, store: SourceStore, audio: dict[str, Path]) -> Path:
    eng = CommandEngine(Project(name="Демо — звук в сцене → закадровый"))
    src = store.import_file(audio["clean"])
    r = eng.apply(make_plan([op("source.add", params={"source": src.to_dict()}),
                             op("clip.add", target={"source_id": src.id},
                                time={"coord": "timeline", "unit": "s", "start": 0.0})], eng.project.revision))
    assert r.ok, r.errors
    tid = eng.project.tracks[0].id
    eng.apply(make_plan([op("track.set", target={"track_id": tid}, params={"name": "Музыка (радио в комнате)"})],
                        eng.project.revision))
    r = eng.apply(make_plan([op("space.create", target={"track_id": tid},
                                time={"coord": "timeline", "unit": "s", "start": 12.0, "end": 16.0},
                                params={"from_preset": "radio_room", "to_preset": "offscreen",
                                        "from_values": {"pan": 0.6}})], eng.project.revision))
    assert r.ok, r.errors
    path = folder / "Демо — звук в сцене.asiko"
    save_project(eng.project, path)
    return path


def build_music_project(folder: Path, store: SourceStore, audio: dict[str, Path], progress=None) -> Path:
    eng = CommandEngine(Project(name="Демо — переход A → B"))
    sa = store.import_file(audio["a"])
    sb = store.import_file(audio["b"])
    ops = [op("source.add", params={"source": sa.to_dict()}), op("source.add", params={"source": sb.to_dict()}),
           op("clip.add", target={"source_id": sa.id}, time={"coord": "timeline", "unit": "s", "start": 0.0}),
           op("clip.add", target={"source_id": sb.id}, time={"coord": "timeline", "unit": "s", "start": 36.0})]
    r = eng.apply(make_plan(ops, 0))
    assert r.ok, r.errors
    app_paths.setup_numba_cache()
    grids = []
    for i, s in enumerate((sa, sb)):
        g = analyze_beats(np.asarray(store.load(s)), store.sample_rate,
                          progress=(lambda f, m="": progress(0.4 + 0.25 * (i + f), m)) if progress else None)
        grids.append(op("beats.set", target={"source_id": s.id}, params={"grid": asdict(g)}))
    r = eng.apply(make_plan(grids, eng.project.revision))
    assert r.ok, r.errors
    ca, cb = (t.clips[0].id for t in eng.project.tracks)
    r = eng.apply(make_plan([op("music.create", target={"clip_a_id": ca, "clip_b_id": cb},
                                time={"coord": "timeline", "unit": "s", "at": 40.0},
                                params={"b_on_downbeat": True, "keep_a_tail": True},
                                constraints=[{"kind": "keep_tempo"}, {"kind": "keep_pitch"}])], eng.project.revision))
    assert r.ok, r.errors
    path = folder / "Демо — переход A-B.asiko"
    save_project(eng.project, path)
    return path


def ensure_demo(folder: str | Path | None = None, progress=None) -> dict[str, Path]:
    folder = Path(folder) if folder else app_paths.demo_dir() / DEMO_VERSION
    folder.mkdir(parents=True, exist_ok=True)
    space = folder / "Демо — звук в сцене.asiko"
    music = folder / "Демо — переход A-B.asiko"
    if progress:
        progress(0.05, "Синтез демонстрационной музыки…")
    audio = demo_audio(folder)
    store = SourceStore(app_paths.audio_cache_dir(), 48000)
    if not space.exists():
        if progress:
            progress(0.3, "Проект «звук в сцене → закадровый»…")
        build_space_project(folder, store, audio)
    if not music.exists():
        if progress:
            progress(0.4, "Проект «переход A → B» (анализ долей)…")
        build_music_project(folder, store, audio, progress)
    if progress:
        progress(1.0, "Готово")
    return {"space": space, "music": music, "folder": folder}
