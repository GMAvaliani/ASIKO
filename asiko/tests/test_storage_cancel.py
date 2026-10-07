"""Хранение (автосохранение, восстановление, поиск перемещённых исходников, переносимый проект)
и 14. отмена анализа, импорта и экспорта."""
from __future__ import annotations

import shutil
import threading
import time

import numpy as np
import pytest

from asiko import testsignals as ts
from asiko.audio.analysis import analyze_beats
from asiko.audio.export import ExportOptions, export_wav
from asiko.audio.io import Cancelled
from asiko.storage.project_io import Autosaver, check_sources, collect_project, find_moved_sources, load_project, save_project
from conftest import write

SR = 48000


def test_autosave_and_recovery(builder, work, tmp_path):
    p = write(work / "м.wav", ts.noise(3.0, SR, seed=1), SR)
    builder.add_file(p)
    a = Autosaver(tmp_path / "автосохранение")
    a.write(builder.project, None)
    ents = a.entries()
    assert len(ents) == 1 and ents[0].original is None
    rec = load_project(ents[0].path)
    assert rec.to_dict()["tracks"] == builder.project.to_dict()["tracks"]
    saved = work / "сохранён.asiko"
    save_project(builder.project, saved)
    time.sleep(0.05)
    a.write(builder.project, str(saved))   # новее файла — предлагается
    assert a.entries()
    save_project(builder.project, saved)   # файл новее автосохранения — не предлагается
    assert not a.entries()
    a.clear(builder.project.id)
    assert not list((tmp_path / "автосохранение").glob("*"))


def test_relink_moved_sources(builder, work, tmp_path):
    p = write(work / "исходник для поиска.wav", ts.noise(2.0, SR, seed=8), SR)
    builder.add_file(p)
    proj = work / "проект.asiko"
    save_project(builder.project, proj)
    moved = tmp_path / "Новое место" / "вложенная папка"
    moved.mkdir(parents=True)
    shutil.move(str(p), str(moved / "переименован.wav"))
    project = load_project(proj)
    st = check_sources(project)
    assert st[0].status == "missing"
    found = find_moved_sources(project, tmp_path / "Новое место")
    assert found == {project.sources[0].id: str((moved / "переименован.wav").resolve())}


def test_relative_paths_after_folder_move(builder, tmp_path):
    folder = tmp_path / "Папка проекта"
    p = write(folder / "media" / "звук.wav", ts.noise(1.0, SR, seed=2), SR)
    builder.add_file(p)
    save_project(builder.project, folder / "проект.asiko")
    shutil.move(str(folder), str(tmp_path / "Перенесённая папка"))
    proj = load_project(tmp_path / "Перенесённая папка" / "проект.asiko")
    assert check_sources(proj)[0].status == "ok"


def test_collect_portable(builder, work, tmp_path):
    p = write(work / "звук.wav", ts.noise(1.0, SR, seed=2), SR)
    builder.add_file(p)
    pp = work / "исх.asiko"
    save_project(builder.project, pp)
    out = collect_project(builder.project, pp, tmp_path / "Переносимый проект")
    proj = load_project(out)
    assert (tmp_path / "Переносимый проект" / "media" / "звук.wav").exists()
    assert all(s.status == "ok" for s in check_sources(proj))


def _cancel_after(token_holder, delay):
    def run():
        time.sleep(delay)
        token_holder["cancel"] = True

    threading.Thread(target=run, daemon=True).start()


def test_cancel_analysis():
    x = ts.music_piece(120, 120, SR, seed=1)   # 4 минуты
    flag = {"cancel": False}
    seen = []

    def progress(f, m=""):
        seen.append(f)
        if f > 0.2:
            flag["cancel"] = True

    t0 = time.perf_counter()
    with pytest.raises(Cancelled):
        analyze_beats(x, SR, cancel=lambda: flag["cancel"], progress=progress)
    assert time.perf_counter() - t0 < 60
    assert seen and max(seen) < 0.9


def test_cancel_export_leaves_no_file(builder, work):
    p = write(work / "длинный.wav", ts.noise(60.0, SR, seed=3, amp=0.2), SR)
    src, clip = builder.add_file(p)
    tid = builder.project.tracks[0].id
    builder.space(tid, 10.0, 20.0, from_preset="distant", to_preset="offscreen")
    out = work / "отменённый.wav"
    calls = {"n": 0}

    def cancel():
        calls["n"] += 1
        return calls["n"] > 3

    with pytest.raises(Cancelled):
        export_wav(builder.project, builder.store, ExportOptions(path=str(out)), cancel=cancel)
    assert not out.exists()
    assert not list(work.glob(".*partial*"))


def test_export_error_leaves_no_partial(builder, work, monkeypatch):
    p = write(work / "м.wav", ts.noise(20.0, SR, seed=3), SR)
    builder.add_file(p)
    import asiko.audio.export as ex

    real = ex.render_range
    n = {"c": 0}

    def boom(*a, **k):
        n["c"] += 1
        if n["c"] > 1:
            raise RuntimeError("сбой рендера")
        return real(*a, **k)

    monkeypatch.setattr(ex, "render_range", boom)
    out = work / "сбой.wav"
    with pytest.raises(RuntimeError):
        export_wav(builder.project, builder.store, ExportOptions(path=str(out)))
    assert not out.exists() and not list(work.glob(".*partial*"))


def test_cancel_import(store, work):
    p = write(work / "большой.wav", ts.noise(120.0, 44100, seed=4), 44100)
    with pytest.raises(Cancelled):
        store.import_file(p, cancel=lambda: True)
