"""Дополнительные пути интерфейса: демо-проекты, фиксация PCM и защита из меню, настройки, поиск исходников."""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from asiko import testsignals as ts
from conftest import write
from test_ui import preview_ready, wait, win  # noqa: F401  (фикстура окна)

SR = 48000


def test_open_demo_from_menu(win, qapp):
    win.open_demo("space")
    assert wait(qapp, lambda: bool(win.ctrl.project.space_transitions), timeout=240)
    assert win.ctrl.path is not None and win.ctrl.path.name.startswith("Демо")
    assert wait(qapp, preview_ready(win), timeout=120)
    win.open_demo("music")
    assert wait(qapp, lambda: bool(win.ctrl.project.music_transitions), timeout=240)
    assert len(win.ctrl.project.music_transitions[0].variants) == 3


def test_protect_and_freeze_selection(win, qapp, work):
    p = write(work / "м.wav", ts.music_piece(120, 12, SR, seed=3), SR)
    win.ctrl.import_files(win, [str(p)])
    wait(qapp, lambda: win.ctrl.project.tracks)
    win.ctrl.select(time_range=(8.0, 12.0))
    win.protect_selection()
    assert win.ctrl.project.protections[0].start == 8.0
    win.ctrl.freeze_selection(win)                    # проект не сохранён → предупреждение
    assert any("сохраните" in m for _l, m in win.messages)
    win.ctrl.save(work / "фиксация.asiko")
    win.ctrl.freeze_selection(win)
    assert wait(qapp, lambda: bool(win.ctrl.project.frozen), timeout=60)
    fr = win.ctrl.project.frozen[0]
    assert (work / "фиксация.asiko_data" / fr.file).exists()
    assert wait(qapp, preview_ready(win), timeout=60)
    buf = win.ctrl.playback.buffers["result"]
    data = win.ctrl.frozen(fr)
    m = int(round(fr.margin * SR))
    assert np.array_equal(buf.data[8 * SR:12 * SR], data[m:m + 4 * SR])
    win.clear_protections()
    assert not win.ctrl.project.protections


def test_settings_dialog_saves_llm_config(win, qapp):
    from asiko.ui.dialogs import SettingsDialog

    d = SettingsDialog(win)
    d.provider.setCurrentIndex(d.provider.findData("anthropic"))
    assert d.endpoint.text() == "https://api.anthropic.com" and d.model.text() == "claude-opus-5-5"
    d.key.setText("sk-test")
    d._save()
    assert win.ctrl.llm_config.provider == "anthropic"
    assert win.ctrl.llm().available()[0]
    assert win.command.use_llm.isEnabled()
    d2 = SettingsDialog(win)
    d2.provider.setCurrentIndex(d2.provider.findData("none"))
    d2._save()
    assert not win.command.use_llm.isEnabled()


def test_missing_source_marked_and_relinked(win, qapp, work, tmp_path):
    from asiko.storage.project_io import find_moved_sources

    p = write(work / "уедет.wav", ts.noise(3.0, SR, seed=4), SR)
    win.ctrl.import_files(win, [str(p)])
    wait(qapp, lambda: win.ctrl.project.tracks)
    path = work / "проект.asiko"
    win.ctrl.save(path)
    dest = tmp_path / "Новая папка"
    dest.mkdir()
    shutil.move(str(p), str(dest / "уехал.wav"))
    from asiko.ui.controller import Controller

    win.ctrl.store.forget()
    shutil.rmtree(win.ctrl.store.cache_dir, ignore_errors=True)
    win.ctrl.store.cache_dir.mkdir(parents=True, exist_ok=True)
    assert win.open_path(str(path), check=False)
    from asiko.storage.project_io import check_sources

    st = check_sources(win.ctrl.project)
    win._after_check.__func__  # метод существует
    found = find_moved_sources(win.ctrl.project, dest)
    from asiko.commands.schema import op

    win.ctrl.ui_ops([op("source.relink", target={"source_id": sid}, params={"path": pth}) for sid, pth in found.items()])
    assert st[0].status == "missing" and win.ctrl.project.sources[0].path.endswith("уехал.wav")
