"""Интерфейс (offscreen): окно, сценарии, поле задания, экспорт, кнопки, видео."""
from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from asiko import testsignals as ts
from conftest import write

SR = 48000


def wait(app, cond, timeout=120.0):
    from asiko.ui.workers import _running

    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents()
        if not _running and cond():
            return True
        time.sleep(0.02)
    app.processEvents()
    return cond()


def preview_ready(win):
    return lambda: win.ctrl.preview_current("result")


@pytest.fixture
def win(qapp):
    from asiko.ui.main_window import MainWindow

    w = MainWindow()
    w._quiet = True
    w.resize(1400, 900)
    w.show()
    qapp.processEvents()
    yield w
    w.ctrl.engine.dirty = False
    w.close()
    qapp.processEvents()


def test_window_import_and_selection(win, qapp, work):
    p = write(work / "чистая музыка.wav", ts.music_piece(120, 10, SR, seed=1), SR)
    win.ctrl.import_files(win, [str(p)])
    assert wait(qapp, lambda: len(win.ctrl.project.tracks) == 1)
    assert win.timeline._headers and win.inspector.stack.currentWidget() is win.inspector.clip
    assert win.library.list.count() == 1
    assert wait(qapp, preview_ready(win))


def test_space_wizard_and_listen(win, qapp, work):
    from asiko.ui.dialogs import SpaceWizard

    p = write(work / "м.wav", ts.music_piece(120, 12, SR, seed=2), SR)
    win.ctrl.import_files(win, [str(p)])
    wait(qapp, lambda: win.ctrl.project.tracks)
    wz = SpaceWizard(win)
    wz.start.setValue(12.0)
    wz.end.setValue(16.0)
    wz.side.setCurrentIndex(0)
    wz._create(True)
    x = win.ctrl.project.space_transitions[0]
    assert x.from_state.values["pan"] == 0.6 and (x.start, x.end) == (12.0, 16.0)
    assert win.inspector.stack.currentWidget() is win.inspector.space
    assert win.ctrl.playback.playing and abs(win.ctrl.playback.position - 9.0) < 0.5
    win.stop()


def test_command_panel_and_undo_actions(win, qapp, work):
    p = write(work / "м.wav", ts.music_piece(120, 12, SR, seed=2), SR)
    win.ctrl.import_files(win, [str(p)])
    wait(qapp, lambda: win.ctrl.project.tracks)
    win.command.edit.setPlainText("Сделай переход с 12 до 16 секунды")
    win.command.btn_run.click()
    assert len(win.ctrl.project.space_transitions) == 1
    assert "Выполнено" in win.command.result.toPlainText()
    win.command.edit.setPlainText("Продли выбранный переход на 2 секунды")
    win.command.btn_run.click()
    assert win.ctrl.project.space_transitions[0].end == 18.0
    win.a_undo.trigger()
    assert win.ctrl.project.space_transitions[0].end == 16.0
    win.a_redo.trigger()
    assert win.ctrl.project.space_transitions[0].end == 18.0
    assert win.history.list.count() >= 3
    win.command.edit.setPlainText("Сделай драматичнее")
    win.command.btn_run.click()
    rev = win.ctrl.project.revision
    assert win.command.btn_apply.isVisibleTo(win.command) and win.ctrl.project.revision == rev
    win.command.btn_apply.click()
    assert win.ctrl.project.revision == rev + 1


def test_inspector_edits_go_through_commands(win, qapp, work):
    p = write(work / "м.wav", ts.music_piece(120, 12, SR, seed=2), SR)
    win.ctrl.import_files(win, [str(p)])
    wait(qapp, lambda: win.ctrl.project.tracks)
    win.command.edit.setPlainText("Сделай переход с 10 до 14 секунды")
    win.command.btn_run.click()
    page = win.inspector.space
    page.end.setValue(15.0)
    assert win.ctrl.project.space_transitions[0].end == 15.0
    lock, s0, s1, sh = page.curves["lp_hz"]
    s0.setValue(3.0)
    x = win.ctrl.project.space_transitions[0]
    assert x.curve_abs("lp_hz") == pytest.approx((13.0, 15.0))
    lock.setChecked(True)
    assert "lp_hz" in win.ctrl.project.space_transitions[0].locked_params
    page.vals["from"]["lp_hz"].setValue(3000.0)    # заблокировано — поле отключено, команда не проходит
    assert win.ctrl.project.space_transitions[0].from_state.values["lp_hz"] == 5000.0


def test_export_dialog(win, qapp, work):
    from asiko.ui.dialogs import ExportDialog

    p = write(work / "м.wav", ts.music_piece(120, 6, SR, seed=2), SR)
    win.ctrl.import_files(win, [str(p)])
    wait(qapp, lambda: win.ctrl.project.tracks)
    d = ExportDialog(win)
    out = work / "экспорт из диалога.wav"
    d.path.setText(str(out))
    opts = d.options()
    assert opts.subtype == "PCM_24" and opts.sample_rate == 48000 and opts.normalize_dbfs is None
    d._go()
    assert wait(qapp, lambda: win.last_export is not None)
    info = sf.info(str(out))
    assert info.subtype == "PCM_24" and info.samplerate == 48000


def _clicked_connected(btn) -> bool:
    from PySide6.QtCore import SIGNAL

    return btn.receivers(SIGNAL("clicked()")) + btn.receivers(SIGNAL("clicked(bool)")) + \
        btn.receivers(SIGNAL("toggled(bool)")) > 0 or btn.isCheckable()


def test_all_visible_buttons_are_connected(win, qapp, work):
    from PySide6.QtWidgets import QAbstractButton, QCheckBox, QRadioButton

    from asiko.ui.dialogs import ExportDialog, HelpDialog, MusicWizard, SettingsDialog, SpaceWizard

    p = write(work / "м.wav", ts.music_piece(120, 6, SR, seed=2), SR)
    win.ctrl.import_files(win, [str(p)])
    wait(qapp, lambda: win.ctrl.project.tracks)
    widgets = [win, ExportDialog(win), SettingsDialog(win), MusicWizard(win), SpaceWizard(win), HelpDialog(win)]
    pages = [win.inspector.space, win.inspector.music, win.inspector.clip, win.inspector.track, win.inspector.welcome]
    bad = []
    for w in widgets + pages:
        for b in w.findChildren(QAbstractButton):
            if isinstance(b, (QCheckBox, QRadioButton)):
                continue
            if type(b).__name__ in ("QToolButton",) and b.parent() is not None and "QDockWidget" in type(b.parent()).__name__:
                continue
            if b.text() in ("", "OK") and not b.toolTip():
                continue
            if not _clicked_connected(b):
                # кнопки QDialogButtonBox подключены через сигналы самого box
                if type(b.parent()).__name__ == "QDialogButtonBox":
                    continue
                bad.append(f"{type(w).__name__}: {b.text()!r}")
    assert not bad, bad


def test_unsupported_video_reports_error(win, qapp, work):
    bad = work / "битое видео.mp4"
    bad.write_bytes(b"\x00\x01garbage" * 1000)
    from asiko.commands.schema import op

    win.ctrl.ui_ops([op("video.set", params={"path": str(bad), "offset": 0.0})])
    qapp.processEvents()
    assert wait(qapp, lambda: bool(win.video.error), timeout=20)
    assert "работа с аудио продолжается" in win.video.status.text()


def _make_test_video(path: Path, seconds: int = 6, fps: int = 25) -> bool:
    ff = shutil.which("ffmpeg")
    if not ff:
        return False
    w, h = 64, 48
    frames = []
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for i in range(seconds * fps):
        c = palette[(i // fps) % len(palette)]
        frames.append(np.tile(np.array(c, np.uint8), (h, w, 1)))
    raw = np.stack(frames).tobytes()
    cmd = [ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(fps),
           "-i", "-", "-c:v", "mpeg4", "-q:v", "2", "-g", "1", str(path)]
    subprocess.run(cmd, input=raw, check=True)
    return True


# 6 с, 25 к/с, MPEG-4 Part 2: каждая секунда — свой цвет; создано _make_test_video
VIDEO_FIXTURE = Path(__file__).parent / "fixtures" / "color_seconds.mp4"


def test_video_sync_on_seek(win, qapp, tmp_path):
    vid = tmp_path / "тестовое видео.mp4"
    if VIDEO_FIXTURE.exists():
        shutil.copyfile(VIDEO_FIXTURE, vid)
    elif not _make_test_video(vid):
        pytest.skip("ffmpeg недоступен для создания тестового видео")
    from asiko.commands.schema import op

    win.ctrl.ui_ops([op("video.set", params={"path": str(vid), "offset": 1.0})])
    panel = win.video
    assert wait(qapp, lambda: panel.player is not None and panel.player.duration() > 0, timeout=30), panel.status.text()
    if panel.error:
        pytest.skip(f"кодек недоступен в этой сборке Qt: {panel.error}")
    frames = []
    panel.widget.videoSink().videoFrameChanged.connect(lambda f: frames.append(f.toImage()))
    palette = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
    for t in (1.5, 4.5, 3.2):            # время шкалы; видео начинается на 1.0 с
        frames.clear()
        win.seek(t)
        assert wait(qapp, lambda: abs(panel.player.position() / 1000 - (t - 1.0)) < 0.05 and frames, timeout=10)
        img = frames[-1]
        c = img.pixelColor(img.width() // 2, img.height() // 2)
        exp = palette[int(t - 1.0) % len(palette)]
        assert abs(c.red() - exp[0]) < 60 and abs(c.green() - exp[1]) < 60 and abs(c.blue() - exp[2]) < 60, (t, c.getRgb())
