"""Сквозные приемочные сценарии A, B, C (раздел 15 ТЗ) через окно приложения.

Действия выполняются теми же виджетами, что и у пользователя (мастера сценариев,
поле задания, кнопки инспектора, действия меню, диалог экспорта), в offscreen-режиме.
Реальное прослушивание недоступно (нет звуковой карты) — проверяется сигнал.
"""
from __future__ import annotations

import socket

import numpy as np
import pytest
import soundfile as sf

from asiko import testsignals as ts
from conftest import write
from test_ui import preview_ready, wait

SR = 48000


@pytest.fixture
def make_win(qapp):
    from asiko.ui.main_window import MainWindow

    wins = []

    def make():
        w = MainWindow()
        w._quiet = True
        w.resize(1400, 900)
        w.show()
        qapp.processEvents()
        wins.append(w)
        return w

    yield make
    for w in wins:
        w.ctrl.engine.dirty = False
        w.close()
    qapp.processEvents()


def run_text(win, text):
    win.command.edit.setPlainText(text)
    win.command.btn_run.click()
    return win.command.result.toPlainText()


def scenario_a(win, qapp, work):
    from asiko.ui.dialogs import ExportDialog, SpaceWizard

    music = write(work / "Чистая музыка.wav", ts.music_piece(110, 16, SR, seed=21), SR)
    win.ctrl.import_files(win, [str(music)])
    assert wait(qapp, lambda: len(win.ctrl.project.tracks) == 1)
    # «Радио справа → закадровый» с 12-й по 16-ю секунду — через видимый сценарий
    wz = SpaceWizard(win)
    wz.preset.setCurrentIndex(wz.preset.findData("radio_room"))
    wz.side.setCurrentIndex(wz.side.findData(0.6))
    wz.start.setValue(12.0)
    wz.end.setValue(16.0)
    wz.btn_play.click()                                  # «Создать и прослушать»
    x = win.ctrl.project.space_transitions[0]
    assert (x.start, x.end) == (12.0, 16.0) and x.from_state.values["pan"] == 0.6
    assert win.ctrl.playback.playing
    assert wait(qapp, preview_ready(win))
    buf = win.ctrl.playback.buffers["result"]
    seg_before = buf.data[int(8 * SR):int(11 * SR)]
    seg_after = buf.data[int(17 * SR):int(20 * SR)]
    # до перехода звук смещён вправо (радио справа), после — по центру
    rms = lambda a: np.sqrt(np.mean(a.astype(np.float64) ** 2, axis=0))
    rb, ra = rms(seg_before), rms(seg_after)
    assert rb[1] > rb[0] * 1.5 and abs(20 * np.log10(ra[1] / ra[0])) < 1.0
    win.stop()
    # продлить до 18-й секунды и перенести раскрытие частот в конец — текстом
    out = run_text(win, "Продли выбранный переход до 18 секунды")
    assert win.ctrl.project.space_transitions[0].end == 18.0, out
    out = run_text(win, "Перенеси раскрытие частот в конец")
    x = win.ctrl.project.space_transitions[0]
    assert x.curve_abs("hp_hz") == pytest.approx((16.0, 18.0)) and x.curve_abs("pan") == pytest.approx((12.0, 18.0))
    assert "Раскрытие частот" in out and "остальные параметры сохранены" in out
    # отменить и повторить
    win.a_undo.trigger()
    assert win.ctrl.project.space_transitions[0].curve_abs("hp_hz") == pytest.approx((12.0, 18.0))
    win.a_redo.trigger()
    assert win.ctrl.project.space_transitions[0].curve_abs("hp_hz") == pytest.approx((16.0, 18.0))
    # сохранить, открыть заново, экспортировать WAV
    path = work / "Сценарий A.asiko"
    win.ctrl.save(path)
    return path


def export_and_check(win, qapp, work, name, setup=None):
    from asiko.ui.dialogs import ExportDialog

    d = ExportDialog(win)
    out = work / name
    d.path.setText(str(out))
    if setup:
        setup(d)
    win.last_export = None
    d._go()
    assert wait(qapp, lambda: win.last_export is not None)
    info = sf.info(str(out))
    assert info.samplerate == 48000 and info.subtype == "PCM_24" and info.channels == 2
    data, _ = sf.read(str(out))
    assert np.all(np.isfinite(data)) and np.max(np.abs(data)) > 0.01
    return win.last_export


def test_scenario_a(make_win, qapp, work):
    w1 = make_win()
    path = scenario_a(w1, qapp, work)
    w2 = make_win()
    assert w2.open_path(str(path), check=False)
    x = w2.ctrl.project.space_transitions[0]
    assert (x.start, x.end) == (12.0, 18.0) and x.curve_abs("hp_hz") == pytest.approx((16.0, 18.0))
    rep = export_and_check(w2, qapp, work, "Сценарий A.wav")
    assert rep.duration > w2.ctrl.project.content_end()      # с хвостом реверберации


def scenario_b(win, qapp, work, compare=True):
    from asiko.ui.dialogs import MusicWizard

    a = write(work / "Трек A.flac", ts.music_piece(120, 24, 44100, seed=7, final_ring=3.0), 44100, fmt="FLAC")
    b = write(work / "Трек B.wav", ts.music_piece(100, 20, SR, root="C", minor=False, seed=11), SR)
    win.ctrl.import_files(win, [str(a)])
    assert wait(qapp, lambda: len(win.ctrl.project.tracks) == 1)
    win.ctrl.import_files(win, [str(b)], at=44.0)
    assert wait(qapp, lambda: len(win.ctrl.project.tracks) == 2)
    wz = MusicWizard(win)
    wz.at.setValue(40.0)
    assert wz.keep.isChecked() and wz.downbeat.isChecked() and wz.tail.isChecked()
    wz.btn_ok.click()                                    # анализ долей → варианты
    assert wait(qapp, lambda: bool(win.ctrl.project.music_transitions), timeout=180)
    p = win.ctrl.project
    mt = p.music_transitions[0]
    assert p.constraints == {"keep_tempo": True, "keep_pitch": True}
    assert [v.kind for v in mt.variants] == ["short", "long", "phrase"]
    assert all(s.beats is not None and s.beats.confidence > 0.3 for s in p.sources)
    assert mt.variant("long").conflicts                    # темпы 120/100 при запрете изменения темпа
    win.ctrl.select(transition_id=mt.id)
    page = win.inspector.music
    assert win.inspector.stack.currentWidget() is page
    # выбрать вариант «завершение фразы» в списке
    item = page.variants.item(2)
    page._pick(item)
    mt = win.ctrl.project.music_transitions[0]
    assert mt.variant(mt.selected_variant).kind == "phrase"
    # ручная правка точки входа: +1 доля, затем −10 мс
    before = mt.params.b_entry_src
    page.nudges[4].click()
    page.nudges[2].click()
    mt = win.ctrl.project.music_transitions[0]
    beat = 60.0 / win.ctrl.project.source(win.ctrl.project.clip(mt.clip_b_id).source_id).beats.bpm
    assert mt.params.b_entry_src == pytest.approx(before + beat - 0.01, abs=1e-6) and mt.modified
    if compare:
        page.btn_cmp.click()
        modes = [m for m, _ in win.ctrl.compare_modes() if m.startswith("variant:")]
        assert len(modes) == 3
        assert wait(qapp, lambda: all(m in win.ctrl.playback.buffers and win.ctrl.playback.buffers[m].complete()
                                      for m in modes), timeout=120)
        win.lmatch.setChecked(True)
        win.compare.setCurrentIndex(win.compare.findData(modes[1]))
        win._compare_changed(0)
        assert win.ctrl.playback.active == modes[1]
        win.ctrl.set_compare("result")
        win.stop()
    return mt


def test_scenario_b(make_win, qapp, work):
    w = make_win()
    mt = scenario_b(w, qapp, work)
    long_v = mt.variant("long")

    def pick_variant(d):
        d.r_x.setChecked(True)
        d.variant.setCurrentIndex(d.variant.findData((mt.id, long_v.id)))

    rep = export_and_check(w, qapp, work, "Сценарий B (вариант 2).wav", pick_variant)
    assert rep.frames > 0
    # экспорт выбранного варианта не меняет проект
    assert w.ctrl.project.music_transitions[0].variant(w.ctrl.project.music_transitions[0].selected_variant).kind == "phrase"


def test_scenario_c_offline(make_win, qapp, work, monkeypatch):
    """Без сети: оба сценария через ручное управление и локальные команды; сохранение и экспорт."""
    def no_network(*a, **k):
        raise OSError("Сеть отключена (сценарий C)")

    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket.socket, "connect", no_network)
    w = make_win()
    assert not w.command.use_llm.isEnabled()          # модель не настроена → только локальные шаблоны
    assert "Локальный обработчик" in w.command.mode_label.text()
    # Сценарий A — только локальными командами
    music = write(work / "Музыка.wav", ts.music_piece(110, 16, SR, seed=21), SR)
    w.ctrl.import_files(w, [str(music)])
    assert wait(qapp, lambda: w.ctrl.project.tracks)
    run_text(w, "До 12-й секунды музыка звучит из радио справа в комнате. С 12-й по 16-ю переходит в закадровую.")
    run_text(w, "Продли выбранный переход до 18 секунды")
    run_text(w, "Перенеси раскрытие частот в конец")
    run_text(w, "Отмени последнее изменение")
    run_text(w, "Повтори отменённое")
    x = w.ctrl.project.space_transitions[0]
    assert (x.start, x.end) == (12.0, 18.0) and x.curve_abs("hp_hz") == pytest.approx((16.0, 18.0))
    run_text(w, "Уменьши громкость выбранной дорожки на 3 дБ")
    assert w.ctrl.project.tracks[0].gain_db == -3.0
    w.ctrl.save(work / "Без сети A.asiko")
    export_and_check(w, qapp, work, "Без сети A.wav")
    # внешняя модель с недоступным адресом: ошибка сети не мешает работе
    from asiko.nlp.llm import LlmConfig

    w.ctrl.save_llm_config(LlmConfig(provider="openai_compatible", endpoint="http://10.255.255.1:9/v1", model="m"), None)
    w.command.update_mode()
    w.command.use_llm.setChecked(True)
    rev = w.ctrl.project.revision
    run_text(w, "Сдвинь выбранный переход на 1 секунду вправо")
    assert wait(qapp, lambda: "Внешняя модель" in w.command.result.toPlainText(), timeout=30)
    assert "Нет соединения" in w.command.result.toPlainText()
    assert w.ctrl.project.revision == rev                    # проект не изменён
    w.command.use_llm.setChecked(False)
    # Сценарий B — через мастер (анализ локальный) и ручные кнопки
    w2 = make_win()
    scenario_b(w2, qapp, work, compare=False)
    run_text(w2, "Не меняй темп и высоту тона")
    w2.ctrl.save(work / "Без сети B.asiko")
    export_and_check(w2, qapp, work, "Без сети B.wav")
