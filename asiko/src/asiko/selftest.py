"""Самопроверка (в том числе упакованного приложения): `ASIKO --self-test [папка]`.

Проверяет на реальных файлах: импорт форматов, оба сценария, локальные команды,
отмену/повтор, сохранение/открытие, экспорт WAV, запуск интерфейса Qt.
Работает без сети и без ключа модели. Результат — JSON-отчёт и код возврата.
"""
from __future__ import annotations

import json
import os
import platform
import sys
import tempfile
import time
import traceback
from dataclasses import asdict
from pathlib import Path


def main(out_dir: str | None = None) -> int:
    base = Path(out_dir) if out_dir else Path(tempfile.mkdtemp(prefix="asiko-selftest-"))
    root = base / "Самопроверка ASIKO"
    root.mkdir(parents=True, exist_ok=True)
    os.environ["ASIKO_HOME"] = str(root / "home")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from . import app_paths

    app_paths.setup_numba_cache()
    report: dict = {"app": "ASIKO", "platform": platform.platform(), "python": sys.version.split()[0],
                    "frozen": bool(getattr(sys, "frozen", False)), "checks": []}

    def check(name, fn):
        t0 = time.perf_counter()
        try:
            info = fn()
            report["checks"].append({"name": name, "ok": True, "seconds": round(time.perf_counter() - t0, 2),
                                     "info": info})
            print(f"[OK]   {name}: {info}")
        except Exception as e:  # отчёт должен собрать все проверки
            report["checks"].append({"name": name, "ok": False, "error": f"{type(e).__name__}: {e}",
                                     "trace": traceback.format_exc()})
            print(f"[FAIL] {name}: {type(e).__name__}: {e}")

    import numpy as np
    import soundfile as sf

    from . import testsignals as ts
    from .audio.export import ExportOptions, export_wav
    from .audio.render import ChunkCache, compile_plan, render_range
    from .audio.sources import SourceStore
    from .commands.engine import CommandEngine
    from .commands.schema import make_plan, op
    from .model.project import Project
    from .nlp.local_parser import ParseContext, Selection, parse
    from .storage.project_io import load_project, save_project

    src_dir = root / "исходники с пробелами"
    src_dir.mkdir(exist_ok=True)
    store = SourceStore(app_paths.audio_cache_dir(), 48000)
    cache = ChunkCache()
    state: dict = {}

    def libs():
        import librosa
        import PySide6
        import scipy
        import soxr

        out = {"numpy": np.__version__, "scipy": scipy.__version__, "libsndfile": sf.__libsndfile_version__,
               "soxr": soxr.__version__, "librosa": librosa.__version__, "PySide6": PySide6.__version__}
        try:
            from asiko.audio.playback import import_sounddevice

            sd = import_sounddevice()

            out["portaudio"] = sd.get_portaudio_version()[1]
            out["output_devices"] = sum(1 for d in sd.query_devices() if d["max_output_channels"] > 0)
        except Exception as e:
            out["portaudio"] = f"недоступен: {e}"
        return out

    def formats():
        files = {
            "моно 44.1k 16bit.wav": (ts.sine(440, 3.0, 44100, 0.5, 1), 44100, "PCM_16", "WAV"),
            "стерео 96k float.wav": (ts.sine(1000, 2.0, 96000, 0.5, 2), 96000, "FLOAT", "WAV"),
            "стерео 48k.flac": (ts.noise(2.0, 48000, seed=3), 48000, "PCM_24", "FLAC"),
            "стерео 48k.mp3": (ts.sine(500, 3.0, 48000, 0.4, 2), 48000, "MPEG_LAYER_III", "MP3"),
        }
        res = {}
        for name, (data, sr, sub, fmt) in files.items():
            p = src_dir / name
            sf.write(str(p), data, sr, subtype=sub, format=fmt)
            s = store.import_file(p)
            a = store.load(s)
            expected = int(round(s.frames * 48000 / s.sample_rate))
            assert a.shape[0] == expected, (name, a.shape, expected)
            assert np.all(np.isfinite(a))
            res[name] = f"{s.format} {s.sample_rate} Гц, {s.channels} кан., {s.duration:.3f} с"
        return res

    def scenario_a():
        music = src_dir / "чистая музыка.wav"
        sf.write(str(music), ts.music_piece(110, 16, 48000, seed=21), 48000, subtype="PCM_24")
        eng = CommandEngine(Project(name="Самопроверка A"))
        s = store.import_file(music)
        r = eng.apply(make_plan([op("source.add", params={"source": s.to_dict()}),
                                 op("clip.add", target={"source_id": s.id},
                                    time={"coord": "timeline", "unit": "s", "start": 0})], 0))
        assert r.ok, r.errors
        sel = Selection(track_id=eng.project.tracks[0].id)

        def run(text):
            plan = parse(text, ParseContext(eng.project, sel))
            res = eng.apply(plan)
            assert res.ok and (res.applied or plan["kind"] in ("undo", "redo")), (text, res.errors, plan)
            if res.select and eng.project.any_transition(res.select):
                sel.transition_id = res.select
            return res

        run("До 12-й секунды музыка звучит из радио справа в комнате. С 12-й по 16-ю переходит в закадровую.")
        run("Продли выбранный переход до 18 секунды")
        run("Перенеси раскрытие частот в конец")
        x = eng.project.space_transitions[0]
        assert (x.start, x.end) == (12.0, 18.0)
        assert abs(x.curve_abs("hp_hz")[0] - 16.0) < 1e-9
        run("Отмени последнее изменение")
        assert abs(eng.project.space_transitions[0].curve_abs("hp_hz")[0] - 12.0) < 1e-9
        run("Повтори отменённое")
        assert abs(eng.project.space_transitions[0].curve_abs("hp_hz")[0] - 16.0) < 1e-9
        pp = root / "проект A.asiko"
        save_project(eng.project, pp)
        p2 = load_project(pp)
        assert p2.to_dict()["space_transitions"] == eng.project.to_dict()["space_transitions"]
        out = root / "экспорт A.wav"
        rep = export_wav(p2, store, ExportOptions(path=str(out)), cache=cache)
        info = sf.info(str(out))
        assert info.samplerate == 48000 and info.subtype == "PCM_24" and info.frames == rep.frames
        assert rep.duration >= s.duration
        state["a"] = p2
        return f"переход 12–18 с, экспорт {rep.duration:.2f} с, пик {rep.peak_dbfs:.1f} дБFS, {rep.lufs:.1f} LUFS"

    def scenario_b():
        from .audio.analysis import analyze_beats

        pa = src_dir / "трек A.flac"
        pb = src_dir / "трек B.wav"
        sf.write(str(pa), ts.music_piece(120, 24, 44100, seed=7, final_ring=3.0), 44100, subtype="PCM_24", format="FLAC")
        sf.write(str(pb), ts.music_piece(120, 16, 48000, root="C", minor=False, seed=11), 48000, subtype="PCM_24")
        eng = CommandEngine(Project(name="Самопроверка B"))
        sa, sb = store.import_file(pa), store.import_file(pb)
        r = eng.apply(make_plan([op("source.add", params={"source": sa.to_dict()}),
                                 op("source.add", params={"source": sb.to_dict()}),
                                 op("clip.add", target={"source_id": sa.id}, time={"coord": "timeline", "unit": "s", "start": 0}),
                                 op("clip.add", target={"source_id": sb.id}, time={"coord": "timeline", "unit": "s", "start": 45})], 0))
        assert r.ok, r.errors
        grids = []
        for s in (sa, sb):
            g = analyze_beats(np.asarray(store.load(s)), 48000)
            assert abs(g.bpm - 120) < 0.5, g.bpm
            grids.append(op("beats.set", target={"source_id": s.id}, params={"grid": asdict(g)}))
        assert eng.apply(make_plan(grids, eng.project.revision)).ok
        sel = Selection()
        plan = parse("Перейти с A на B примерно на 40-й секунде. B должен войти на сильную долю. "
                     "Темп и высоту тона не менять. Хвост A сохранить.", ParseContext(eng.project, sel))
        r = eng.apply(plan)
        assert r.ok and r.applied, r.errors
        mt = eng.project.music_transitions[0]
        assert len(mt.variants) == 3 and eng.project.constraints["keep_tempo"]
        r = eng.apply(make_plan([op("music.select_variant", target={"transition_id": mt.id}, params={"variant": "long"}),
                                 op("music.set", target={"transition_id": mt.id}, params={"b_entry_shift_beats": 4})],
                                eng.project.revision))
        assert r.ok, r.errors
        out = root / "экспорт B.wav"
        o0, o1 = eng.project.music_transitions[0].params.overlap
        rep = export_wav(eng.project, store, ExportOptions(path=str(out), start=max(0, o0 - 3), end=o1 + 3), cache=cache)
        assert sf.info(str(out)).frames == rep.frames
        return f"BPM ≈ 120, варианты: {[v.kind for v in mt.variants]}, экспорт участка {rep.duration:.2f} с"

    def repeatability():
        p = state["a"]
        plan = compile_plan(p, store)
        n = plan.end_with_tails()
        y1 = render_range(plan, 0, n, None)
        y2 = render_range(compile_plan(p, store), 0, n, ChunkCache())
        assert np.array_equal(y1, y2)
        s0, s1 = int(10.5 * 48000), int(19.25 * 48000)
        assert np.array_equal(render_range(plan, s0, s1, None), y1[s0:s1])
        assert np.all(np.isfinite(y1))
        return f"{n} сэмплов, побитово совпадает; рендер участка = часть полного рендера"

    def gui():
        from PySide6.QtWidgets import QApplication

        from .ui import style
        from .ui.main_window import MainWindow

        app = QApplication.instance() or QApplication([sys.argv[0]])
        style.apply(app)
        w = MainWindow()
        w._quiet = True
        w.show()
        app.processEvents()
        w.ctrl.engine.dirty = False
        title = w.windowTitle()
        w.close()
        app.processEvents()
        return f"окно создано ({title}), платформа Qt: {app.platformName()}"

    check("Библиотеки", libs)
    check("Импорт WAV/FLAC/MP3, разные частоты и каналы, кириллица в путях", formats)
    check("Сценарий A: в сцене → закадровый, правки, undo/redo, сохранение, экспорт", scenario_a)
    check("Сценарий B: переход A → B, анализ долей, варианты, экспорт", scenario_b)
    check("Повторяемость рендера", repeatability)
    check("Интерфейс Qt", gui)
    ok = all(c["ok"] for c in report["checks"])
    report["ok"] = ok
    rp = root / "selftest-report.json"
    rp.write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str), "utf-8")
    print(("САМОПРОВЕРКА ПРОЙДЕНА" if ok else "САМОПРОВЕРКА НЕ ПРОЙДЕНА") + f". Отчёт: {rp}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
