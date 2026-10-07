"""15. Запуск упакованного приложения (если сборка есть в dist/): самопроверка собранного бинарника."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent


def _exe() -> Path | None:
    cands = [ROOT / "dist" / "ASIKO" / "ASIKO", ROOT / "dist" / "ASIKO" / "ASIKO.exe"]
    cands += list((ROOT / "dist").glob("ASIKO-*/ASIKO.app/Contents/MacOS/ASIKO"))
    return next((c for c in cands if c.exists()), None)


@pytest.mark.packaged
def test_packaged_selftest(tmp_path):
    exe = _exe()
    if exe is None:
        pytest.skip("сборка не найдена — выполните python packaging/build.py")
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([str(exe), "--self-test", str(tmp_path / "проверка сборки")], env=env, timeout=1800)
    report = json.loads((tmp_path / "проверка сборки" / "Самопроверка ASIKO" / "selftest-report.json").read_text("utf-8"))
    assert proc.returncode == 0 and report["ok"] and report["frozen"], [c for c in report["checks"] if not c["ok"]]
    assert {c["name"].split(":")[0] for c in report["checks"]} >= {"Библиотеки", "Интерфейс Qt"}


def test_version_flag():
    out = subprocess.run([sys.executable, "-m", "asiko", "--version"], capture_output=True, text=True,
                         env=dict(os.environ, PYTHONPATH=str(ROOT / "src")), timeout=60)
    assert out.returncode == 0 and "ASIKO" in out.stdout
