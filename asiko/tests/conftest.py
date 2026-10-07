"""Общие фикстуры: изолированный ASIKO_HOME, пути с кириллицей и пробелами, генерация сигналов."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("ASIKO_NO_KEYRING", "1")
_HOME = Path(os.environ.get("PYTEST_ASIKO_HOME", "")) if os.environ.get("PYTEST_ASIKO_HOME") else None


@pytest.fixture(scope="session", autouse=True)
def asiko_home(tmp_path_factory):
    home = _HOME or tmp_path_factory.mktemp("asiko home")
    os.environ["ASIKO_HOME"] = str(home)
    os.environ["NUMBA_CACHE_DIR"] = os.environ.get("NUMBA_CACHE_DIR") or str(Path(home) / "numba")
    yield home


@pytest.fixture
def work(tmp_path) -> Path:
    """Рабочая папка с кириллицей и пробелами в пути."""
    p = tmp_path / "Проекты ASIKO" / "тест 1"
    p.mkdir(parents=True)
    return p


@pytest.fixture
def store(tmp_path):
    from asiko.audio.sources import SourceStore

    return SourceStore(tmp_path / "кэш аудио", 48000)


def write(path: Path, data, sr: int, subtype: str = "PCM_24", fmt: str | None = None) -> Path:
    import soundfile as sf

    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), data, sr, subtype=subtype, format=fmt)
    return path


class Builder:
    """Упрощённая сборка проекта через ту же систему команд, что и в приложении."""

    def __init__(self, store, sr: int = 48000):
        from asiko.commands.engine import CommandEngine
        from asiko.model.project import Project

        self.store = store
        self.engine = CommandEngine(Project(sample_rate=sr))

    @property
    def project(self):
        return self.engine.project

    def apply(self, ops, **kw):
        from asiko.commands.schema import make_plan

        r = self.engine.apply(make_plan(ops, self.engine.project.revision), **kw)
        assert r.ok, r.errors
        return r

    def add_file(self, path, start: float = 0.0, track_id: str | None = None):
        from asiko.commands.schema import op

        src = self.store.import_file(path)
        tgt = {"source_id": src.id}
        if track_id:
            tgt["track_id"] = track_id
        r = self.apply([op("source.add", params={"source": src.to_dict()}),
                        op("clip.add", target=tgt, time={"coord": "timeline", "unit": "s", "start": start})])
        clip_id = [c for c in r.created if c.startswith("clp_")][0]
        return src, self.project.clip(clip_id)

    def space(self, track_id, start, end, **params):
        from asiko.commands.schema import op

        cons = params.pop("constraints", None)
        r = self.apply([op("space.create", target={"track_id": track_id},
                           time={"coord": "timeline", "unit": "s", "start": start, "end": end},
                           params=params or None, constraints=cons)])
        return self.project.space_transition(r.select)


@pytest.fixture
def builder(store):
    return Builder(store)


@pytest.fixture(scope="session")
def qapp():
    from PySide6.QtWidgets import QApplication

    from asiko.ui import style

    app = QApplication.instance() or QApplication(sys.argv[:1])
    style.apply(app)
    yield app


@pytest.fixture(autouse=True)
def no_modal_message_boxes(monkeypatch):
    """Модальные окна не должны блокировать автотесты: фиксируем вызовы и отвечаем «Нет/OK»."""
    try:
        from PySide6.QtWidgets import QMessageBox
    except ImportError:
        yield []
        return
    calls = []

    def make(kind, answer):
        def f(*a, **k):
            calls.append((kind, a[1:3] if len(a) > 2 else a))
            return answer
        return staticmethod(f)

    monkeypatch.setattr(QMessageBox, "question", make("question", QMessageBox.StandardButton.No))
    monkeypatch.setattr(QMessageBox, "warning", make("warning", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "information", make("information", QMessageBox.StandardButton.Ok))
    monkeypatch.setattr(QMessageBox, "critical", make("critical", QMessageBox.StandardButton.Ok))
    yield calls
