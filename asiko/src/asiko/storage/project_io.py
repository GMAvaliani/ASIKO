"""Сохранение и открытие проектов, автосохранение, поиск перемещённых исходников,
переносимый проект с копиями материалов."""
from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .. import __version__
from ..audio.io import AUDIO_EXTENSIONS, Cancelled, atomic_write_bytes, sha256_file
from ..model.migrate import ProjectFormatError
from ..model.project import Project

PROJECT_EXT = ".asiko"


def data_dir_for(project_path: str | os.PathLike) -> Path:
    p = Path(project_path)
    return p.with_name(p.stem + ".asiko_data")


def _rel(path: str | None, base: Path) -> str | None:
    if not path:
        return None
    try:
        return os.path.relpath(path, base)
    except ValueError:  # другой диск (Windows)
        return None


def project_to_json(project: Project, path: str | os.PathLike | None = None) -> bytes:
    d = project.to_dict()
    if path is not None:
        base = Path(path).resolve().parent
        for s in d["sources"]:
            s["rel_path"] = _rel(s.get("path"), base)
        if d.get("video"):
            d["video"]["rel_path"] = _rel(d["video"].get("path"), base)
    d["app_version"] = __version__
    d["saved_at"] = _dt.datetime.now().isoformat(timespec="seconds")
    return json.dumps(d, ensure_ascii=False, indent=1).encode("utf-8")


def save_project(project: Project, path: str | os.PathLike) -> None:
    """Атомарное сохранение: временный файл в той же папке → fsync → замена."""
    path = Path(path)
    if path.suffix.lower() != PROJECT_EXT:
        path = path.with_suffix(PROJECT_EXT)
    data = project_to_json(project, path)
    atomic_write_bytes(path, data)


def load_project(path: str | os.PathLike) -> Project:
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as e:
        raise ProjectFormatError(f"Не удалось открыть файл: {e}")
    try:
        d = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ProjectFormatError(f"Файл проекта повреждён (ошибка JSON: {e}).")
    try:
        p = Project.from_dict(d)
    except ProjectFormatError:
        raise
    except (KeyError, TypeError, ValueError) as e:
        raise ProjectFormatError(f"Файл проекта повреждён: {e}")
    base = path.resolve().parent
    # Если абсолютный путь не существует, но относительный есть — используем его.
    for s in p.sources:
        if (not s.path or not Path(s.path).is_file()) and s.rel_path:
            cand = (base / s.rel_path).resolve()
            if cand.is_file():
                s.path = str(cand)
    if p.video and (not Path(p.video.path).is_file()) and p.video.rel_path:
        cand = (base / p.video.rel_path).resolve()
        if cand.is_file():
            p.video.path = str(cand)
    return p


# --------------------------------------------------------------------------- sources check / relink


@dataclass
class SourceStatus:
    source_id: str
    name: str
    path: str
    status: str          # ok | missing | changed
    detail: str = ""


def check_sources(project: Project, verify_hash: bool = True, cancel=None, progress=None) -> list[SourceStatus]:
    out = []
    n = max(1, len(project.sources))
    for i, s in enumerate(project.sources):
        if cancel is not None and cancel():
            raise Cancelled()
        p = Path(s.path) if s.path else None
        if p is None or not p.is_file():
            out.append(SourceStatus(s.id, s.name, s.path, "missing", "файл не найден"))
        else:
            size = p.stat().st_size
            if size != s.size:
                out.append(SourceStatus(s.id, s.name, s.path, "changed", "размер файла изменился"))
            elif verify_hash and sha256_file(p, cancel=cancel) != s.sha256:
                out.append(SourceStatus(s.id, s.name, s.path, "changed", "контрольная сумма не совпадает"))
            else:
                out.append(SourceStatus(s.id, s.name, s.path, "ok"))
        if progress:
            progress((i + 1) / n)
    return out


def find_moved_sources(project: Project, folder: str | os.PathLike, source_ids: list[str] | None = None,
                       cancel=None, progress=None) -> dict[str, str]:
    """Ищет исходники в папке (рекурсивно): по имени и размеру, затем по контрольной сумме."""
    want = [s for s in project.sources if source_ids is None or s.id in source_ids]
    by_size: dict[int, list] = {}
    for s in want:
        by_size.setdefault(s.size, []).append(s)
    found: dict[str, str] = {}
    files = []
    for root, _dirs, names in os.walk(folder):
        if cancel is not None and cancel():
            raise Cancelled()
        for nm in names:
            if os.path.splitext(nm)[1].lower() in AUDIO_EXTENSIONS:
                files.append(os.path.join(root, nm))
    # сначала совпадения по имени
    files.sort(key=lambda f: 0 if any(Path(f).name == Path(s.path).name for s in want) else 1)
    total = max(1, len(files))
    for i, f in enumerate(files):
        if cancel is not None and cancel():
            raise Cancelled()
        if progress:
            progress((i + 1) / total)
        try:
            size = os.path.getsize(f)
        except OSError:
            continue
        cands = [s for s in by_size.get(size, []) if s.id not in found]
        if not cands:
            continue
        digest = sha256_file(f, cancel=cancel)
        for s in cands:
            if s.sha256 == digest:
                found[s.id] = os.path.abspath(f)
        if len(found) == len(want):
            break
    return found


def collect_project(project: Project, project_path: str | os.PathLike | None, target_dir: str | os.PathLike,
                    cancel=None, progress=None) -> Path:
    """Переносимый проект: копирует используемые исходники и данные фиксации в папку."""
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    media = target / "media"
    media.mkdir(exist_ok=True)
    p = project.clone()
    used = {c.source_id for c in p.all_clips()}
    total = max(1, len(used) + 1)
    done = 0
    names: set[str] = set()
    for s in p.sources:
        if s.id not in used:
            continue
        if cancel is not None and cancel():
            raise Cancelled()
        if not s.path or not Path(s.path).is_file():
            raise FileNotFoundError(f"Исходник «{s.name}» не найден — найдите его перед сборкой проекта.")
        base = Path(s.path).name
        nm = base
        k = 1
        while nm.lower() in names:
            nm = f"{Path(base).stem} ({k}){Path(base).suffix}"
            k += 1
        names.add(nm.lower())
        dst = media / nm
        shutil.copy2(s.path, dst)
        if sha256_file(dst) != s.sha256:
            raise IOError(f"Копия «{nm}» не совпадает с исходником по контрольной сумме.")
        s.path = str(dst.resolve())
        done += 1
        if progress:
            progress(done / total)
    p.sources = [s for s in p.sources if s.id in used]
    out_path = target / (Path(project_path).stem + PROJECT_EXT if project_path else f"{p.name}{PROJECT_EXT}")
    if project_path is not None and p.frozen:
        src_data = data_dir_for(project_path)
        dst_data = data_dir_for(out_path)
        for fr in p.frozen:
            sp = src_data / fr.file
            if sp.is_file():
                dp = dst_data / fr.file
                dp.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(sp, dp)
    if p.video and Path(p.video.path).is_file():
        dst = media / Path(p.video.path).name
        shutil.copy2(p.video.path, dst)
        p.video.path = str(dst.resolve())
    save_project(p, out_path)
    if progress:
        progress(1.0)
    return out_path


# --------------------------------------------------------------------------- autosave


@dataclass
class AutosaveEntry:
    path: Path
    project_id: str
    name: str
    original: str | None
    saved_at: float


class Autosaver:
    def __init__(self, directory: str | os.PathLike):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _paths(self, project_id: str) -> tuple[Path, Path]:
        return self.dir / f"{project_id}{PROJECT_EXT}", self.dir / f"{project_id}.meta.json"

    def write(self, project: Project, original: str | None) -> Path:
        p, m = self._paths(project.id)
        atomic_write_bytes(p, project_to_json(project, None))
        meta = {"project_id": project.id, "name": project.name, "original": original, "saved_at": time.time()}
        atomic_write_bytes(m, json.dumps(meta, ensure_ascii=False).encode("utf-8"))
        return p

    def clear(self, project_id: str) -> None:
        for p in self._paths(project_id):
            try:
                p.unlink()
            except OSError:
                pass

    def entries(self) -> list[AutosaveEntry]:
        out = []
        for m in self.dir.glob("*.meta.json"):
            try:
                meta = json.loads(m.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            p = self.dir / f"{meta.get('project_id')}{PROJECT_EXT}"
            if not p.exists():
                continue
            orig = meta.get("original")
            # предлагаем только если автосохранение новее сохранённого файла
            if orig and Path(orig).exists() and Path(orig).stat().st_mtime >= meta.get("saved_at", 0):
                continue
            out.append(AutosaveEntry(path=p, project_id=meta["project_id"], name=meta.get("name", ""),
                                     original=orig, saved_at=float(meta.get("saved_at", 0))))
        return sorted(out, key=lambda e: -e.saved_at)
