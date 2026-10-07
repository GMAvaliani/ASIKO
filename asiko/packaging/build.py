"""Автоматизированная сборка дистрибутива ASIKO для текущей ОС.

    python packaging/build.py            # сборка + самопроверка собранного приложения + архив
    python packaging/build.py --no-selftest

Результат: dist/ASIKO-<версия>-<ос>-<арх>.(tar.gz|zip|dmg) и dist/selftest-report.json.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from asiko import __version__  # noqa: E402

BUNDLED = ["PySide6", "PySide6_Essentials", "PySide6_Addons", "shiboken6", "numpy", "scipy", "soundfile", "soxr",
           "sounddevice", "librosa", "numba", "llvmlite", "pyloudnorm", "keyring", "platformdirs", "anthropic",
           "httpx2", "pydantic", "pydantic_core", "certifi", "cffi", "joblib", "lazy_loader", "msgpack", "pooch",
           "decorator", "jiter", "anyio", "sniffio", "typing_extensions", "requests", "urllib3", "idna",
           "charset_normalizer", "packaging", "jaraco.classes", "jaraco.functools", "jaraco.context",
           "more_itertools", "SecretStorage", "jeepney", "cryptography", "pywin32-ctypes", "threadpoolctl",
           "docstring_parser", "typing_inspection", "annotated_types", "distro", "h11", "httpcore", "pycparser"]


def os_tag() -> str:
    s = sys.platform
    name = "windows" if s == "win32" else "macos" if s == "darwin" else "linux"
    arch = platform.machine().lower().replace("amd64", "x86_64")
    return f"{name}-{arch}"


def run(cmd, **kw):
    print("+", " ".join(map(str, cmd)), flush=True)
    subprocess.run(cmd, check=True, **kw)


def collect_licenses(dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "docs" / "THIRD_PARTY_LICENSES.md", dest / "THIRD_PARTY_LICENSES.md")
    for name in BUNDLED:
        try:
            dist = metadata.distribution(name)
        except metadata.PackageNotFoundError:
            continue
        files = [f for f in (dist.files or []) if any(k in f.name.upper() for k in ("LICENSE", "COPYING", "NOTICE"))]
        out = dest / dist.metadata["Name"]
        for f in files:
            src = Path(dist.locate_file(f))
            if src.is_file():
                out.mkdir(exist_ok=True)
                shutil.copy2(src, out / f.name)
        lic = dist.metadata.get("License-Expression") or dist.metadata.get("License") or ""
        if lic and not files:
            out.mkdir(exist_ok=True)
            (out / "LICENSE-metadata.txt").write_text(f"{dist.metadata['Name']} {dist.version}\n{lic}\n", "utf-8")


def make_demo(dest: Path) -> None:
    home = Path(tempfile.mkdtemp(prefix="asiko-demo-home-"))
    env = dict(os.environ, ASIKO_HOME=str(home), PYTHONPATH=str(ROOT / "src"), QT_QPA_PLATFORM="offscreen")
    code = ("from pathlib import Path; from asiko.demo import ensure_demo; "
            f"print(ensure_demo(Path(r'{dest}')))")
    run([sys.executable, "-c", code], env=env)
    shutil.rmtree(home, ignore_errors=True)


def selftest(exe: Path, out: Path) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="asiko-selftest-"))
    env = dict(os.environ, QT_QPA_PLATFORM="offscreen")
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([str(exe), "--self-test", str(tmp)], env=env, timeout=1800)
    rp = tmp / "Самопроверка ASIKO" / "selftest-report.json"
    report = json.loads(rp.read_text("utf-8")) if rp.exists() else {"ok": False, "error": "отчёт не создан"}
    report["exit_code"] = proc.returncode
    report["executable"] = str(exe)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), "utf-8")
    shutil.rmtree(tmp, ignore_errors=True)
    return report


def main() -> int:
    no_selftest = "--no-selftest" in sys.argv
    dist = ROOT / "dist"
    build = ROOT / "build"
    for d in (dist, build):
        shutil.rmtree(d, ignore_errors=True)
    run([sys.executable, "-m", "PyInstaller", str(ROOT / "packaging" / "asiko.spec"), "--noconfirm",
         "--distpath", str(dist), "--workpath", str(build)])
    tag = f"ASIKO-{__version__}-{os_tag()}"
    if sys.platform == "darwin":
        stage = dist / tag
        stage.mkdir()
        shutil.move(str(dist / "ASIKO.app"), str(stage / "ASIKO.app"))
        exe = stage / "ASIKO.app" / "Contents" / "MacOS" / "ASIKO"
    else:
        stage = dist / "ASIKO"
        exe = stage / ("ASIKO.exe" if sys.platform == "win32" else "ASIKO")
    collect_licenses(stage / "licenses")
    make_demo(stage / "Демо-проекты")
    shutil.copy2(ROOT / "README.md", stage / "README.md")
    if not no_selftest:
        rep = selftest(exe, dist / "selftest-report.json")
        print("Самопроверка упакованного приложения:", "ПРОЙДЕНА" if rep.get("ok") and rep["exit_code"] == 0 else "НЕ ПРОЙДЕНА")
        if not (rep.get("ok") and rep["exit_code"] == 0):
            return 1
    if sys.platform == "win32":
        archive = dist / f"{tag}.zip"
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in stage.rglob("*"):
                z.write(p, Path(tag) / p.relative_to(stage))
    elif sys.platform == "darwin":
        archive = dist / f"{tag}.dmg"
        try:
            run(["hdiutil", "create", "-volname", "ASIKO", "-srcfolder", str(stage), "-ov", "-format", "UDZO",
                 str(archive)])
        except Exception:
            archive = Path(shutil.make_archive(str(dist / tag), "zip", root_dir=dist, base_dir=tag))
    else:
        archive = dist / f"{tag}.tar.gz"
        with tarfile.open(archive, "w:gz") as t:
            t.add(stage, arcname=tag)
    size = archive.stat().st_size / 1e6
    print(f"Дистрибутив: {archive} ({size:.0f} МБ)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
