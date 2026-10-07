"""Каталоги приложения (данные, кэш, настройки, автосохранение).

ASIKO_HOME переопределяет корень (используется тестами и переносной установкой).
"""
from __future__ import annotations

import os
from pathlib import Path

from platformdirs import user_cache_dir, user_config_dir, user_data_dir

APP = "ASIKO"


def _root(kind: str) -> Path:
    home = os.environ.get("ASIKO_HOME")
    if home:
        return Path(home) / kind
    if kind == "data":
        return Path(user_data_dir(APP, appauthor=False))
    if kind == "cache":
        return Path(user_cache_dir(APP, appauthor=False))
    return Path(user_config_dir(APP, appauthor=False))


def data_dir() -> Path:
    p = _root("data")
    p.mkdir(parents=True, exist_ok=True)
    return p


def cache_dir() -> Path:
    p = _root("cache")
    p.mkdir(parents=True, exist_ok=True)
    return p


def config_dir() -> Path:
    p = _root("config")
    p.mkdir(parents=True, exist_ok=True)
    return p


def autosave_dir() -> Path:
    p = data_dir() / "autosave"
    p.mkdir(parents=True, exist_ok=True)
    return p


def audio_cache_dir() -> Path:
    p = cache_dir() / "audio"
    p.mkdir(parents=True, exist_ok=True)
    return p


def demo_dir() -> Path:
    p = data_dir() / "demo"
    p.mkdir(parents=True, exist_ok=True)
    return p


def log_dir() -> Path:
    p = data_dir() / "logs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def setup_numba_cache() -> None:
    """numba (через librosa) кэширует JIT в каталог пакета; в упакованном
    приложении он только для чтения — направляем кэш в пользовательский каталог."""
    p = cache_dir() / "numba"
    p.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("NUMBA_CACHE_DIR", str(p))
