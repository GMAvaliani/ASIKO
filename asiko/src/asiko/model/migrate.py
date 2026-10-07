"""Проверка и миграция формата проекта."""
from __future__ import annotations

FORMAT_NAME = "asiko.project"
FORMAT_VERSION = 1


class ProjectFormatError(ValueError):
    pass


def migrate(d: dict) -> dict:
    if not isinstance(d, dict):
        raise ProjectFormatError("Файл проекта повреждён: ожидался объект JSON")
    fmt = d.get("format", FORMAT_NAME)
    if fmt != FORMAT_NAME:
        raise ProjectFormatError(f"Это не проект ASIKO (format={fmt!r})")
    ver = int(d.get("format_version", 1))
    if ver > FORMAT_VERSION:
        raise ProjectFormatError(
            f"Проект создан более новой версией ASIKO (формат {ver}, поддерживается {FORMAT_VERSION})")
    # Версия 1 — текущая. Будущие миграции добавляются здесь последовательно.
    d["format_version"] = FORMAT_VERSION
    return d
