"""Параметры пространственного состояния, пресеты, группы параметров."""
from __future__ import annotations

import copy

# Порядок важен: используется в интерфейсе и отчётах.
SPACE_PARAMS: tuple[str, ...] = (
    "gain_db",
    "pan",
    "hp_hz",
    "lp_hz",
    "width",
    "color_mix",
    "direct_db",
    "reverb_db",
)

# domain: как интерполировать значение между состояниями
#   db     — линейно в децибелах
#   log    — линейно по логарифму частоты
#   linear — линейно
PARAM_INFO: dict[str, dict] = {
    "gain_db": dict(label="Усиление", unit="дБ", min=-60.0, max=12.0, domain="db", neutral=0.0),
    "pan": dict(label="Панорама", unit="", min=-1.0, max=1.0, domain="linear", neutral=0.0),
    "hp_hz": dict(label="Срез низких (ФВЧ)", unit="Гц", min=20.0, max=4000.0, domain="log", neutral=20.0),
    "lp_hz": dict(label="Срез высоких (ФНЧ)", unit="Гц", min=200.0, max=20000.0, domain="log", neutral=20000.0),
    "width": dict(label="Стереоширина", unit="×", min=0.0, max=2.0, domain="linear", neutral=1.0),
    "color_mix": dict(label="Доля обработанного звука", unit="доля", min=0.0, max=1.0, domain="linear", neutral=1.0),
    "direct_db": dict(label="Прямой звук", unit="дБ", min=-60.0, max=6.0, domain="db", neutral=0.0),
    "reverb_db": dict(label="Комната (посыл)", unit="дБ", min=-80.0, max=6.0, domain="db", neutral=-80.0),
}

REVERB_OFF_DB = -80.0

ROOMS: dict[str, str] = {
    "none": "Без комнаты",
    "small_room": "Маленькая комната",
    "room": "Комната",
    "hall": "Большое помещение",
}

CURVE_SHAPES: dict[str, str] = {
    "linear": "Линейная",
    "smooth": "Плавная (S)",
    "ease_in": "Медл. начало",
    "ease_out": "Быстр. начало",
}

FILTER_SLOPES = (12, 24)

# Группы параметров для текстовых команд и интерфейса.
PARAM_GROUPS: dict[str, tuple[str, ...]] = {
    "filter": ("hp_hz", "lp_hz"),
    "color": ("hp_hz", "lp_hz", "color_mix"),
    "width": ("width",),
    "pan": ("pan",),
    "room": ("reverb_db",),
    "level": ("gain_db",),
    "direct": ("direct_db",),
}

GROUP_LABELS: dict[str, str] = {
    "filter": "Раскрытие частот",
    "color": "Окраска (эффект радио)",
    "width": "Стереоширина",
    "pan": "Панорама",
    "room": "Комнатная обработка",
    "level": "Громкость",
    "direct": "Прямой звук",
}

# Значения нейтрального (необработанного) состояния.
NEUTRAL_VALUES: dict[str, float] = {p: PARAM_INFO[p]["neutral"] for p in SPACE_PARAMS}

BUILTIN_PRESETS: dict[str, dict] = {
    "radio_room": {
        "name": "Радио в комнате",
        "values": {"gain_db": -3.0, "pan": 0.6, "hp_hz": 180.0, "lp_hz": 5000.0, "width": 0.0,
                   "color_mix": 1.0, "direct_db": -2.0, "reverb_db": -11.0},
        "room": "room",
        "slope": 12,
        "limitation": None,
    },
    "phone": {
        "name": "Телефонный динамик",
        "values": {"gain_db": -4.0, "pan": 0.0, "hp_hz": 400.0, "lp_hz": 3400.0, "width": 0.0,
                   "color_mix": 1.0, "direct_db": 0.0, "reverb_db": REVERB_OFF_DB},
        "room": "none",
        "slope": 24,
        "limitation": None,
    },
    "distant": {
        "name": "Удалённый источник",
        "values": {"gain_db": -8.0, "pan": 0.0, "hp_hz": 120.0, "lp_hz": 4500.0, "width": 0.3,
                   "color_mix": 1.0, "direct_db": -8.0, "reverb_db": -4.0},
        "room": "hall",
        "slope": 12,
        "limitation": None,
    },
    "muffled": {
        "name": "Приглушённый звук за препятствием",
        "values": {"gain_db": -5.0, "pan": 0.0, "hp_hz": 40.0, "lp_hz": 700.0, "width": 0.4,
                   "color_mix": 1.0, "direct_db": -1.0, "reverb_db": -14.0},
        "room": "small_room",
        "slope": 12,
        "limitation": None,
    },
    "offscreen": {
        "name": "Полное закадровое звучание",
        "values": dict(NEUTRAL_VALUES),
        "room": "none",
        "slope": 12,
        "limitation": None,
    },
}

# Ограничение, которое показывается при выборе сценария (п. 6 ТЗ).
SOURCE_LIMITATION_NOTE = (
    "Обработка добавляет пространство и окраску, но не убирает их из записи. "
    "Если исходник уже содержит сильную реверберацию или узкополосную окраску "
    "(телефон, рация), чистое закадровое звучание из него восстановить нельзя."
)


def preset_names(user_presets: dict | None = None) -> dict[str, str]:
    out = {k: v["name"] for k, v in BUILTIN_PRESETS.items()}
    for k, v in (user_presets or {}).items():
        out[k] = v.get("name", k)
    return out


def get_preset(key: str, user_presets: dict | None = None) -> dict | None:
    if user_presets and key in user_presets:
        return copy.deepcopy(user_presets[key])
    if key in BUILTIN_PRESETS:
        return copy.deepcopy(BUILTIN_PRESETS[key])
    return None


def clamp_param(name: str, value: float) -> float:
    info = PARAM_INFO[name]
    return float(min(max(value, info["min"]), info["max"]))


def check_param_value(name: str, value) -> str | None:
    """Возвращает текст ошибки или None."""
    if name not in PARAM_INFO:
        return f"неизвестный параметр «{name}»"
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return f"параметр «{name}» должен быть числом"
    import math

    if not math.isfinite(float(value)):
        return f"параметр «{name}»: недопустимое значение"
    info = PARAM_INFO[name]
    if not (info["min"] - 1e-9 <= float(value) <= info["max"] + 1e-9):
        return (f"параметр «{info['label']}»: {value} вне диапазона "
                f"{info['min']}…{info['max']} {info['unit']}")
    return None


def format_param(name: str, value: float) -> str:
    info = PARAM_INFO[name]
    if name == "pan":
        if abs(value) < 0.005:
            return "центр"
        side = "справа" if value > 0 else "слева"
        return f"{abs(value) * 100:.0f}% {side}"
    if name == "width":
        return f"{value * 100:.0f}%"
    if name == "color_mix":
        return f"{value * 100:.0f}%"
    if name == "reverb_db" and value <= REVERB_OFF_DB + 0.01:
        return "выкл"
    if name == "hp_hz" and value <= 20.01:
        return "выкл"
    if name == "lp_hz" and value >= 19999.0:
        return "выкл"
    if info["unit"] == "Гц":
        return f"{value:.0f} Гц" if value < 1000 else f"{value / 1000:.2f} кГц"
    return f"{value:+.1f} дБ" if info["unit"] == "дБ" else f"{value:g}"
