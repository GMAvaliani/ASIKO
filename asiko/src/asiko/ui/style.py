"""Оформление: тёмная тема, цвета элементов таймлайна."""
from __future__ import annotations

from PySide6.QtGui import QColor, QPalette

BG = "#1e2126"
PANEL = "#262a31"
PANEL2 = "#2e333b"
TEXT = "#e6e8eb"
MUTED = "#9aa3ad"
ACCENT = "#4fa3e0"
ACCENT2 = "#e0a84f"
DANGER = "#e05a4f"
OK = "#5fbf7a"

TRACK_COLORS = ["#4fa3e0", "#9b7ee0", "#5fbf7a", "#e0a84f", "#e07fb0", "#5fc8c8"]
GROUP_COLORS = {
    "gain_db": "#e6e8eb", "pan": "#e0a84f", "hp_hz": "#4fa3e0", "lp_hz": "#5fc8c8",
    "width": "#9b7ee0", "color_mix": "#e07fb0", "direct_db": "#b0b0b0", "reverb_db": "#5fbf7a",
}


def qcolor(hex_: str, alpha: int = 255) -> QColor:
    c = QColor(hex_)
    c.setAlpha(alpha)
    return c


def apply(app) -> None:
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.ColorRole.Window, QColor(BG))
    pal.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Base, QColor(PANEL))
    pal.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL2))
    pal.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Button, QColor(PANEL2))
    pal.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT))
    pal.setColor(QPalette.ColorRole.HighlightedText, QColor("#0d1117"))
    pal.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL2))
    pal.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    pal.setColor(QPalette.ColorRole.PlaceholderText, QColor(MUTED))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor("#666d75"))
    pal.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.ButtonText, QColor("#666d75"))
    app.setPalette(pal)
    app.setStyleSheet(f"""
        QToolTip {{ color: {TEXT}; background: {PANEL2}; border: 1px solid #444; }}
        QDockWidget::title {{ background: {PANEL2}; padding: 4px 6px; }}
        QGroupBox {{ border: 1px solid #3a4049; border-radius: 4px; margin-top: 10px; padding-top: 6px; }}
        QGroupBox::title {{ subcontrol-origin: margin; left: 8px; padding: 0 3px; color: {MUTED}; }}
        QPushButton {{ padding: 4px 10px; border-radius: 3px; border: 1px solid #454c56; background: {PANEL2}; }}
        QPushButton:hover {{ border-color: {ACCENT}; }}
        QPushButton:checked {{ background: #35506a; border-color: {ACCENT}; }}
        QPushButton:disabled {{ color: #666d75; border-color: #353a41; }}
        QPushButton#scenario {{ font-size: 14px; font-weight: 600; padding: 9px 16px; background: #2b4057; border-color: {ACCENT}; }}
        QPushButton#scenario:hover {{ background: #335070; }}
        QPushButton#primary {{ background: #2b4057; border-color: {ACCENT}; font-weight: 600; }}
        QPushButton#danger {{ border-color: {DANGER}; }}
        QPushButton#small {{ padding: 1px 6px; }}
        QLabel#hint {{ color: {MUTED}; }}
        QLabel#warn {{ color: {ACCENT2}; }}
        QLabel#error {{ color: {DANGER}; }}
        QLabel#title {{ font-size: 15px; font-weight: 600; }}
        QFrame#note {{ background: #2b2a24; border: 1px solid #5a5236; border-radius: 4px; }}
        QStatusBar {{ background: {PANEL}; }}
    """)
