"""Генерирует значок приложения (PNG и ICO с PNG-содержимым) средствами Qt."""
from __future__ import annotations

import struct
import sys
from pathlib import Path

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QGuiApplication, QImage, QLinearGradient, QPainter, QPainterPath, QPen

HERE = Path(__file__).resolve().parent


def draw(size: int) -> QImage:
    img = QImage(size, size, QImage.Format.Format_ARGB32)
    img.fill(Qt.GlobalColor.transparent)
    p = QPainter(img)
    p.setRenderHint(QPainter.RenderHint.Antialiasing)
    r = QRectF(size * 0.06, size * 0.06, size * 0.88, size * 0.88)
    g = QLinearGradient(r.topLeft(), r.bottomRight())
    g.setColorAt(0, QColor("#2b4057"))
    g.setColorAt(1, QColor("#1e2126"))
    p.setBrush(g)
    p.setPen(QPen(QColor("#4fa3e0"), size * 0.03))
    p.drawRoundedRect(r, size * 0.18, size * 0.18)
    # две волны: «в сцене» (узкая) → «закадровая» (широкая)
    for i, (col, amp) in enumerate((("#e0a84f", 0.10), ("#4fa3e0", 0.22))):
        path = QPainterPath()
        n = 64
        for k in range(n + 1):
            x = size * (0.18 + 0.64 * k / n)
            env = (k / n) if i == 1 else (1 - k / n)
            import math

            y = size * 0.5 + size * amp * env * math.sin(k / n * math.pi * 6)
            if k == 0:
                path.moveTo(QPointF(x, y))
            else:
                path.lineTo(QPointF(x, y))
        p.setPen(QPen(QColor(col), size * 0.045, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawPath(path)
    p.end()
    return img


def png_bytes(img: QImage) -> bytes:
    ba = QByteArray()
    buf = QBuffer(ba)
    buf.open(QIODevice.OpenModeFlag.WriteOnly)
    img.save(buf, "PNG")
    return bytes(ba.data())


def main() -> None:
    app = QGuiApplication.instance() or QGuiApplication(sys.argv)
    draw(512).save(str(HERE / "icon.png"))
    sizes = [16, 32, 48, 64, 128, 256]
    imgs = [png_bytes(draw(s)) for s in sizes]
    header = struct.pack("<HHH", 0, 1, len(sizes))
    entries = b""
    offset = 6 + 16 * len(sizes)
    for s, data in zip(sizes, imgs):
        entries += struct.pack("<BBBBHHII", s % 256, s % 256, 0, 0, 1, 32, len(data), offset)
        offset += len(data)
    (HERE / "icon.ico").write_bytes(header + entries + b"".join(imgs))
    # ICNS (macOS): элементы с PNG-данными ic07…ic10 (128, 256, 512, 1024)
    body = b""
    for tag, s in ((b"ic07", 128), (b"ic08", 256), (b"ic09", 512), (b"ic10", 1024)):
        data = png_bytes(draw(s))
        body += tag + struct.pack(">I", len(data) + 8) + data
    (HERE / "icon.icns").write_bytes(b"icns" + struct.pack(">I", len(body) + 8) + body)
    print("icons written")


if __name__ == "__main__":
    main()
