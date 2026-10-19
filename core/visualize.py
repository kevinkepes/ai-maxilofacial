"""Desenarea detectiilor pe radiografie."""
from __future__ import annotations

import cv2
import numpy as np

from .config import CLASS_COLORS


def draw(img_gray: np.ndarray, findings: list[dict]) -> np.ndarray:
    """Intoarce o imagine RGB cu dreptunghiuri, numere si scoruri."""
    out = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    thick = max(2, img_gray.shape[1] // 600)
    font_scale = max(0.6, img_gray.shape[1] / 2400)
    for f in findings:
        x1, y1, x2, y2 = f["box"]
        color = CLASS_COLORS.get(f["cls"], (255, 255, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thick)
        label = f"#{f['id']} {f['score']:.0%}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thick)
        ty = max(y1 - 6, th + 6)
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, label, (x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thick)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
