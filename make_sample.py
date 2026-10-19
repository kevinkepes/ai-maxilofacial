"""Genereaza o panoramica SINTETICA pentru testarea flow-ului (nu sunt date reale).

Rulare: python make_sample.py  → samples/panoramica_sintetica.png
"""
from pathlib import Path

import cv2
import numpy as np


def make(seed: int = 7, w: int = 2400, h: int = 1200) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 40, np.uint8)

    def center(y0, amp, x):  # linia mediana a unei arcade
        t = (x / w - 0.5) * 2
        return int(y0 + amp * h * (1 - t * t))

    upper, lower = (int(h * 0.30), -0.06), (int(h * 0.72), 0.08)

    # arcadele maxilare (benzi osoase curbe)
    for (y0, amp), val in [(upper, 150), (lower, 165)]:
        for x in range(int(w * 0.08), int(w * 0.92)):
            yc = center(y0, amp, x)
            cv2.line(img, (x, yc - 120), (x, yc + 120), val, 1)

    # dinti
    for i in range(14):
        x = int(w * (0.2 + i * 0.043))
        cv2.ellipse(img, (x, center(*upper, x) + 95), (26, 70), 0, 0, 360, 215, -1)
        cv2.ellipse(img, (x, center(*lower, x) - 95), (26, 75), 0, 0, 360, 220, -1)

    # doua leziuni radiotransparente (zone intunecate, rotunjite, cu margine clara)
    a = (int(w * 0.28), center(*lower, int(w * 0.28)) + 25)  # mandibula, dreapta pacientului
    b = (int(w * 0.72), center(*upper, int(w * 0.72)) - 25)  # maxilar, stanga pacientului
    cv2.ellipse(img, a, (80, 55), 0, 0, 360, 70, -1)
    cv2.ellipse(img, a, (80, 55), 0, 0, 360, 190, 4)
    cv2.circle(img, b, 50, 75, -1)
    cv2.circle(img, b, 50, 185, 3)

    img = cv2.GaussianBlur(img, (0, 0), 3)
    noise = rng.normal(0, 6, img.shape)
    return np.clip(img + noise, 0, 255).astype(np.uint8)


if __name__ == "__main__":
    Path("samples").mkdir(exist_ok=True)
    out = Path("samples/panoramica_sintetica.png")
    cv2.imwrite(str(out), make())
    print(f"Scris: {out}")
