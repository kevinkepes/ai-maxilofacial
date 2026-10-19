"""Localizare anatomica aproximativa a unei detectii pe panoramica.

Conventie radiologica: pe panoramica, partea DREAPTA a pacientului apare in
STANGA imaginii. Maxilarul este in jumatatea superioara a arcadei, mandibula
in cea inferioara. Pragurile sunt euristice si vor fi rafinate in Etapa 2
prin segmentarea regiunii maxilare.
"""
from __future__ import annotations


def locate(box: tuple[int, int, int, int], img_w: int, img_h: int) -> str:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2 / img_w, (y1 + y2) / 2 / img_h

    jaw = "maxilar" if cy < 0.5 else "mandibula"
    side = "dreapta" if cx < 0.5 else "stanga"  # dreapta pacientului = stanga imaginii
    position = "anterior" if abs(cx - 0.5) < 0.15 else "posterior"
    return f"{jaw}, {side}, {position}"
