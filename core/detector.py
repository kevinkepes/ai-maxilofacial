"""Detectoare: baseline clasic (Etapa 1) si YOLO antrenat (Etapa 2).

Ambele respecta aceeasi interfata: primesc tensorul letterbox I ∈ [0,1]^{H×W}
si intorc o lista de Detection in coordonatele letterbox. Aplicatia nu trebuie
modificata cand trecem de la baseline la model.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .config import CLASSES, CONF_THRESHOLD, IOU_THRESHOLD, YOLO_WEIGHTS


@dataclass
class Detection:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2
    cls: str
    score: float


class Detector(Protocol):
    name: str

    def predict(self, tensor: np.ndarray) -> list[Detection]: ...


def iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(dets: list[Detection], thr: float = IOU_THRESHOLD) -> list[Detection]:
    """Non-Maximum Suppression: pastram detectia cea mai sigura dintre cele suprapuse."""
    kept: list[Detection] = []
    for d in sorted(dets, key=lambda d: d.score, reverse=True):
        if all(iou(d.box, k.box) < thr for k in kept):
            kept.append(d)
    return kept


class BaselineDetector:
    """Detector clasic, fara invatare: cauta zone radiotransparente bine delimitate.

    Chisturile si majoritatea tumorilor odontogene apar ca zone intunecate,
    rotunjite, cu margine clara in osul maxilar. Baseline-ul le aproximeaza prin:
    masca osului (Otsu) → „gauri” complet inconjurate de os → morfologie →
    filtrare dupa arie si forma; scorul creste cu circularitatea si contrastul.
    Nu clasifica tipul leziunii (intoarce „alta leziune”); serveste ca reper
    pentru flow-ul de baza si ca termen de comparatie pentru modelul din Etapa 2.
    """

    name = "Baseline clasic (fara invatare)"

    def predict(self, tensor: np.ndarray) -> list[Detection]:
        img = (tensor * 255).astype(np.uint8)
        h, w = img.shape
        smooth = cv2.GaussianBlur(img, (0, 0), sigmaX=3)
        valid = img > 8  # excludem padding-ul negru din letterbox

        # 1. Masca osului: prag Otsu calculat doar pe pixelii reali
        thr, _ = cv2.threshold(smooth[valid].reshape(-1, 1), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bone = ((smooth > thr) & valid).astype(np.uint8) * 255
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        bone = cv2.morphologyEx(bone, cv2.MORPH_CLOSE, kernel)

        # 2. „Gaurile” din os = zone intunecate complet inconjurate de os
        #    (umplem din exterior; ce ramane neatins in interiorul osului e o gaura)
        flood = bone.copy()
        ff_mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(flood, ff_mask, (0, 0), 255)
        mask = cv2.bitwise_not(flood)  # pixeli neatinsi = interior inchis, ne-os
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Contrast local, folosit pentru scor
        bg = cv2.GaussianBlur(img, (0, 0), sigmaX=25).astype(np.int16)
        diff = bg - smooth.astype(np.int16)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area, max_area = 0.0015 * h * w, 0.06 * h * w
        dets: list[Detection] = []
        for c in contours:
            area = cv2.contourArea(c)
            if not (min_area <= area <= max_area):
                continue
            perim = cv2.arcLength(c, True)
            circularity = 4 * np.pi * area / (perim * perim) if perim else 0
            if circularity < 0.45:  # leziunile sunt rotunjite, nu alungite
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            region = np.zeros_like(mask)
            cv2.drawContours(region, [c], -1, 255, -1)
            contrast = float(diff[region > 0].mean())
            score = float(np.clip(0.25 + 0.4 * circularity + contrast / 120, 0, 0.95))
            dets.append(Detection((x, y, x + bw, y + bh), "alta leziune", score))
        return nms(dets)


class YoloDetector:
    """Wrapper peste un model Ultralytics (YOLOv8/YOLO11/RT-DETR) antrenat pe CLASSES."""

    def __init__(self, weights: str):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.name = f"YOLO ({Path(weights).name})"

    def predict(self, tensor: np.ndarray) -> list[Detection]:
        img = cv2.cvtColor((tensor * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        res = self.model.predict(img, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD, verbose=False)[0]
        dets = []
        for box, cls_id, score in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist(), res.boxes.conf.tolist()):
            cls = CLASSES[int(cls_id)] if int(cls_id) < len(CLASSES) else "alta leziune"
            dets.append(Detection(tuple(box), cls, float(score)))
        return dets


def get_detector(weights: str = YOLO_WEIGHTS) -> Detector:
    """Foloseste modelul antrenat daca exista si ultralytics e instalat; altfel baseline-ul."""
    if Path(weights).exists():
        try:
            return YoloDetector(weights)
        except ImportError:
            pass
    return BaselineDetector()
