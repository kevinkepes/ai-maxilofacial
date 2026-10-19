"""Preprocesarea radiografiei: CLAHE, denoise, letterbox."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import TARGET_SIZE


@dataclass
class Letterbox:
    scale: float
    pad_x: int
    pad_y: int

    def to_original(self, box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        """Mapeaza un dreptunghi din spatiul letterbox inapoi pe imaginea originala."""
        x1, y1, x2, y2 = box
        return (
            int(round((x1 - self.pad_x) / self.scale)),
            int(round((y1 - self.pad_y) / self.scale)),
            int(round((x2 - self.pad_x) / self.scale)),
            int(round((y2 - self.pad_y) / self.scale)),
        )


def enhance(img: np.ndarray) -> np.ndarray:
    """Egalizare adaptiva a contrastului + reducere de zgomot, pastrand marginile."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    out = clahe.apply(img)
    return cv2.bilateralFilter(out, d=5, sigmaColor=40, sigmaSpace=40)


def letterbox(img: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> tuple[np.ndarray, Letterbox]:
    """Redimensioneaza pastrand proportiile si completeaza cu negru pana la `size`."""
    tw, th = size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((th, tw), dtype=img.dtype)
    px, py = (tw - nw) // 2, (th - nh) // 2
    canvas[py : py + nh, px : px + nw] = resized
    return canvas, Letterbox(scale, px, py)


def preprocess(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, Letterbox]:
    """Intoarce (imagine imbunatatita la rezolutia originala, tensor normalizat, letterbox)."""
    enhanced = enhance(img)
    boxed, lb = letterbox(enhanced)
    tensor = boxed.astype(np.float32) / 255.0  # I ∈ [0,1]^{H×W}
    return enhanced, tensor, lb
