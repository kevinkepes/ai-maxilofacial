"""Incarcarea si validarea radiografiilor."""
from __future__ import annotations

import io
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .config import ALLOWED_EXT, MAX_ASPECT, MIN_ASPECT, MIN_WIDTH


@dataclass
class ValidationResult:
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def load_image(data: bytes, filename: str) -> np.ndarray:
    """Citeste PNG/JPG/DICOM din bytes si intoarce o imagine grayscale uint8."""
    ext = Path(filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        raise ValueError(f"Format neacceptat: {ext}. Folositi PNG, JPG sau DICOM.")

    if ext == ".dcm":
        import pydicom

        ds = pydicom.dcmread(io.BytesIO(data))
        arr = ds.pixel_array.astype(np.float32)
        if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
            arr = arr.max() - arr  # inversam ca oasele sa fie albe
        arr = (arr - arr.min()) / max(float(arr.max() - arr.min()), 1e-6)
        return (arr * 255).astype(np.uint8)

    buf = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Fisierul nu a putut fi citit ca imagine.")
    return img


def validate(img: np.ndarray) -> ValidationResult:
    """Verifica daca imaginea arata ca o radiografie panoramica utilizabila."""
    res = ValidationResult(ok=True)
    h, w = img.shape[:2]
    if w < MIN_WIDTH:
        res.errors.append(f"Rezolutie prea mica: {w}px latime (minim {MIN_WIDTH}px).")
    aspect = w / h
    if not (MIN_ASPECT <= aspect <= MAX_ASPECT):
        res.errors.append(
            f"Raport latime/inaltime {aspect:.2f} — nu pare o panoramica "
            f"(asteptat {MIN_ASPECT}–{MAX_ASPECT})."
        )
    std = float(img.std())
    if std < 10:
        res.errors.append("Imaginea este aproape uniforma (fara continut util).")
    elif std < 25:
        res.warnings.append("Contrast scazut — rezultatele pot fi mai putin sigure.")
    res.ok = not res.errors
    return res
