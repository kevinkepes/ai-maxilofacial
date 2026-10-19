"""Flow-ul de baza: incarcare → validare → preprocesare → detectie → localizare → rezultat."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime

import numpy as np

from .anatomy import locate
from .config import CONF_THRESHOLD
from .detector import Detector, get_detector
from .io_utils import ValidationResult, load_image, validate
from .preprocess import preprocess


@dataclass
class AnalysisResult:
    analysis_id: str
    patient_code: str
    filename: str
    created_at: str
    model: str
    image_size: tuple[int, int]
    duration_s: float
    validation: ValidationResult
    findings: list[dict] = field(default_factory=list)
    original: np.ndarray | None = None
    enhanced: np.ndarray | None = None

    def to_report(self) -> dict:
        """Raportul exportabil (fara imagini)."""
        return {
            "analysis_id": self.analysis_id,
            "patient_code": self.patient_code,
            "filename": self.filename,
            "created_at": self.created_at,
            "model": self.model,
            "image_size": list(self.image_size),
            "duration_s": round(self.duration_s, 3),
            "warnings": self.validation.warnings,
            "findings": self.findings,
            "disclaimer": "Instrument de suport la decizie. Diagnosticul final apartine medicului.",
        }


def analyze(
    data: bytes,
    filename: str,
    patient_code: str = "",
    detector: Detector | None = None,
) -> AnalysisResult:
    t0 = time.perf_counter()
    img = load_image(data, filename)
    h, w = img.shape[:2]
    validation = validate(img)
    detector = detector or get_detector()

    result = AnalysisResult(
        analysis_id=uuid.uuid4().hex[:8],
        patient_code=patient_code.strip() or "anonim",
        filename=filename,
        created_at=datetime.now().isoformat(timespec="seconds"),
        model=detector.name,
        image_size=(w, h),
        duration_s=0.0,
        validation=validation,
        original=img,
    )
    if not validation.ok:
        result.duration_s = time.perf_counter() - t0
        return result

    enhanced, tensor, lb = preprocess(img)
    result.enhanced = enhanced

    dets = [d for d in detector.predict(tensor) if d.score >= CONF_THRESHOLD]
    for i, d in enumerate(sorted(dets, key=lambda d: d.score, reverse=True), start=1):
        x1, y1, x2, y2 = lb.to_original(d.box)
        box = (max(0, x1), max(0, y1), min(w - 1, x2), min(h - 1, y2))
        result.findings.append(
            {
                "id": i,
                "cls": d.cls,
                "score": round(d.score, 3),
                "box": list(box),
                "region": locate(box, w, h),
                "status": "nevalidat",  # nevalidat / confirmat / respins
                "note": "",
            }
        )
    result.duration_s = time.perf_counter() - t0
    return result


__all__ = ["analyze", "AnalysisResult", "asdict"]
