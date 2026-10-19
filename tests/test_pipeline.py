import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import analyze  # noqa: E402
from core.anatomy import locate  # noqa: E402
from core.detector import Detection, nms  # noqa: E402
from core.preprocess import letterbox  # noqa: E402
from make_sample import make  # noqa: E402


def png(img: np.ndarray) -> bytes:
    return cv2.imencode(".png", img)[1].tobytes()


def test_full_flow_finds_synthetic_lesions():
    res = analyze(png(make()), "sample.png", "P-TEST")
    assert res.validation.ok
    assert len(res.findings) >= 1
    for f in res.findings:
        x1, y1, x2, y2 = f["box"]
        assert 0 <= x1 < x2 <= 2400 and 0 <= y1 < y2 <= 1200
        assert 0 <= f["score"] <= 1
    rep = res.to_report()
    assert rep["patient_code"] == "P-TEST" and "disclaimer" in rep


def test_rejects_non_panoramic():
    square = np.random.default_rng(0).integers(0, 255, (1000, 1000), dtype=np.uint8)
    res = analyze(png(square), "x.png")
    assert not res.validation.ok and res.findings == []


def test_rejects_bad_extension():
    with pytest.raises(ValueError):
        analyze(b"abc", "x.gif")


def test_letterbox_roundtrip():
    img = np.zeros((1200, 2400), np.uint8)
    _, lb = letterbox(img)
    assert lb.to_original((lb.pad_x, lb.pad_y, lb.pad_x + 640, lb.pad_y + 320)) == (0, 0, 1200, 600)


def test_nms_removes_overlaps():
    d = [Detection((0, 0, 10, 10), "a", 0.9), Detection((1, 1, 10, 10), "a", 0.5), Detection((50, 50, 60, 60), "a", 0.4)]
    assert len(nms(d)) == 2


def test_anatomy_convention():
    # stanga imaginii, jos = mandibula dreapta, posterior
    assert locate((100, 800, 200, 900), 2400, 1200) == "mandibula, dreapta, posterior"
    assert locate((1150, 200, 1250, 300), 2400, 1200) == "maxilar, stanga, anterior"
