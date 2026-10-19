"""Constante comune pentru tot pipeline-ul."""

# Clasele de leziuni (aceeasi ordine ca in setul de date YOLO din Etapa 2)
CLASSES = [
    "ameloblastom",
    "keratochist odontogen",
    "chist dentiger",
    "chist radicular",
    "alta leziune",
]

# Culori BGR pentru desenare (OpenCV)
CLASS_COLORS = {
    "ameloblastom": (60, 60, 230),
    "keratochist odontogen": (40, 160, 240),
    "chist dentiger": (200, 120, 40),
    "chist radicular": (90, 180, 60),
    "alta leziune": (180, 90, 180),
}

# Validare
MIN_WIDTH = 800
MIN_ASPECT, MAX_ASPECT = 1.6, 2.8  # raport latime/inaltime tipic unei panoramice
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".dcm"}

# Preprocesare / inferenta
TARGET_SIZE = (1280, 640)  # (W, H) dupa letterbox
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

# Calea implicita a modelului antrenat (Etapa 2). Daca lipseste → baseline.
YOLO_WEIGHTS = "models/best.pt"
