# AI pentru zambete: Identificarea tumorilor osoase maxilofaciale

Proiect MIRPR 2026–2027 · Lesan Dominique-Arcadii, Magdas Vlad, Kepes Kevin

Aplicatie web care primeste o radiografie panoramica (OPG) si marcheaza leziunile
suspecte (chisturi si tumori ale maxilarelor), cu clasa, scor de incredere si
regiune anatomica. Este un instrument de suport la decizie, nu de diagnostic.

## Etapa 1 — flow-ul de baza

```
incarcare → validare → preprocesare → detectie → localizare → vizualizare → validare medic → export / istoric
```

| Pas | Modul | Ce face |
| --- | --- | --- |
| Incarcare | `core/io_utils.py` | PNG / JPG / DICOM → imagine grayscale |
| Validare | `core/io_utils.py` | rezolutie ≥ 800 px, raport latime/inaltime 1,6–2,8, contrast |
| Preprocesare | `core/preprocess.py` | CLAHE, filtru bilateral, letterbox 1280×640, normalizare [0,1] |
| Detectie | `core/detector.py` | baseline clasic acum; YOLO automat daca exista `models/best.pt` |
| Localizare | `core/anatomy.py` | maxilar/mandibula, dreapta/stanga pacientului, anterior/posterior |
| Vizualizare | `core/visualize.py` | dreptunghiuri colorate pe clasa + scor |
| Orchestrare | `core/pipeline.py` | leaga pasii, intoarce `AnalysisResult` + raport JSON |
| Istoric | `core/storage.py` | SQLite local (`data/analize.db`) |
| Interfata | `app.py` | Streamlit: analiza noua + istoric |

**Detectorul baseline** (fara invatare) cauta „gauri” intunecate si rotunjite
complet inconjurate de os — aspectul tipic al unei leziuni radiotransparente.
Nu distinge tipul leziunii (intoarce „alta leziune”); rolul lui este sa faca
flow-ul functional cap-coada si sa fie termenul de comparatie pentru modelul din
Etapa 2. Pentru a trece la YOLO nu se modifica aplicatia: se pune modelul in
`models/best.pt` si se instaleaza `ultralytics`.

## Rulare

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python make_sample.py          # creeaza samples/panoramica_sintetica.png (date sintetice)
streamlit run app.py
```

Teste: `pytest -q`

## Etapa 2 (urmatoarea)

- set de date adnotat (CVAT / Label Studio, export YOLO) pe clasele din `core/config.py`;
- antrenare YOLOv8/YOLO11 cu transfer learning si augmentare (Albumentations);
- comparatie cu baseline-ul: precizie, recall, F1, mAP@0.5.
