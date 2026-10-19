"""AI pentru zambete — interfata web (Streamlit).

Rulare:  streamlit run app.py
"""
from __future__ import annotations

import io
import json

import cv2
import pandas as pd
import streamlit as st
from PIL import Image

from core import analyze
from core.config import CONF_THRESHOLD
from core.detector import get_detector
from core import storage
from core.visualize import draw

st.set_page_config(page_title="AI pentru zambete", page_icon="🦷", layout="wide")


@st.cache_resource
def detector():
    return get_detector()


def png_bytes(rgb) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


# ---------- bara laterala ----------
st.sidebar.title("AI pentru zambete")
st.sidebar.caption("Identificarea tumorilor osoase maxilofaciale pe radiografii panoramice")
page = st.sidebar.radio("Pagina", ["Analiza noua", "Istoric"])
st.sidebar.markdown(f"**Model activ:** {detector().name}")
st.sidebar.info(
    "Instrument de suport la decizie. Rezultatele nu inlocuiesc diagnosticul "
    "medicului si confirmarea histopatologica."
)

# ---------- pagina: istoric ----------
if page == "Istoric":
    st.header("Istoricul analizelor")
    rows = storage.history()
    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.write("Nicio analiza salvata inca.")
    st.stop()

# ---------- pagina: analiza noua ----------
st.header("Analiza noua")

c1, c2 = st.columns([2, 1])
upload = c1.file_uploader(
    "1. Incarcati radiografia panoramica (OPG)", type=["png", "jpg", "jpeg", "dcm"]
)
patient = c2.text_input("Cod pacient (anonimizat)", placeholder="ex. P-0042")

if upload is None:
    st.info("Incarcati o radiografie pentru a incepe. Pentru test: `python make_sample.py`.")
    st.stop()

key = f"{upload.name}-{upload.size}-{patient}"
if st.session_state.get("key") != key:
    with st.spinner("2–5. Validare, preprocesare, detectie, localizare…"):
        try:
            st.session_state.result = analyze(upload.getvalue(), upload.name, patient, detector())
        except ValueError as e:
            st.error(str(e))
            st.stop()
    st.session_state.key = key

res = st.session_state.result

# Pasul 2: validare
for w in res.validation.warnings:
    st.warning(w)
if not res.validation.ok:
    st.error("Imaginea nu a trecut validarea:")
    for e in res.validation.errors:
        st.write(f"- {e}")
    st.stop()

# Pasul 6: vizualizare
st.subheader("Rezultate")
thr = st.slider("Prag de incredere", 0.0, 1.0, CONF_THRESHOLD, 0.05)
shown = [f for f in res.findings if f["score"] >= thr]

m1, m2, m3 = st.columns(3)
m1.metric("Leziuni suspecte", len(shown))
m2.metric("Timp de analiza", f"{res.duration_s:.2f} s")
m3.metric("Rezolutie", f"{res.image_size[0]}×{res.image_size[1]}")

base_img = res.enhanced if st.toggle("Afiseaza imaginea preprocesata (CLAHE)", value=True) else res.original
annotated = draw(base_img, shown)
st.image(annotated, use_container_width=True)

# Pasul 7: validarea medicului
st.subheader("Validare medic")
if not shown:
    st.success("Nicio leziune suspecta peste pragul ales.")
for f in shown:
    with st.expander(f"#{f['id']} · {f['cls']} · {f['score']:.0%} · {f['region']}", expanded=True):
        a, b = st.columns([1, 2])
        f["status"] = a.radio(
            "Decizie",
            ["nevalidat", "confirmat", "respins"],
            index=["nevalidat", "confirmat", "respins"].index(f["status"]),
            key=f"st-{res.analysis_id}-{f['id']}",
            horizontal=True,
        )
        f["note"] = b.text_input("Nota", f["note"], key=f"nt-{res.analysis_id}-{f['id']}")

# Pasul 8: export si istoric
st.subheader("Export")
report = res.to_report()
report["findings"] = shown
e1, e2, e3 = st.columns(3)
e1.download_button(
    "Raport JSON",
    json.dumps(report, ensure_ascii=False, indent=2),
    file_name=f"raport_{res.analysis_id}.json",
    mime="application/json",
)
e2.download_button(
    "Imagine adnotata (PNG)",
    png_bytes(annotated),
    file_name=f"adnotat_{res.analysis_id}.png",
    mime="image/png",
)
if e3.button("Salveaza in istoric", type="primary"):
    storage.save(report)
    st.toast("Analiza a fost salvata.")
