# app.py — Step A: CSV uploader + canonical normalization using your src/ module
import streamlit as st
import pandas as pd
from src.narrative_io import normalize_to_canonical  # ← reuse your module

st.set_page_config(page_title="Narrative Analysis", layout="wide")
st.title("Narrative Analysis — Data Loader")

st.sidebar.header("Load Data")
uploaded = st.sidebar.file_uploader("Upload CSV (UTF-8)", type=["csv"])

use_demo = st.sidebar.checkbox("Use tiny demo data", value=False)

df = None
error = None

try:
    if uploaded:
        raw = pd.read_csv(uploaded)
        df = normalize_to_canonical(raw)
    elif use_demo:
        demo = pd.DataFrame({
            "headline": [
                "Company X faces investor lawsuit after earnings miss",
                "CEO of Company Y announces leadership changes",
                "Analysts debate revenues ahead of Q3 guidance"
            ],
            "summary": [
                "Plaintiffs allege misleading statements in prior quarter.",
                "Restructuring aims to streamline operations.",
                "Market expects mixed results amid broader sector headwinds."
            ],
            "published": ["2025-09-01", "2025-09-05", "2025-09-10"],
            "link": [
                "https://example.com/x-lawsuit",
                "https://example.com/y-ceo",
                "https://example.com/analyst-q3"
            ]
        })
        df = normalize_to_canonical(demo)
except Exception as e:
    error = str(e)

if error:
    st.error(error)

if df is not None and not df.empty:
    st.success(f"Loaded {len(df)} rows.")
    st.dataframe(df.head(20), use_container_width=True)
    # keep for later steps (embeddings, clustering, etc.)
    st.session_state["df"] = df
else:
    st.info(
        "Upload a CSV or enable the demo to proceed.\n\n"
        "Required columns (any alias): Title/Headline and Snippet/Summary/Description.\n"
        "Optional: Date, URL."
    )
