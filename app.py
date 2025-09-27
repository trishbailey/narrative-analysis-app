# app.py — Narrative Analysis (lean, narrative-focused)
# Features:
# - Upload & normalize CSV (Title/Snippet/Date/URL) with robust encoding handling
# - Embeddings (SBERT, cached)
# - KMeans clustering (one click)
# - Generate fresh narrative summaries and detailed labels per cluster
# - Narrative volumes (bar chart)
# - Sentiment by narrative (colored bars)
# - Timeline of narratives over time (sentiment trend)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import datetime as _dt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter
import openai
import os

# --- Reusable modules ---
from narrative.narrative_io import normalize_to_canonical
from narrative.narrative_embed import load_sbert, concat_title_snippet, embed_texts
from narrative.narrative_cluster import run_kmeans, attach_clusters
from narrative.narrative_sentiment import add_vader_sentiment

# --- Page setup ---
st.set_page_config(page_title="Narrative Analysis", layout="wide")
st.title("Narrative Analysis")

# --- API Setup ---
api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY")
if not api_key:
    st.error("XAI_API_KEY not found. Please add it in Streamlit Secrets.")
else:
    client = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

# --- Helpers ---
JUNK = {
    "rt", "please", "join", "join me", "link", "watch", "live", "breaking", "thanks", "thank", "proud",
    "honored", "glad", "great", "amazing", "incredible", "tonight", "yesterday", "today", "tomorrow",
    "hard work", "grateful", "pray", "praying", "deeply", "heartbroken", "sad", "rip", "thread",
    "foxnews", "cnn", "msnbc", "youtube", "tiktok", "instagram", "facebook", "x", "twitter"
}
STOP = set(ENGLISH_STOP_WORDS) | {w for p in JUNK for w in p.split()}

def first_sentence(text: str, max_len=180) -> str:
    text = str(text or "").strip()
    parts = re.split(r"(?<=[\.!?])\s+", text)
    for s in parts:
        s = s.strip()
        if len(s) >= 25:
            return s[:max_len]
    return (text[:max_len] or "").strip()

def central_indexes(mask_idx, k=5):
    sub = emb[mask_idx]
    if len(sub) == 0:
        return []
    centroid = sub.mean(axis=0, keepdims=True)
    sims = (centroid @ sub.T).ravel()
    order = sims.argsort()[::-1][:min(k, len(sims))]
    return [mask_idx[i] for i in order]

def llm_narrative_summary(texts: list[str]) -> tuple[str, str]:
    combined_text = "\n\n".join(texts)
    prompt = (
        "Create a fresh, original summary of the main gist of these social media posts clustered around a theme. "
        "Do not copy-paste, quote, or repeat any specific tweet content literally. "
        "Identify key actors, events, conflicts, resolutions, and themes in a concise 1-2 sentence summary (under 50 words). "
        "Suggest a detailed, meaningful label for this narrative (10-20 words). "
        "Output format: Summary: [your fresh summary] Label: [your detailed label]"
        f"Posts:\n{combined_text}"
    )
    try:
        response = client.chat.completions.create(
            model="grok-4-fast-reasoning",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.7
        )
        output = response.choices[0].message.content.strip()
        summary = output.split("Label:")[0].replace("Summary: ", "").strip()
        label = output.split("Label:")[-1].strip()
        return summary, label
    except Exception as e:
        return f"Error: {e}", f"Narrative {cid}"

# --- Section 1: Load & Normalize with Robust Encoding ---
st.sidebar.header("Load Data")
uploaded = st.sidebar.file_uploader("Upload CSV (UTF-8 / UTF-16 / TSV)", type=["csv", "tsv"])
use_demo = st.sidebar.checkbox("Use tiny demo data", value=False)

if st.sidebar.button("Reset data & state"):
    for k in ["df", "embeddings", "data_sig", "clustered", "labels", "baseline", "assigned_from_baseline"]:
        st.session_state.pop(k, None)
    st.success("State cleared. Upload or enable demo again.")
    st.stop()

def _df_signature(d: pd.DataFrame):
    try:
        sig = (d.shape, pd.util.hash_pandas_object(d, index=True).sum())
    except Exception:
        sig = (d.shape, tuple(sorted(d.columns)))
    return sig

def read_file_with_encoding(file):
    encodings = ["utf-8-sig", "utf-16", "utf-16le", "utf-8"]
    for enc in encodings:
        try:
            return pd.read_csv(file, encoding=enc, on_bad_lines="skip")
        except UnicodeDecodeError:
            continue
    st.error("Unable to decode file with supported encodings. Please check the file format.")
    return None

df = None
error = None
try:
    if uploaded is not None:
        raw = read_file_with_encoding(uploaded)
        if raw is not None:
            df = normalize_to_canonical(raw)
    elif use_demo:
        demo = pd.DataFrame({
            "headline": ["Company X faces investor lawsuit", "CEO of Company Y announces changes", "Analysts debate revenues"],
            "summary": ["Plaintiffs allege misleading statements.", "Restructuring aims to streamline.", "Market expects mixed results."],
            "published": ["2025-09-01", "2025-09-05", "2025-09-10"],
            "link": ["https://example.com/x-lawsuit", "https://example.com/y-ceo", "https://example.com/analyst-q3"]
        })
        df = normalize_to_canonical(demo)
except Exception as e:
    error = str(e)
if error:
    st.error(error)
if df is not None and not df.empty:
    new_sig = _df_signature(df)
    if ("data_sig" not in st.session_state) or (st.session_state["data_sig"] != new_sig):
        st.session_state["df"] = df.reset_index(drop=True)
        st.session_state["data_sig"] = new_sig
        for k in ["embeddings", "clustered", "assigned_from_baseline", "labels"]:
            st.session_state.pop(k, None)
        st.success(f"Loaded {len(df)} rows (new dataset).")
    else:
        st.success(f"Loaded {len(st.session_state['df'])} rows (using cached state).")
else:
    if
