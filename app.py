# app.py — Data upload → embeddings → KMeans clustering → top terms
import streamlit as st
import pandas as pd
import plotly.express as px

# Reuse your GitHub modules
from src.narrative_io import normalize_to_canonical
from src.narrative_embed import load_sbert, concat_title_snippet, embed_texts
from src.narrative_cluster import run_kmeans, attach_clusters, cluster_counts
from src.narrative_terms import ensure_text_column, cluster_terms_dataframe

st.set_page_config(page_title="Narrative Analysis", layout="wide")
st.title("Narrative Analysis")

# ---------------------------
# Section 1 — Load & normalize
# ---------------------------
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
    st.session_state["df"] = df
else:
    st.info(
        "Upload a CSV or enable the demo to proceed.\n\n"
        "Required columns (any alias): Title/Headline and Snippet/Summary/Description.\n"
        "Optional: Date, URL."
    )

# ---------------------------------------
# Section 2 — Embeddings + KMeans control
# ---------------------------------------
st.sidebar.header("Clustering")
k = st.sidebar.slider("Number of clusters (KMeans)", 2, 12, 6, 1)
run_btn = st.sidebar.button("Run clustering")

# Cache the embedding model and embeddings to speed things up
@st.cache_resource
def get_model():
    return load_sbert("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def embed_df_texts(df_in: pd.DataFrame):
    model = get_model()
    texts = concat_title_snippet(df_in)
    emb = embed_texts(model, texts, show_progress=False)
    return emb

# If user clicks "Run clustering"
if run_btn and "df" in st.session_state and isinstance(st.session_state["df"], pd.DataFrame):
    df_in = st.session_state["df"]
    with st.spinner("Embedding and clustering..."):
        embeddings = embed_df_texts(df_in)
        labels, _ = run_kmeans(embeddings, n_clusters=k)
        df_clustered = attach_clusters(df_in, labels)
        st.session_state["df"] = df_clustered
        st.session_state["embeddings"] = embeddings

    st.success("Clustering complete.")

# If we have clustered data in session, display results
if "df" in st.session_state and isinstance(st.session_state["df"], pd.DataFrame) and "Cluster" in st.session_state["df"].columns:
    df_clustered = st.session_state["df"]

    st.subheader("Cluster sizes")
    counts = cluster_counts(df_clustered)
    st.dataframe(counts, use_container_width=True)
    fig = px.bar(counts, x="Cluster", y="Count", title="Posts per Cluster")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top terms per cluster (TF-IDF)")
    df_norm = ensure_text_column(df_clustered)  # builds _text_norm
    terms_df = cluster_terms_dataframe(df_norm, n_terms=12)
    st.dataframe(terms_df, use_container_width=True)
