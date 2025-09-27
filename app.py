import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import datetime as _dt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from collections import Counter
import openai
import os

# --- Reusable modules ---
from narrative.narrative_io import read_csv_auto, normalize_to_canonical
from narrative.narrative_embed import load_sbert, concat_title_snippet, embed_texts
from narrative.narrative_cluster import run_kmeans, attach_clusters

# --- Page setup ---
st.set_page_config(page_title="Narrative Analysis", layout="wide")

# Custom CSS for a polished look
st.markdown("""
    <style>
    .main .block-container {
        padding: 2rem;
        background-color: #f7f9fc;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    h1, h2, h3 {
        font-family: 'Roboto', sans-serif;
        color: #1a3c6d;
    }
    .stButton>button {
        background-color: #1a3c6d;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #2e5aa8;
    }
    </style>
""", unsafe_allow_html=True)

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

def central_indexes(emb, mask_idx, k=5):
    sub = emb[mask_idx]
    if len(sub) == 0:
        return []
    centroid = sub.mean(axis=0, keepdims=True)
    sims = (centroid @ sub.T).ravel()
    order = sims.argsort()[::-1][:min(k, len(sims))]
    return [mask_idx[i] for i in order]

def llm_narrative_summary(texts: list[str], cid) -> tuple[str, str, str]:
    combined_text = "\n\n".join(texts)
    prompt = (
        "Create a fresh, original summary of the main gist of these social media posts clustered around a theme. "
        "Do not copy-paste, quote, or repeat any specific tweet content literally. "
        "Identify key actors, events, conflicts, resolutions, and themes in a concise 1-2 sentence summary (under 50 words). "
        "Suggest a detailed, meaningful label for this narrative (10-20 words). "
        "Suggest a short 2-4 word label derived from the summary. "
        "Output format: Summary: [your fresh summary] Detailed Label: [your detailed label] Short Label: [your 2-4 word label]"
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
        summary = output.split("Detailed Label:")[0].replace("Summary: ", "").strip()
        detailed_label = output.split("Detailed Label:")[1].split("Short Label:")[0].strip()
        short_label = output.split("Short Label:")[1].strip()
        return summary, detailed_label, short_label
    except Exception as e:
        return f"Error: {e}", f"Narrative {cid}", f"Cluster {cid}"

# --- Section 1: Load & Normalize ---
st.sidebar.header("Load Data")
uploaded = st.sidebar.file_uploader("Upload CSV (UTF-8 / UTF-16 / TSV)", type=["csv", "tsv"])
use_demo = st.sidebar.checkbox("Use tiny demo data", value=False)

if st.sidebar.button("Reset data & state"):
    for k in ["df", "embeddings", "data_sig", "clustered", "labels", "baseline", "assigned_from_baseline", "narratives_generated"]:
        st.session_state.pop(k, None)
    st.success("State cleared. Upload or enable demo again.")
    st.stop()

def _df_signature(d: pd.DataFrame):
    try:
        sig = (d.shape, pd.util.hash_pandas_object(d, index=True).sum())
    except Exception:
        sig = (d.shape, tuple(sorted(d.columns)))
    return sig

df = None
error = None
try:
    if uploaded is not None:
        raw = read_csv_auto(uploaded)
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
        for k in ["embeddings", "clustered", "assigned_from_baseline", "labels", "narratives_generated"]:
            st.session_state.pop(k, None)
        st.write("Dataset Uploaded Successfully! 🎉")
    else:
        st.write("Dataset Uploaded Successfully! 🎉")
else:
    if "df" in st.session_state:
        st.info("Run clustering to generate narratives.")
    else:
        st.info("Upload a CSV/TSV or enable demo to proceed. Required: Title, Snippet. Optional: Date, URL.")
    st.stop()

# --- Embeddings ---
@st.cache_resource
def get_model():
    return load_sbert("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def embed_df_texts(df_in: pd.DataFrame):
    model = get_model()
    texts = concat_title_snippet(df_in)
    emb = embed_texts(model, texts, show_progress=False)
    return emb

# --- Clustering with Narrative Generation ---
st.header("Clustering")
k = st.slider("Number of clusters (KMeans)", 2, 12, 6, 1)
if st.button("Run clustering"):
    with st.spinner("Hold on, we are generating narratives for you!"):
        embeddings = embed_df_texts(st.session_state["df"])
        labels, _ = run_kmeans(embeddings, n_clusters=k)
        df_clustered = attach_clusters(st.session_state["df"], labels)
        st.session_state["df"] = df_clustered
        st.session_state["embeddings"] = embeddings
        st.session_state["clustered"] = True
        # Generate narratives
        labels_map = {}
        short_labels_map = {}
        narratives = {}
        for cid in sorted(df_clustered["Cluster"].unique()):
            mask_idx = np.where(df_clustered["Cluster"].values == cid)[0]
            if len(mask_idx) == 0:
                continue
            top_idx = central_indexes(embeddings, mask_idx, k=5)
            central_texts = [" ".join([str(df_clustered.iloc[i].get("Title", "") or ""), first_sentence(df_clustered.iloc[i].get("Snippet", ""))]).strip() for i in top_idx]
            summary, detailed_label, short_label = llm_narrative_summary(central_texts, cid)
            labels_map[cid] = detailed_label
            short_labels_map[cid] = short_label
            narratives[cid] = summary
        st.session_state["labels_map"] = labels_map
        st.session_state["short_labels_map"] = short_labels_map
        st.session_state["narratives"] = narratives
        st.session_state["narratives_generated"] = True
    st.success("Clustering and narrative generation complete.")

# --- Custom Color Palette ---
COLOR_PALETTE = [
    '#1a3c6d',  # Deep Blue
    '#d32f2f',  # Red
    '#2e7d32',  # Green
    '#f57c00',  # Orange
    '#6a1b9a',  # Purple
    '#0288d1',  # Light Blue
    '#c2185b',  # Pink
    '#388e3c',  # Forest Green
    '#f4a261',  # Peach
    '#00838f',  # Cyan
    '#8e24aa',  # Violet
    '#689f38'   # Lime Green
]

# --- Main Display ---
if "df" in st.session_state and "Cluster" in st.session_state["df"].columns and "narratives_generated" in st.session_state:
    dfc = st.session_state["df"].copy()
    emb = st.session_state.get("embeddings")
    if emb is None or len(emb) != len(dfc):
        emb = embed_df_texts(dfc)
        st.session_state["embeddings"] = emb
    labels_map = st.session_state["labels_map"]
    short_labels_map = st.session_state["short_labels_map"]
    narratives = st.session_state["narratives"]
    # Display Narratives with short headers
    st.subheader("Narratives")
    for cid, narrative in narratives.items():
        st.write(f"**{short_labels_map[cid]}**: {narrative}")
    # Bar Chart of Narrative Volumes with horizontal short labels
    volume_data = dfc["Cluster"].value_counts().reset_index()
    volume_data.columns = ["Cluster", "Volume"]
    volume_data["Narrative"] = volume_data["Cluster"].map(short_labels_map)
    st.subheader("Narrative Volumes")
    fig_volumes = px.bar(
        volume_data,
        x="Narrative",
        y="Volume",
        title="Narrative Volumes",
        color="Narrative",
        color_discrete_sequence=COLOR_PALETTE
    )
    # Enhance bar chart
    fig_volumes.update_traces(
        marker=dict(
            line=dict(width=1, color='#ffffff'),
            opacity=0.9
        ),
        text=volume_data["Volume"],
        textposition='auto'
    )
    fig_volumes.update_layout(
        font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
        title=dict(text="Narrative Volumes", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
        xaxis=dict(
            title="Narrative",
            tickangle=0,
            title_font=dict(size=14),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Volume",
            title_font=dict(size=14),
            tickfont=dict(size=12),
            gridcolor="rgba(0,0,0,0.1)"
        ),
        plot_bgcolor="rgba(247,249,252,0.8)",
        paper_bgcolor="rgba(255,255,255,0)",
        showlegend=True,
        margin=dict(l=50, r=50, t=80, b=50),
        hovermode="closest",
        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto")
    )
    # Add annotation for highest volume
    max_volume = volume_data["Volume"].max()
    max_narrative = volume_data[volume_data["Volume"] == max_volume]["Narrative"].iloc[0]
    fig_volumes.add_annotation(
        x=max_narrative,
        y=max_volume,
        text=f"Peak: {max_volume}",
        showarrow=True,
        arrowhead=2,
        ax=20,
        ay=-30,
        font=dict(size=12, color="#1a3c6d")
    )
    st.plotly_chart(fig_volumes, use_container_width=True)
    # Timeline of Narratives (Volume Trend)
    date_column = next((col for col in ["Date", "published"] if col in dfc.columns), None)
    if date_column and dfc[date_column].notna().any():
        dfc[date_column] = pd.to_datetime(dfc[date_column], errors="coerce")
        min_date = dfc[date_column].min()
        max_date = dfc[date_column].max()
        span_days = (max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else 0
        if span_days > 0:
            # Adaptive frequency based on time span
            if span_days <= 3:
                freq = 'D'
            elif span_days < 7:
                freq = 'D'
            elif span_days < 30:
                freq = '3D'
            elif span_days < 90:
                freq = 'W'
            else:
                freq = 'M'
            if freq == 'D':
                x_title = 'Day'
            elif freq == '3D':
                x_title = '3-Day Period'
            elif freq == 'W':
                x_title = 'Week'
            elif freq == 'M':
                x_title = 'Month'
            else:
                x_title = 'Period'
            try:
                timeline_data = dfc.groupby(["Cluster", pd.Grouper(key=date_column, freq=freq)]).size().reset_index(name="Volume")
                timeline_data["Narrative"] = timeline_data["Cluster"].map(short_labels_map)
                if timeline_data.empty or timeline_data["Volume"].sum() == 0:
                    st.warning("No data available for the timeline. Ensure there are posts across multiple time periods.")
                else:
                    st.subheader("Trends Over Time")
                    fig_timeline = px.line(
                        timeline_data,
                        x=date_column,
                        y="Volume",
                        color="Narrative",
                        title="Trends Over Time",
                        labels={"Volume": "Post Count", date_column: x_title},
                        markers=True,
                        line_shape='spline',
                        color_discrete_sequence=COLOR_PALETTE
                    )
                    # Enhance line chart
                    fig_timeline.update_traces(
                        line=dict(width=3),
                        marker=dict(size=8, line=dict(width=1, color='#ffffff')),
                        mode='lines+markers',
                        fill='tozeroy',
                        fillcolor='rgba(0,0,0,0.05)'
                    )
                    fig_timeline.update_layout(
                        font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                        title=dict(text="Trends Over Time", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
                        xaxis=dict(
                            title=x_title,
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            gridcolor="rgba(0,0,0,0.1)"
                        ),
                        yaxis=dict(
                            title="Number of Posts",
                            title_font=dict(size=14),
                            tickfont=dict(size=12),
                            gridcolor="rgba(0,0,0,0.1)",
                            tickformat="d"
                        ),
                        plot_bgcolor="rgba(247,249,252,0.8)",
                        paper_bgcolor="rgba(255,255,255,0)",
                        showlegend=True,
                        margin=dict(l=50, r=50, t=80, b=50),
                        hovermode="x unified",
                        hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto")
                    )
                    # Add annotation for highest volume point
                    max_volume_row = timeline_data.loc[timeline_data["Volume"].idxmax()]
                    fig_timeline.add_annotation(
                        x=max_volume_row[date_column],
                        y=max_volume_row["Volume"],
                        text=f"Peak: {max_volume_row['Volume']}",
                        showarrow=True,
                        arrowhead=2,
                        ax=20,
                        ay=-30,
                        font=dict(size=12, color="#1a3c6d")
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
            except KeyError:
                st.warning(f"No valid {date_column} data for timeline. Ensure dates are properly formatted.")
        else:
            st.warning("Insufficient time span in data for trends. All dates are the same or invalid.")
    else:
        st.warning("No 'Date' or 'published' column found or no valid dates. Add it to your dataset for the timeline.")
else:
    if "df" in st.session_state and not "clustered" in st.session_state:
        st.info("Run clustering to generate narratives.")
    else:
        st.info("Upload a dataset to proceed.")
