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
import io

# --- Reusable modules ---
from narrative.narrative_io import read_csv_auto
from narrative.narrative_embed import load_sbert, concat_title_snippet, embed_texts
from narrative.narrative_cluster import run_kmeans, attach_clusters

# --- Modified normalize_to_canonical to preserve Influencer and Twitter Screen Name ---
def normalize_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw DataFrame to the canonical columns:
    Title, Snippet, (optional) Date, URL, author, display_name.
    Handles Meltwater X columns like Opening Text / Hit Sentence / Parent URL / Alternate Date Format / Time.
    Preserves Influencer as author and Twitter Screen Name as display_name.
    """
    # Map normalized -> original names
    norm2orig: dict[str, str] = {}
    cols_norm: list[str] = []
    for c in df_raw.columns:
        k = re.sub(r"\s+|_", "", c.strip().lower())
        norm2orig[k] = c
        cols_norm.append(k)
    
    # Canonical alias lists
    ALIASES = {
        "title": ["title", "headline", "headlines", "inputname", "keywords"],
        "snippet": ["snippet", "summary", "description", "dek", "selftext", "selftext_html", "body", "text", "openingtext", "hitsentence"],
        "date": ["date", "published", "pubdate", "time", "created", "created_iso", "created_utc", "alternatedateformat"],
        "url": ["url", "link", "permalink", "parenturl"],
        "author": ["author", "influencer"],
        "display_name": ["twitter screen name"]
    }

    # Identify common fields
    title_key = next((k for k in ALIASES["title"] if k in norm2orig), None)
    snippet_key = next((k for k in ALIASES["snippet"] if k in norm2orig), None)
    date_key = next((k for k in ALIASES["date"] if k in norm2orig), None)
    url_key = next((k for k in ALIASES["url"] if k in norm2orig), None)
    author_key = next((k for k in ALIASES["author"] if k in norm2orig), None)
    display_name_key = next((k for k in ALIASES["display_name"] if k in norm2orig), None)

    # --- MELTWATER-SPECIFIC TITLE/SNIPPET LOGIC ---
    headline = norm2orig.get("headline")
    hitsent = norm2orig.get("hitsentence")
    opentxt = norm2orig.get("openingtext")
    inputname = norm2orig.get("inputname")
    keywords = norm2orig.get("keywords")

    # Build Title
    if headline:
        title_series = df_raw[headline].fillna("").astype(str).str.strip()
    elif hitsent:
        title_series = df_raw[hitsent].fillna("").astype(str).str.strip()
    elif opentxt:
        title_series = df_raw[opentxt].fillna("").astype(str).str.strip()
    elif inputname:
        title_series = df_raw[inputname].fillna("").astype(str).str.strip()
    elif keywords:
        title_series = df_raw[keywords].fillna("").astype(str).str.strip()
    else:
        title_series = pd.Series([""] * len(df_raw), dtype=object)
        for k in ALIASES["title"]:
            if k in norm2orig:
                s = df_raw[norm2orig[k]].fillna("").astype(str).str.strip()
                title_series = title_series.mask(~title_series.astype(bool), s)

    # Build Snippet
    if opentxt:
        snippet_series = df_raw[opentxt].fillna("").astype(str).str.strip()
    elif hitsent:
        snippet_series = df_raw[hitsent].fillna("").astype(str).str.strip()
    elif headline:
        snippet_series = df_raw[headline].fillna("").astype(str).str.strip()
    else:
        snippet_series = pd.Series([""] * len(df_raw), dtype=object)
        for k in ALIASES["snippet"]:
            if k in norm2orig:
                s = df_raw[norm2orig[k]].fillna("").astype(str).str.strip()
                snippet_series = snippet_series.mask(~snippet_series.astype(bool), s)

    # Build Date
    def _parse_meltwater_datetime(df: pd.DataFrame, norm2orig: dict) -> pd.Series:
        def _coerce(s: pd.Series) -> pd.Series:
            s = s.astype(str).str.replace(r'(?i)(am|pm)$', r' \1', regex=True)
            return pd.to_datetime(s, errors="coerce", dayfirst=True)
        date_col = norm2orig.get("date")
        alt_col = norm2orig.get("alternatedateformat")
        time_col = norm2orig.get("time")
        if date_col:
            d = _coerce(df[date_col])
        else:
            d = pd.Series(pd.NaT, index=df.index)
        need = d.isna()
        if need.any() and (alt_col or time_col):
            alt = df[norm2orig.get("alternatedateformat", "")].astype(str).str.strip() if alt_col else ""
            tim = df[norm2orig.get("time", "")].astype(str).str.strip() if time_col else ""
            combo = (alt + " " + tim).str.strip()
            d2 = _coerce(combo)
            d = d.fillna(d2)
        return d

    if date_key:
        date_series = _parse_meltwater_datetime(df_raw, norm2orig)
    else:
        date_series = _parse_meltwater_datetime(df_raw, norm2orig)

    # Build URL
    if "parenturl" in norm2orig:
        url_series = df_raw[norm2orig["parenturl"]].fillna("").astype(str).str.strip()
    elif url_key:
        url_series = df_raw[norm2orig[url_key]].fillna("").astype(str).str.strip()
    else:
        url_series = pd.Series([""] * len(df_raw), dtype=object)

    # Build Author (Influencer)
    author_series = df_raw[norm2orig[author_key]].fillna("").astype(str).str.strip() if author_key else pd.Series([""] * len(df_raw), dtype=object)

    # Build Display Name (Twitter Screen Name)
    display_name_series = df_raw[norm2orig[display_name_key]].fillna("").astype(str).str.strip() if display_name_key else pd.Series([""] * len(df_raw), dtype=object)

    # Assemble canonical DataFrame
    df = pd.DataFrame({
        "Title": title_series,
        "Snippet": snippet_series,
        "Date": date_series,
        "URL": url_series,
        "author": author_series,
        "display_name": display_name_series
    })

    # Clean rows: require either Title or Snippet
    df = df[(df["Title"].str.len() > 0) | (df["Snippet"].str.len() > 0)].copy()
    df.drop_duplicates(subset=["Title", "Snippet"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# --- Page setup ---
st.set_page_config(page_title="Narrative Analysis", layout="wide")

# Custom CSS for a polished look with updated sidebar background
st.markdown("""
    <style>
    .main .block-container {
        padding: 2rem;
        background: linear-gradient(to bottom, #f7f9fc, #e3e9f2);
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSidebar {
        background-color: #d1d5db;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stSidebar, .stSidebar h2, .stSidebar label, .stSidebar .stCheckbox label, .stSidebar .stFileUploader label {
        color: #4b5563 !important;
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
    .stSlider [type="range"] {
        -webkit-appearance: none;
        appearance: none;
        height: 8px;
        background: #d3d8e0;
        border-radius: 5px;
    }
    .stSlider [type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none;
        appearance: none;
        width: 16px;
        height: 16px;
        background-color: #1a3c6d;
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stSlider [type="range"]::-moz-range-thumb {
        width: 16px;
        height: 16px;
        background-color: #1a3c6d;
        border-radius: 50%;
        cursor: pointer;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stSlider [type="range"]::-webkit-slider-runnable-track {
        background: #d3d8e0;
        height: 8px;
        border-radius: 5px;
    }
    .stSlider [type="range"]::-moz-range-track {
        background: #d3d8e0;
        height: 8px;
        border-radius: 5px;
    }
    .stSlider [type="range"]::-webkit-slider-runnable-track {
        background: linear-gradient(to right, #4a90e2 0%, #4a90e2 var(--thumb-position), #d3d8e0 var(--thumb-position), #d3d8e0 100%);
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

if st.sidebar.button("Reset data & state"):
    for k in ["df", "embeddings", "data_sig", "clustered", "labels", "baseline", "assigned_from_baseline", "narratives_generated"]:
        st.session_state.pop(k, None)
    st.success("State cleared. Upload to proceed.")
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
            st.write(f"Raw DataFrame columns: {raw.columns.tolist()}")
            df = normalize_to_canonical(raw)
            st.write(f"Normalized DataFrame columns: {df.columns.tolist()}")
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
        st.info("Upload a CSV/TSV to proceed. Required: Title, Snippet, Influencer. Optional: Date, URL, Twitter Screen Name, Likes, Reposts, Replies.")
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
st.header("Run Clusters")
st.markdown("Use the sliding scale to set the number of narrative categories for your data. You can always adjust it to capture the main narratives more exactly.")
k = st.slider("Number of clusters", 2, 12, 6, 1)
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
    for i, cid in enumerate(sorted(narratives.keys()), start=1):
        st.write(f"{i}. **{short_labels_map[cid]}**: {narratives[cid]}")
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

    # Top 10 Posters Analysis
    st.subheader("Top Posters Analysis")
    if 'author' in dfc.columns:
        # Create display label combining display_name and author
        dfc['display_label'] = dfc.apply(
            lambda x: f"{x['display_name']} ({x['author']})" if pd.notnull(x.get('display_name')) and x['display_name'] else x['author'],
            axis=1
        )
        
        # Compute top 10 by volume
        volume_by_author = dfc.groupby(['author', 'display_label']).size().reset_index(name='Volume').nlargest(10, 'Volume')
        # Compute top 10 by engagement
        engagement_cols = [col for col in ['likes', 'reposts', 'replies', 'Likes', 'Reposts', 'Retweets', 'Comments', 'Shares', 'Reactions'] if col in dfc.columns]
        if engagement_cols:
            engagement_by_author = dfc.groupby(['author', 'display_label'])[engagement_cols].sum().sum(axis=1).reset_index(name='Engagement').nlargest(10, 'Engagement')
        else:
            engagement_by_author = pd.DataFrame({'author': [], 'display_label': [], 'Engagement': []})
            st.warning("No engagement columns (likes, reposts, replies, Likes, Reposts, Retweets, Comments, Shares, Reactions) found in dataset. Engagement chart skipped.")
        
        # Display table of top posters
        if not volume_by_author.empty:
            table_data = volume_by_author[['display_label', 'Volume']].merge(
                engagement_by_author[['display_label', 'Engagement']], on='display_label', how='outer'
            ).fillna({'Volume': 0, 'Engagement': 0})
            table_data.columns = ['Poster', 'Volume (Posts)', 'Engagement (Likes + Reposts + Replies)']
            st.write("**Top Posters Table**")
            st.dataframe(table_data, use_container_width=True)
        
        # Create two-column layout for bar charts
        col1, col2 = st.columns(2)
        
        # Volume Bar Chart
        with col1:
            fig_volume_authors = px.bar(
                volume_by_author,
                x='Volume',
                y='display_label',
                title="Top 10 Posters by Volume",
                color='display_label',
                color_discrete_sequence=COLOR_PALETTE,
                orientation='h'
            )
            fig_volume_authors.update_traces(
                marker=dict(line=dict(width=1, color='#ffffff'), opacity=0.9),
                text=volume_by_author['Volume'],
                textposition='auto'
            )
            fig_volume_authors.update_layout(
                font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                title=dict(text="Top 10 Posters by Volume", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
                xaxis=dict(title="Post Count", title_font=dict(size=14), tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)"),
                yaxis=dict(title="Poster", title_font=dict(size=14), tickfont=dict(size=12)),
                plot_bgcolor="rgba(247,249,252,0.8)",
                paper_bgcolor="rgba(255,255,255,0)",
                showlegend=False,
                margin=dict(l=50, r=50, t=80, b=50),
                hovermode="closest",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto")
            )
            if not volume_by_author.empty:
                max_volume_author = volume_by_author.iloc[0]['display_label']
                max_volume_value = volume_by_author.iloc[0]['Volume']
                fig_volume_authors.add_annotation(
                    x=max_volume_value,
                    y=max_volume_author,
                    text=f"Top: {max_volume_value}",
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-30,
                    font=dict(size=12, color="#1a3c6d")
                )
            st.plotly_chart(fig_volume_authors, use_container_width=True)
        
        # Engagement Bar Chart
        with col2:
            if not engagement_by_author.empty:
                fig_engagement_authors = px.bar(
                    engagement_by_author,
                    x='Engagement',
                    y='display_label',
                    title="Top 10 Posters by Engagement",
                    color='display_label',
                    color_discrete_sequence=COLOR_PALETTE,
                    orientation='h'
                )
                fig_engagement_authors.update_traces(
                    marker=dict(line=dict(width=1, color='#ffffff'), opacity=0.9),
                    text=engagement_by_author['Engagement'].round(0).astype(int),
                    textposition='auto'
                )
                fig_engagement_authors.update_layout(
                    font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                    title=dict(text="Top 10 Posters by Engagement", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
                    xaxis=dict(title="Engagement (Likes + Reposts + Replies)", title_font=dict(size=14), tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)"),
                    yaxis=dict(title="Poster", title_font=dict(size=14), tickfont=dict(size=12)),
                    plot_bgcolor="rgba(247,249,252,0.8)",
                    paper_bgcolor="rgba(255,255,255,0)",
                    showlegend=False,
                    margin=dict(l=50, r=50, t=80, b=50),
                    hovermode="closest",
                    hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto")
                )
                max_engagement_author = engagement_by_author.iloc[0]['display_label']
                max_engagement_value = engagement_by_author.iloc[0]['Engagement']
                fig_engagement_authors.add_annotation(
                    x=max_engagement_value,
                    y=max_engagement_author,
                    text=f"Top: {int(max_engagement_value)}",
                    showarrow=True,
                    arrowhead=2,
                    ax=20,
                    ay=-30,
                    font=dict(size=12, color="#1a3c6d")
                )
                st.plotly_chart(fig_engagement_authors, use_container_width=True)

    # Author-Theme Correlation
    st.subheader("Author-Theme Correlation")
    if 'author' in dfc.columns:
        # Compute correlation (post counts by author and cluster)
        correlation_data = dfc.groupby(['author', 'display_label', 'Cluster']).size().reset_index(name='Volume').pivot_table(
            index=['author', 'display_label'], columns='Cluster', values='Volume', fill_value=0
        )
        # Select top 10 authors by volume
        top_authors = volume_by_author[['author', 'display_label']].set_index('author')['display_label']
        correlation_data = correlation_data.loc[correlation_data.index.get_level_values('author').isin(top_authors.index)]
        if not correlation_data.empty:
            # Map cluster IDs to short labels
            correlation_data.columns = [short_labels_map.get(col, f"Cluster {col}") for col in correlation_data.columns]
            # Use display_label for visualization
            correlation_data = correlation_data.reset_index().set_index('display_label').drop(columns='author')
            # Create heatmap
            fig_correlation = px.imshow(
                correlation_data,
                labels=dict(x="Narrative", y="Poster", color="Post Count"),
                title="Poster-Narrative Correlation (Post Counts)",
                color_continuous_scale="RdBu",
                aspect="auto"
            )
            fig_correlation.update_layout(
                font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                title=dict(text="Poster-Narrative Correlation (Post Counts)", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
                xaxis=dict(title="Narrative", title_font=dict(size=14), tickfont=dict(size=12), tickangle=45),
                yaxis=dict(title="Poster", title_font=dict(size=14), tickfont=dict(size=12)),
                plot_bgcolor="rgba(247,249,252,0.8)",
                paper_bgcolor="rgba(255,255,255,0)",
                margin=dict(l=50, r=50, t=80, b=50),
                hovermode="closest",
                hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto")
            )
            # Add annotation for highest correlation
            max_value = correlation_data.max().max()
            max_loc = np.where(correlation_data == max_value)
            max_author = correlation_data.index[max_loc[0][0]]
            max_narrative = correlation_data.columns[max_loc[1][0]]
            fig_correlation.add_annotation(
                x=max_narrative,
                y=max_author,
                text=f"Peak: {int(max_value)}",
                showarrow=True,
                arrowhead=2,
                ax=20,
                ay=-30,
                font=dict(size=12, color="#1a3c6d")
            )
            st.plotly_chart(fig_correlation, use_container_width=True)
        else:
            st.warning("No data available for poster-theme correlation. Ensure posters and clusters are present.")
    else:
        st.warning("No 'author' or 'Influencer' column found in dataset. Poster-theme correlation skipped. Ensure your CSV includes an 'Influencer' column.")

else:
    if "df" in st.session_state and not "clustered" in st.session_state:
        st.info("Run clustering to generate narratives.")
    else:
        st.info("Upload a CSV/TSV to proceed. Required: Title, Snippet, Influencer. Optional: Date, URL, Twitter Screen Name, Likes, Reposts, Replies.")
    st.stop()
