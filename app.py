import logging
import os
import io
import re
import time
import datetime as _dt
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import streamlit as st

# --- Streamlit page setup MUST be the first Streamlit call ---
st.set_page_config(page_title="Narrative Analysis", layout="wide")

# Logging setup
logging.basicConfig(level=logging.INFO, filename='app.log')
logger = logging.getLogger(__name__)
logger.info("App starting")

# --- LLM / API client (xAI via OpenAI SDK) ---
# Use the modern OpenAI client class and catch APIStatusError for 429/5xx
try:
    from openai import OpenAI, APIStatusError
except Exception as e:  # pragma: no cover
    # If the package isn't available, surface an actionable error
    st.error("The 'openai' package is missing. Add 'openai' to requirements.txt.")
    raise

# --- Third-party ML imports ---
from sklearn.metrics.pairwise import cosine_similarity  # noqa: F401 (kept for future use)
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS  # noqa: F401

# --- App-specific modules (ensure these are vendored or installable) ---
try:
    from narrative.narrative_io import read_csv_auto
    from narrative.narrative_embed import load_sbert, concat_title_snippet, embed_texts
    from narrative.narrative_cluster import run_kmeans, attach_clusters
    logger.info("Custom modules imported")
except Exception as e:
    logger.error(f"Error importing custom modules: {e}")
    st.error(f"Error importing custom modules: {e}. Ensure the 'narrative' package is present in the repo or installed via requirements.txt.")
    raise

# --- Custom CSS ---
st.markdown(
    """
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
    .stSidebar, .stSidebar h2, .stSidebar label, .stSidebar .stCheckbox label, .stSidebar .stFileUploader label { color: #4b5563 !important; }
    h1, h2, h3 { font-family: 'Roboto', sans-serif; color: #1a3c6d; }
    .stButton>button { background-color: #1a3c6d; color: white; border-radius: 8px; padding: 0.5rem 1rem; }
    .stButton>button:hover { background-color: #2e5aa8; }
    .stSlider [type="range"] { -webkit-appearance: none; appearance: none; height: 8px; background: #d3d8e0; border-radius: 5px; }
    .stSlider [type="range"]::-webkit-slider-thumb, .stSlider [type="range"]::-moz-range-thumb {
        -webkit-appearance: none; appearance: none; width: 16px; height: 16px; background-color: #1a3c6d; border-radius: 50%; cursor: pointer; box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }
    .stSlider [type="range"]::-webkit-slider-runnable-track, .stSlider [type="range"]::-moz-range-track { background: #d3d8e0; height: 8px; border-radius: 5px; }
    .stTable { background-color: rgba(247,249,252,0.8); }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Narrative Analysis")
logger.info("Title set")

# --- API Setup ---
try:
    api_key = os.getenv("XAI_API_KEY") or st.secrets.get("XAI_API_KEY")
    logger.info("API key retrieved")
    if not api_key:
        st.error("XAI_API_KEY not found. Add it in Streamlit Secrets (App → Settings → Secrets).")
        logger.error("XAI_API_KEY not found")
        st.stop()
    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
    logger.info("OpenAI (xAI) client initialized")
except Exception as e:
    logger.error(f"Error in API setup: {e}")
    st.error(f"Error in API setup: {e}")
    raise

# --- Helpers / Normalization ---
JUNK = {
    "rt", "please", "join", "join me", "link", "watch", "live", "breaking", "thanks", "thank", "proud",
    "honored", "glad", "great", "amazing", "incredible", "tonight", "yesterday", "today", "tomorrow",
    "hard work", "grateful", "pray", "praying", "deeply", "heartbroken", "sad", "rip", "thread",
    "foxnews", "cnn", "msnbc", "youtube", "tiktok", "instagram", "facebook", "x", "twitter"
}
STOP = set(ENGLISH_STOP_WORDS) | {w for p in JUNK for w in p.split()}
logger.info("Helper variables defined")


def _norm_key(s: str) -> str:
    return re.sub(r"\s+|_", "", (s or "").strip().lower())


def normalize_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize diverse social data schemas into a canonical frame."""
    logger.info("Normalizing DataFrame")
    try:
        # Map normalized->original
        norm2orig: Dict[str, str] = {}
        for c in df_raw.columns:
            norm2orig[_norm_key(c)] = c

        # Alias table expressed in human-friendly form; normalize for matching
        ALIASES = {
            "title": ["title", "headline", "headlines", "inputname", "keywords"],
            "snippet": [
                "snippet", "summary", "description", "dek", "selftext", "selftext_html",
                "body", "text", "openingtext", "hitsentence"
            ],
            "date": ["date", "published", "pubdate", "time", "created", "created_iso", "created_utc", "alternatedateformat"],
            "url": ["url", "link", "permalink", "parenturl"],
            "author": ["author", "influencer"],
            "display_name": ["twitter screen name", "screen_name", "display_name"],
            "likes": ["likes"],
            "retweets": ["retweets", "reposts"],
            "replies": ["replies"],
            "comments": ["comments"],
            "shares": ["shares"],
            "reactions": ["reactions"],
        }

        def pick_key(group: str):
            for alias in ALIASES[group]:
                k = _norm_key(alias)
                if k in norm2orig:
                    return k
            return None

        title_key = pick_key("title")
        snippet_key = pick_key("snippet")
        date_key = pick_key("date")
        url_key = pick_key("url")
        author_key = pick_key("author")
        display_name_key = pick_key("display_name")
        likes_key = pick_key("likes")
        retweets_key = pick_key("retweets")
        replies_key = pick_key("replies")
        comments_key = pick_key("comments")
        shares_key = pick_key("shares")
        reactions_key = pick_key("reactions")

        # Helper to choose best-available text columns
        def series_from_keys(priority_group: str) -> pd.Series:
            s = pd.Series([""] * len(df_raw), dtype=object)
            for alias in ALIASES[priority_group]:
                k = _norm_key(alias)
                if k in norm2orig:
                    s_candidate = df_raw[norm2orig[k]].fillna("").astype(str).str.strip()
                    s = s.mask(~s.astype(bool), s_candidate)
            return s

        # Title / Snippet selection
        title_series = df_raw[norm2orig[title_key]].fillna("").astype(str).str.strip() if title_key else series_from_keys("title")
        snippet_series = df_raw[norm2orig[snippet_key]].fillna("").astype(str).str.strip() if snippet_key else series_from_keys("snippet")

        # Date parsing with multiple fallbacks
        def _parse_meltwater_datetime(df: pd.DataFrame, _norm2orig: Dict[str, str]) -> pd.Series:
            def _coerce(s: pd.Series) -> pd.Series:
                s = s.astype(str).str.replace(r'(?i)(am|pm)$', r' \1', regex=True)
                formats = [
                    "%d-%b-%Y %I:%M%p",
                    "%Y-%m-%d %H:%M:%S",
                    "%m/%d/%Y %I:%M %p",
                    "%d/%m/%Y %H:%M",
                ]
                result = pd.Series(pd.NaT, index=s.index)
                for fmt in formats:
                    temp = pd.to_datetime(s, errors="coerce", format=fmt, dayfirst=True)
                    result = result.fillna(temp)
                if result.isna().any():
                    temp = pd.to_datetime(s, errors="coerce", dayfirst=True)
                    result = result.fillna(temp)
                return result

            # Try primary date-like columns
            d = pd.Series(pd.NaT, index=df.index)
            for alias in ALIASES["date"]:
                k = _norm_key(alias)
                if k in _norm2orig:
                    col = _norm2orig[k]
                    d = d.fillna(_coerce(df[col]))
            return d

        date_series = _parse_meltwater_datetime(df_raw, norm2orig)

        # URL
        url_series = df_raw[norm2orig[url_key]].fillna("").astype(str).str.strip() if url_key else pd.Series([""] * len(df_raw), dtype=object)

        # Authors + Engagement
        author_series = (
            df_raw[norm2orig[author_key]].fillna("").astype(str).str.strip() if author_key else pd.Series([""] * len(df_raw), dtype=object)
        )
        display_name_series = (
            df_raw[norm2orig[display_name_key]].fillna("").astype(str).str.strip() if display_name_key else pd.Series([""] * len(df_raw), dtype=object)
        )
        likes_series = df_raw[norm2orig[likes_key]].fillna(0).astype(float) if likes_key else pd.Series([0] * len(df_raw), dtype=float)
        retweets_series = df_raw[norm2orig[retweets_key]].fillna(0).astype(float) if retweets_key else pd.Series([0] * len(df_raw), dtype=float)
        replies_series = df_raw[norm2orig[replies_key]].fillna(0).astype(float) if replies_key else pd.Series([0] * len(df_raw), dtype=float)
        comments_series = df_raw[norm2orig[comments_key]].fillna(0).astype(float) if comments_key else pd.Series([0] * len(df_raw), dtype=float)
        shares_series = df_raw[norm2orig[shares_key]].fillna(0).astype(float) if shares_key else pd.Series([0] * len(df_raw), dtype=float)
        reactions_series = df_raw[norm2orig[reactions_key]].fillna(0).astype(float) if reactions_key else pd.Series([0] * len(df_raw), dtype=float)

        df = pd.DataFrame({
            "Title": title_series,
            "Snippet": snippet_series,
            "Date": date_series,
            "URL": url_series,
            "author": author_series,
            "display_name": display_name_series,
            "Likes": likes_series,
            "Retweets": retweets_series,
            "Replies": replies_series,
            "Comments": comments_series,
            "Shares": shares_series,
            "Reactions": reactions_series,
        })

        df = df[(df["Title"].str.len() > 0) | (df["Snippet"].str.len() > 0)].copy()
        df.drop_duplicates(subset=["Title", "Snippet"], inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("DataFrame normalized")
        return df
    except Exception as e:
        logger.error(f"Error normalizing DataFrame: {e}")
        raise


def first_sentence(text: str, max_len: int = 180) -> str:
    logger.info("Extracting first sentence")
    try:
        text = str(text or "").strip()
        parts = re.split(r"(?<=[\.!?])\s+", text)
        for s in parts:
            s = s.strip()
            if len(s) >= 25:
                return s[:max_len]
        return (text[:max_len] or "").strip()
    except Exception as e:
        logger.error(f"Error in first_sentence: {e}")
        return ""


def central_indexes(emb: np.ndarray, mask_idx: np.ndarray, k: int = 5) -> List[int]:
    logger.info("Computing central indexes")
    try:
        sub = emb[mask_idx]
        if len(sub) == 0:
            logger.warning("Empty sub-embeddings")
            return []
        centroid = sub.mean(axis=0, keepdims=True)
        sims = (centroid @ sub.T).ravel()
        order = sims.argsort()[::-1][: min(k, len(sims))]
        logger.info("Central indexes computed")
        return [int(mask_idx[i]) for i in order]
    except Exception as e:
        logger.error(f"Error in central_indexes: {e}")
        return []


# --- LLM helpers ---
def llm_narrative_summary(texts: List[str], cid: int) -> Tuple[str, str, str]:
    logger.info(f"Generating summary for cluster {cid}")
    combined_text = "\n\n".join(texts)
    prompt = (
        "Create a fresh, original summary of the main gist of these social media posts clustered around a theme. "
        "Do not copy-paste, quote, or repeat any specific tweet content literally. "
        "Identify key actors, events, conflicts, resolutions, and themes in a concise 1-2 sentence summary (under 50 words). "
        "Suggest a detailed, meaningful label for this narrative (10-20 words). "
        "Suggest a short 2-4 word label derived from the summary. Make the short label highly specific and unique by including "
        "proper nouns, locations, or distinctive events mentioned in the posts to differentiate from other narratives. Avoid generic terms. "
        f"Output format: Summary: [your fresh summary] Detailed Label: [your detailed label] Short Label: [your 2-4 word label]\nPosts:\n{combined_text}"
    )

    backoff = 1.0
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model="grok-4-fast-reasoning",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            output = (response.choices[0].message.content or "").strip()
            # Robust parsing
            summary = ""; detailed_label = f"Narrative {cid}"; short_label = f"Cluster {cid}"
            if "Detailed Label:" in output and "Short Label:" in output:
                summary = output.split("Detailed Label:")[0].replace("Summary:", "").strip()
                detailed_label = output.split("Detailed Label:")[1].split("Short Label:")[0].strip()
                short_label = output.split("Short Label:")[1].strip()
            elif output:
                summary = output[:200]
            logger.info(f"Summary generated for cluster {cid}")
            return summary, detailed_label, short_label
        except APIStatusError as e:  # xAI errors surface here
            logger.warning(f"xAI API error (status {getattr(e, 'status_code', 'NA')}): {e}")
            if getattr(e, 'status_code', 0) == 429 and attempt < 3:
                time.sleep(backoff)
                backoff *= 2
                continue
            return f"Error: {e}", f"Narrative {cid}", f"Cluster {cid}"
        except Exception as e:
            logger.error(f"Error in llm_narrative_summary for cluster {cid}: {e}")
            return f"Error: {e}", f"Narrative {cid}", f"Cluster {cid}"


def llm_key_takeaways(
    narratives: Dict[int, str],
    volume_data: pd.DataFrame,
    top_authors_volume: pd.DataFrame,
    top_authors_engagement: pd.DataFrame,
    correlation_data: pd.DataFrame,
    timeline_data: pd.DataFrame | None = None,
    *,
    short_labels_map: Dict[int, str] | None = None,
    date_column: str | None = None,
) -> List[str]:
    logger.info("Generating key takeaways")
    try:
        short_labels_map = short_labels_map or {}
        narrative_summary = "\n".join([
            f"Narrative {i+1} ({short_labels_map.get(cid, f'Cluster {cid}')}): {narratives[cid]}"
            for i, cid in enumerate(sorted(narratives.keys()))
        ])
        volume_summary = "\n".join([f"{row['Narrative']}: {row['Volume']} posts" for _, row in volume_data.iterrows()])
        tav = (
            "\n".join([f"{row['display_label']}: {row['Volume']} posts" for _, row in top_authors_volume.iterrows()])
            if not top_authors_volume.empty else "No top authors by volume."
        )
        tae = (
            "\n".join([f"{row['display_label']}: {row['Engagement']} engagement" for _, row in top_authors_engagement.iterrows()])
            if not top_authors_engagement.empty else "No top authors by engagement."
        )
        cor = (
            "\n".join(
                [
                    f"{author}: " + ", ".join([f"{col}: {val}" for col, val in correlation_data.loc[author].items() if val > 0])
                    for author in correlation_data.index
                ]
            ) if not correlation_data.empty else "No author-theme correlations."
        )
        tls = (
            "\n".join(
                [
                    f"{row['Narrative']}: {row['Volume']} posts on {row[date_column].strftime('%Y-%m-%d')}"
                    for _, row in (timeline_data.iterrows() if timeline_data is not None else [])
                ]
            ) if (timeline_data is not None and not timeline_data.empty and date_column) else "No timeline data."
        )
        prompt = (
            "Analyze the following social media data to identify 3-5 key takeaways about significant trends and insights. "
            "Focus on dominant narratives, influential posters, engagement patterns, and author-theme correlations. "
            "Provide concise bullet points (each under 50 words) explaining the 'so what' of the data. "
            "Do not quote specific posts or data verbatim, but interpret the overall trends. "
            f"Narratives:\n{narrative_summary}\n\n"
            f"Narrative Volumes:\n{volume_summary}\n\n"
            f"Top Posters by Volume:\n{tav}\n\n"
            f"Top Posters by Engagement:\n{tae}\n\n"
            f"Author-Theme Correlations:\n{cor}\n\n"
            f"Timeline Trends:\n{tls}\n\n"
            "Output format: - Insight 1\n- Insight 2\n- Insight 3\n..."
        )

        backoff = 1.0
        for attempt in range(4):
            try:
                response = client.chat.completions.create(
                    model="grok-4-fast-reasoning",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.7,
                )
                output = (response.choices[0].message.content or "").strip()
                takeaways = [line.strip() for line in output.split("\n") if line.strip().startswith("-")]
                logger.info("Key takeaways generated")
                return takeaways if takeaways else ["- No significant trends identified."]
            except APIStatusError as e:
                logger.warning(f"xAI API error (status {getattr(e, 'status_code', 'NA')}): {e}")
                if getattr(e, 'status_code', 0) == 429 and attempt < 3:
                    time.sleep(backoff)
                    backoff *= 2
                    continue
                return [f"- Error generating takeaways: {e}"]
            except Exception as e:
                logger.error(f"Error generating takeaways: {e}")
                return [f"- Error generating takeaways: {e}"]
    except Exception as e:
        logger.error(f"Error in llm_key_takeaways setup: {e}")
        return [f"- Error in llm_key_takeaways setup: {e}"]


# --- Sidebar: Load & Normalize ---
st.sidebar.header("Load Data")
logger.info("Sidebar header set")
uploaded = st.sidebar.file_uploader("Upload CSV (UTF-8 / UTF-16 / TSV)", type=["csv", "tsv"])
logger.info("File uploader created")

if st.sidebar.button("Reset data & state"):
    logger.info("Reset button clicked")
    for k in [
        "df", "embeddings", "data_sig", "clustered", "labels", "baseline",
        "assigned_from_baseline", "narratives_generated",
    ]:
        st.session_state.pop(k, None)
    st.success("State cleared. Upload to proceed.")
    logger.info("State cleared")
    st.stop()


def _df_signature(d: pd.DataFrame):
    logger.info("Computing DataFrame signature")
    try:
        sig = (d.shape, pd.util.hash_pandas_object(d, index=True).sum())
    except Exception:
        sig = (d.shape, tuple(sorted(d.columns)))
    logger.info("DataFrame signature computed")
    return sig


# Load file
df = None
error = None
try:
    if uploaded is not None:
        logger.info("Processing uploaded file")
        raw = read_csv_auto(uploaded)
        if raw is not None:
            df = normalize_to_canonical(raw)
            logger.info("File processed successfully")
except Exception as e:
    error = str(e)
    logger.error(f"Error processing file: {e}")

if error:
    st.error(error)
    logger.error(f"Displayed error: {error}")

if df is not None and not df.empty:
    new_sig = _df_signature(df)
    if ("data_sig" not in st.session_state) or (st.session_state["data_sig"] != new_sig):
        st.session_state["df"] = df.reset_index(drop=True)
        st.session_state["data_sig"] = new_sig
        for k in ["embeddings", "clustered", "assigned_from_baseline", "labels", "narratives_generated"]:
            st.session_state.pop(k, None)
        st.write("Dataset Uploaded Successfully! 🎉")
        logger.info("Dataset uploaded and state updated")
    else:
        st.write("Dataset Uploaded Successfully! 🎉")
        logger.info("Dataset already loaded")
else:
    if "df" in st.session_state:
        st.info("Run clustering to generate narratives.")
        logger.info("Prompting user to run clustering")
    else:
        st.info("Upload a CSV/TSV to proceed. Required: Title, Snippet, Influencer. Optional: Date, URL, Twitter Screen Name, Likes, Reposts, Replies.")
        logger.info("Prompting user to upload CSV")
    st.stop()


# --- Embeddings ---
@st.cache_resource(show_spinner=False)
def get_model():
    logger.info("Loading SBERT model (cache_resource)")
    try:
        model = load_sbert("all-MiniLM-L6-v2")
        logger.info("SBERT model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading SBERT model: {e}")
        st.error(f"Error loading SBERT model: {e}")
        raise


@st.cache_data(show_spinner=False)
def embed_df_texts(df_in: pd.DataFrame):
    logger.info("Embedding texts")
    try:
        model = get_model()
        texts = concat_title_snippet(df_in)
        emb = embed_texts(model, texts, show_progress=False, batch_size=16)
        logger.info("Texts embedded successfully")
        return emb
    except Exception as e:
        logger.error(f"Error embedding texts: {e}")
        st.error(f"Error embedding texts: {e}")
        raise


# --- Clustering with Narrative Generation ---
st.header("Run Clusters")
st.markdown("Use the sliding scale to set the number of narrative categories for your data. You can always adjust it to capture the main narratives more exactly.")
k = st.slider("Number of clusters", 2, 8, 4, 1)
logger.info(f"Cluster slider set to {k}")

if st.button("Run clustering"):
    with st.spinner("Generating narratives—this can take up to a minute. ☕"):
        logger.info("Clustering started")
        try:
            model = get_model()  # ensure model is present (warm cache)
            logger.info("Model loaded for clustering")
            embeddings = embed_df_texts(st.session_state["df"])
            labels, _ = run_kmeans(embeddings, n_clusters=k)
            df_clustered = attach_clusters(st.session_state["df"], labels)

            st.session_state["df"] = df_clustered
            st.session_state["embeddings"] = embeddings
            st.session_state["clustered"] = True
            logger.info("Clustering completed")

            labels_map: Dict[int, str] = {}
            short_labels_map: Dict[int, str] = {}
            narratives: Dict[int, str] = {}

            for cid in sorted(df_clustered["Cluster"].unique()):
                mask_idx = np.where(df_clustered["Cluster"].values == cid)[0]
                if len(mask_idx) == 0:
                    logger.warning(f"No data for cluster {cid}")
                    continue
                top_idx = central_indexes(embeddings, mask_idx, k=5)
                central_texts = [
                    " ".join([
                        str(df_clustered.iloc[i].get("Title", "") or ""),
                        first_sentence(df_clustered.iloc[i].get("Snippet", "")),
                    ]).strip()
                    for i in top_idx
                ]
                summary, detailed_label, short_label = llm_narrative_summary(central_texts, int(cid))
                labels_map[int(cid)] = detailed_label
                short_labels_map[int(cid)] = short_label
                narratives[int(cid)] = summary

            st.session_state["labels_map"] = labels_map
            st.session_state["short_labels_map"] = short_labels_map
            st.session_state["narratives"] = narratives
            st.session_state["narratives_generated"] = True
            logger.info("Narratives generated")
            st.success("Clustering and narrative generation complete.")
        except Exception as e:
            logger.error(f"Error during clustering: {e}")
            st.error(f"Error during clustering: {e}")


# --- Custom Color Palette ---
COLOR_PALETTE = [
    '#1a3c6d', '#d32f2f', '#2e7d32', '#f57c00', '#6a1b9a', '#0288d1',
    '#c2185b', '#388e3c', '#f4a261', '#00838f', '#8e24aa', '#689f38'
]
logger.info("Color palette defined")

# --- Main Display ---
if ("df" in st.session_state and "Cluster" in st.session_state["df"].columns and st.session_state.get("narratives_generated")):
    dfc = st.session_state["df"].copy()
    emb = st.session_state.get("embeddings")

    if emb is None or len(emb) != len(dfc):
        try:
            emb = embed_df_texts(dfc)
            st.session_state["embeddings"] = emb
            logger.info("Embeddings re-generated")
        except Exception as e:
            logger.error(f"Error re-generating embeddings: {e}")
            st.error(f"Error re-generating embeddings: {e}")
            raise

    labels_map = st.session_state.get("labels_map", {})
    short_labels_map = st.session_state.get("short_labels_map", {})
    narratives = st.session_state.get("narratives", {})
    logger.info("Main display variables set")

    st.subheader("Narratives")
    for i, cid in enumerate(sorted(narratives.keys()), start=1):
        st.write(f"{i}. **{short_labels_map.get(cid, f'Cluster {cid}')}**: {narratives[cid]}")
    logger.info("Narratives displayed")

    st.subheader("Narrative Volumes")
    try:
        volume_data = dfc["Cluster"].value_counts().reset_index()
        volume_data.columns = ["Cluster", "Volume"]
        volume_data["Narrative"] = volume_data["Cluster"].map(short_labels_map)
        fig_volumes = px.bar(
            volume_data,
            x="Narrative",
            y="Volume",
            title="Narrative Volumes",
            color="Narrative",
            color_discrete_sequence=COLOR_PALETTE,
        )
        fig_volumes.update_traces(marker=dict(line=dict(width=1, color='#ffffff'), opacity=0.9), text="")
        fig_volumes.update_layout(
            font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
            title=dict(text="Narrative Volumes", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
            xaxis=dict(title="Narrative", title_font=dict(size=14), tickfont=dict(size=10)),
            yaxis=dict(title="Volume", title_font=dict(size=14), tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)"),
            plot_bgcolor="rgba(247,249,252,0.8)", paper_bgcolor="rgba(255,255,255,0)", showlegend=True,
            margin=dict(l=50, r=50, t=80, b=50), hovermode="closest", hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto"),
        )
        if not volume_data.empty:
            vmax = volume_data["Volume"].max()
            vmax_label = volume_data.loc[volume_data["Volume"].idxmax(), "Narrative"]
            fig_volumes.add_annotation(x=vmax_label, y=vmax, text=f"Peak: {vmax}", showarrow=True, arrowhead=2, ax=20, ay=-30, font=dict(size=12, color="#1a3c6d"))
        st.plotly_chart(fig_volumes, config=dict(responsive=True))
        logger.info("Narrative volumes chart displayed")
    except Exception as e:
        logger.error(f"Error displaying narrative volumes chart: {e}")
        st.error(f"Error displaying narrative volumes chart: {e}")

    # Timeline
    date_column = next((col for col in ["Date", "published"] if col in dfc.columns), None)
    timeline_data = None
    if date_column and dfc[date_column].notna().any():
        dfc[date_column] = pd.to_datetime(dfc[date_column], errors="coerce")
        min_date = dfc[date_column].min()
        max_date = dfc[date_column].max()
        span_days = (max_date - min_date).days if pd.notnull(min_date) and pd.notnull(max_date) else 0
        if span_days > 0:
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
            x_title = {'D': 'Day', '3D': '3-Day Period', 'W': 'Week', 'M': 'Month'}.get(freq, 'Period')
            try:
                timeline_data = dfc.groupby(["Cluster", pd.Grouper(key=date_column, freq=freq)]).size().reset_index(name="Volume")
                timeline_data["Narrative"] = timeline_data["Cluster"].map(short_labels_map)
                if timeline_data.empty or timeline_data["Volume"].sum() == 0:
                    st.warning("No data available for the timeline. Ensure there are posts across multiple time periods.")
                    logger.warning("No data for timeline")
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
                        color_discrete_sequence=COLOR_PALETTE,
                    )
                    fig_timeline.update_traces(line=dict(width=3), marker=dict(size=8, line=dict(width=1, color='#ffffff')),
                                              mode='lines+markers', fill='tozeroy', fillcolor='rgba(0,0,0,0.05)')
                    fig_timeline.update_layout(
                        font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                        title=dict(text="Trends Over Time", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
                        xaxis=dict(title=x_title, title_font=dict(size=14), tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)"),
                        yaxis=dict(title="Number of Posts", title_font=dict(size=14), tickfont=dict(size=12), gridcolor="rgba(0,0,0,0.1)", tickformat="d"),
                        plot_bgcolor="rgba(247,249,252,0.8)", paper_bgcolor="rgba(255,255,255,0)", showlegend=True,
                        margin=dict(l=50, r=50, t=80, b=50), hovermode="x unified", hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto"),
                    )
                    max_row = timeline_data.loc[timeline_data["Volume"].idxmax()]
                    fig_timeline.add_annotation(x=max_row[date_column], y=max_row["Volume"], text=f"Peak: {max_row['Volume']}",
                                                showarrow=True, arrowhead=2, ax=20, ay=-30, font=dict(size=12, color="#1a3c6d"))
                    st.plotly_chart(fig_timeline, config=dict(responsive=True))
                    logger.info("Timeline chart displayed")
            except KeyError as e:
                st.warning(f"No valid {date_column} data for timeline. Ensure dates are properly formatted.")
                logger.error(f"KeyError in timeline: {e}")
        else:
            st.warning("Insufficient time span in data for trends. All dates are the same or invalid.")
            logger.warning("Insufficient time span for timeline")
    else:
        st.warning("No 'Date' or 'published' column found or no valid dates. Add it to your dataset for the timeline.")
        logger.warning("No date column for timeline")

    # Top Posters
    st.subheader("Top Posters Analysis")
    logger.info("Top posters section started")
    if 'author' in dfc.columns:
        # Safe string coercion to avoid TypeErrors on NaN slicing
        def _disp_label(row):
            disp = str(row.get('display_name') or '')
            auth = str(row.get('author') or '')
            if disp:
                return f"{disp} ({auth})"[:40]
            return auth[:40]

        dfc['display_label'] = dfc.apply(_disp_label, axis=1)

        volume_by_author = (
            dfc.groupby(['author', 'display_label']).size().reset_index(name='Volume').nlargest(10, 'Volume')
        )
        engagement_cols = [c for c in ['Likes', 'Retweets', 'Replies', 'Comments', 'Shares', 'Reactions'] if c in dfc.columns]
        if engagement_cols:
            engagement_by_author = (
                dfc.groupby(['author', 'display_label'])[engagement_cols].sum().sum(axis=1).reset_index(name='Engagement').nlargest(10, 'Engagement')
            )
        else:
            engagement_by_author = pd.DataFrame({'author': [], 'display_label': [], 'Engagement': []})
            st.warning("No engagement columns (Likes, Retweets, Replies, Comments, Shares, Reactions) found in dataset. Engagement chart skipped.")
            logger.warning("No engagement columns found")

        col1, col2 = st.columns(2)
        with col1:
            try:
                fig_volume_authors = px.bar(
                    volume_by_author, x='Volume', y='display_label', title="Top 10 Posters by Volume",
                    color='display_label', color_discrete_sequence=COLOR_PALETTE, orientation='h'
                )
                fig_volume_authors.update_traces(marker=dict(line=dict(width=1, color='#ffffff'), opacity=0.9),
                                                 text=volume_by_author['Volume'], textposition='auto')
                fig_volume_authors.update_layout(
                    font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                    title=dict(text="Top 10 Posters by Volume", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
                    xaxis=dict(title="Post Count", title_font=dict(size=14), tickfont=dict(size=14)),
                    yaxis=dict(title="Poster", title_font=dict(size=14), tickfont=dict(size=14)),
                    plot_bgcolor="rgba(247,249,252,0.8)", paper_bgcolor="rgba(255,255,255,0)", showlegend=False,
                    margin=dict(l=50, r=50, t=80, b=50), hovermode="closest", hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto"),
                )
                if not volume_by_author.empty:
                    max_volume_author = volume_by_author.iloc[0]['display_label']
                    max_volume_value = volume_by_author.iloc[0]['Volume']
                    fig_volume_authors.add_annotation(x=max_volume_value, y=max_volume_author, text=f"Top: {max_volume_value}",
                                                      showarrow=True, arrowhead=2, ax=20, ay=-30, font=dict(size=12, color="#1a3c6d"))
                st.plotly_chart(fig_volume_authors, config=dict(responsive=True))
                logger.info("Volume chart displayed")
            except Exception as e:
                logger.error(f"Error displaying volume chart: {e}")
                st.error(f"Error displaying volume chart: {e}")

        with col2:
            if not engagement_by_author.empty:
                try:
                    fig_engagement_authors = px.bar(
                        engagement_by_author, x='Engagement', y='display_label', title="Top 10 Posters by Engagement",
                        color='display_label', color_discrete_sequence=COLOR_PALETTE, orientation='h'
                    )
                    fig_engagement_authors.update_traces(marker=dict(line=dict(width=1, color='#ffffff'), opacity=0.9),
                                                         text=engagement_by_author['Engagement'].round(0).astype(int), textposition='auto')
                    fig_engagement_authors.update_layout(
                        font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                        title=dict(text="Top 10 Posters by Engagement", font=dict(size=20, color="#1a3c6d"), x=0.5, xanchor="center"),
                        xaxis=dict(title="Engagement (Likes + Reposts + Replies)", title_font=dict(size=14), tickfont=dict(size=14)),
                        yaxis=dict(title="Poster", title_font=dict(size=14), tickfont=dict(size=14)),
                        plot_bgcolor="rgba(247,249,252,0.8)", paper_bgcolor="rgba(255,255,255,0)", showlegend=False,
                        margin=dict(l=50, r=50, t=80, b=50), hovermode="closest", hoverlabel=dict(bgcolor="white", font_size=12, font_family="Roboto"),
                    )
                    max_engagement_author = engagement_by_author.iloc[0]['display_label']
                    max_engagement_value = engagement_by_author.iloc[0]['Engagement']
                    fig_engagement_authors.add_annotation(x=max_engagement_value, y=max_engagement_author, text=f"Top: {int(max_engagement_value)}",
                                                          showarrow=True, arrowhead=2, ax=20, ay=-30, font=dict(size=12, color="#1a3c6d"))
                    st.plotly_chart(fig_engagement_authors, config=dict(responsive=True))
                    logger.info("Engagement chart displayed")
                except Exception as e:
                    logger.error(f"Error displaying engagement chart: {e}")
                    st.error(f"Error displaying engagement chart: {e}")

    # Top Authors by Theme
    st.subheader("Top Authors by Theme")
    logger.info("Top authors by theme section started")
    if 'author' in dfc.columns:
        author_counts = dfc.groupby(['Cluster', 'author']).size().reset_index(name='PostCount')
        author_counts['Narrative'] = author_counts['Cluster'].map(short_labels_map)
        top_authors_per_theme = {}
        for narrative in short_labels_map.values():
            theme_data = author_counts[author_counts['Narrative'] == narrative].nlargest(5, 'PostCount')
            if not theme_data.empty:
                top_authors_per_theme[narrative] = theme_data[['author', 'PostCount']].values.tolist()
        for narrative, authors in top_authors_per_theme.items():
            if authors and len(authors) > 0:
                df_theme = pd.DataFrame(authors, columns=['author', 'PostCount'])
                if not df_theme.empty:
                    df_theme['PostCount'] = pd.to_numeric(df_theme['PostCount'], errors='coerce')
                    if df_theme['PostCount'].notna().all():
                        try:
                            fig = px.bar(
                                df_theme, x='author', y='PostCount', title=f"Top Authors for {narrative}",
                                color='author', color_discrete_sequence=COLOR_PALETTE[:len(authors)], text='PostCount'
                            )
                            fig.update_traces(textposition='auto', marker=dict(line=dict(width=1, color='#ffffff'), opacity=0.9))
                            fig.update_layout(
                                font=dict(family="Roboto, sans-serif", size=12, color="#1a3c6d"),
                                title=dict(text=f"Top Authors for {narrative}", font=dict(size=16, color="#1a3c6d"), x=0.5, xanchor="center"),
                                xaxis=dict(title="Author", title_font=dict(size=12), tickfont=dict(size=10), tickangle=0),
                                yaxis=dict(title="Post Count", title_font=dict(size=12), tickfont=dict(size=10), gridcolor="rgba(0,0,0,0.1)"),
                                plot_bgcolor="rgba(247,249,252,0.8)", paper_bgcolor="rgba(255,255,255,0)", showlegend=False,
                                margin=dict(l=50, r=50, t=50, b=50), hovermode="closest", hoverlabel=dict(bgcolor="white", font_size=10, font_family="Roboto"),
                            )
                            st.plotly_chart(fig, config=dict(responsive=True))
                            logger.info(f"Top authors chart for {narrative} displayed")
                        except Exception as e:
                            logger.error(f"Error displaying top authors chart for {narrative}: {e}")
                            st.error(f"Error displaying top authors chart for {narrative}: {e}")
                    else:
                        st.warning(f"Invalid numeric data for {narrative} bar chart. Check PostCount values.")
                        logger.warning(f"Invalid numeric data for {narrative}")
                else:
                    st.warning(f"No valid data for bar chart of {narrative}")
                    logger.warning(f"No valid data for {narrative}")

    # Key Takeaways
    st.subheader("Wut Means? 🤔 Key Takeaways")
    logger.info("Key takeaways section started")
    if narratives and not volume_data.empty:
        try:
            takeaways = llm_key_takeaways(
                narratives,
                volume_data,
                volume_by_author,
                engagement_by_author,
                pd.DataFrame(),
                timeline_data,
                short_labels_map=short_labels_map,
                date_column=date_column,
            )
            for takeaway in takeaways:
                st.markdown(takeaway)
            logger.info("Key takeaways displayed")
        except Exception as e:
            logger.error(f"Error displaying key takeaways: {e}")
            st.error(f"Error displaying key takeaways: {e}")
    else:
        st.warning("Insufficient data for key takeaways. Ensure narratives and volume data are available.")
        logger.warning("Insufficient data for key takeaways")
else:
    if "df" in st.session_state and not st.session_state.get("clustered"):
        st.info("Run clustering to generate narratives.")
        logger.info("Prompting user to run clustering")
    else:
        st.info("Upload a CSV/TSV to proceed. Required: Title, Snippet, Influencer. Optional: Date, URL, Twitter Screen Name, Likes, Reposts, Replies.")
        logger.info("Prompting user to upload CSV")
    st.stop()

# --- Log download button for debugging ---
if os.path.exists('app.log'):
    with open('app.log', 'r') as f:
        st.download_button("Download app.log", f, file_name="app.log")
    logger.info("Log download button added")
