# app.py — Narrative Analysis (full app)
# Features:
# - Upload & normalize CSV (Title/Snippet/Date/URL)
# - Embeddings (SBERT, cached)
# - KMeans clustering (slider for K)
# - TF-IDF top terms per cluster
# - UMAP 2D semantic map (Plotly)
# - KWIC search (regex) + counts + co-occurrence
# - VADER sentiment per row + summary per cluster

import streamlit as st
import pandas as pd
import plotly.express as px

# --- Import reusable modules you added under src/ ---
from narrative.narrative_io import normalize_to_canonical
from narrative.narrative_embed import load_sbert, concat_title_snippet, embed_texts
from narrative.narrative_cluster import run_kmeans, attach_clusters, cluster_counts
from narrative.narrative_terms import ensure_text_column, cluster_terms_dataframe
from narrative.narrative_map import compute_umap, attach_coords, build_hover_html
from narrative.narrative_kwic import kwic_search, kwic_counts_by_cluster, kwic_cooccurrence
from narrative.narrative_sentiment import add_vader_sentiment, sentiment_by_cluster

# ---------------------------
# Page setup
# ---------------------------
st.set_page_config(page_title="Narrative Analysis", layout="wide")
st.title("Narrative Analysis")

# ---------------------------
# Section 1 — Load & normalize (idempotent; preserves clustered df)
# ---------------------------
st.sidebar.header("Load Data")
uploaded = st.sidebar.file_uploader("Upload CSV (UTF-8)", type=["csv"])
use_demo = st.sidebar.checkbox("Use tiny demo data", value=False)

# A Reset button so you can intentionally clear state and start clean
if st.sidebar.button("Reset data & state"):
    for k in ["df", "embeddings", "data_sig", "clustered"]:
        st.session_state.pop(k, None)
    st.success("State cleared. Upload or enable demo again.")
    st.stop()

def _df_signature(d: pd.DataFrame):
    """A cheap signature so we only replace state when data actually changes."""
    try:
        sig = (d.shape, pd.util.hash_pandas_object(d, index=True).sum())
    except Exception:
        sig = (d.shape, tuple(sorted(d.columns)))
    return sig

df = None
error = None

try:
    if uploaded is not None:
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

# If we received new data, update state; otherwise keep whatever (possibly clustered) df we already had
if df is not None and not df.empty:
    new_sig = _df_signature(df)
    if ("data_sig" not in st.session_state) or (st.session_state["data_sig"] != new_sig):
        # New dataset → store it and clear downstream state
        st.session_state["df"] = df.reset_index(drop=True)
        st.session_state["data_sig"] = new_sig
        st.session_state.pop("embeddings", None)
        st.session_state.pop("clustered", None)
        st.success(f"Loaded {len(df)} rows (new dataset).")
    else:
        # Same dataset as before → DO NOT overwrite clustered df
        st.success(f"Loaded {len(st.session_state['df'])} rows (using cached state).")
else:
    if "df" not in st.session_state:
        st.info(
            "Upload a CSV or enable the demo to proceed.\n\n"
            "Required columns (any alias): Title/Headline and Snippet/Summary/Description.\n"
            "Optional: Date, URL."
        )
        st.stop()

# Show a quick preview (whatever is in state now)
st.dataframe(st.session_state["df"].head(20), width="stretch")


# ---------------------------
# Embeddings (cached)
# ---------------------------
st.sidebar.header("Clustering")
k = st.sidebar.slider("Number of clusters (KMeans)", 2, 12, 6, 1)
run_btn = st.sidebar.button("Run clustering")

@st.cache_resource
def get_model():
    return load_sbert("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def embed_df_texts(df_in: pd.DataFrame):
    model = get_model()
    texts = concat_title_snippet(df_in)
    emb = embed_texts(model, texts, show_progress=False)
    return emb

if run_btn:
    with st.spinner("Embedding and clustering..."):
        embeddings = embed_df_texts(st.session_state["df"])
        labels, _ = run_kmeans(embeddings, n_clusters=k)
        df_clustered = attach_clusters(st.session_state["df"], labels)
        st.session_state["df"] = df_clustered
        st.session_state["embeddings"] = embeddings
    st.success("Clustering complete.")

dfc = st.session_state["df"]
embeddings = st.session_state.get("embeddings", None)

# ---------------------------
# Results: cluster counts + top terms
# ---------------------------
st.subheader("Cluster results")

# Local guards so this section doesn't stop the whole app
if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfc = st.session_state["df"]

    if "Cluster" not in dfc.columns:
        st.info("Run clustering in the sidebar to see cluster results.")
    else:
        # Cluster sizes table + bar chart
        counts = cluster_counts(dfc)

        left, right = st.columns([1, 2])
        with left:
            st.dataframe(counts, use_container_width=True)

        with right:
            fig = px.bar(counts, x="Cluster", y="Count", title="Posts per Cluster")
            st.plotly_chart(fig, use_container_width=True)

        # Top terms per cluster (TF-IDF)
        st.subheader("Top terms per cluster (TF-IDF)")
        df_norm = ensure_text_column(dfc)  # builds _text_norm if missing
        terms_df = cluster_terms_dataframe(df_norm, n_terms=12)
        st.dataframe(terms_df, use_container_width=True)

        # Download buttons (cluster counts / top terms)
        cc_csv = counts.to_csv(index=False).encode("utf-8")
        tt_csv = terms_df.to_csv(index=False).encode("utf-8")
        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "Download cluster counts (CSV)",
                cc_csv, "cluster_counts.csv", "text/csv"
            )
        with col_dl2:
            st.download_button(
                "Download top terms (CSV)",
                tt_csv, "cluster_top_terms.csv", "text/csv"
            )

# ---------------------------
# 2D Semantic Map (UMAP)
# ---------------------------
st.subheader("2D Semantic Map (UMAP)")

if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfc = st.session_state["df"]

    # Require clusters (map is colored by cluster)
    if "Cluster" not in dfc.columns:
        st.info("Run clustering in the sidebar first to enable the map.")
    else:
        col_a, col_b = st.columns(2)
        with col_a:
            n_neighbors = st.slider("UMAP: n_neighbors", 5, 50, 15, 1,
                                    help="Lower = more local structure; higher = smoother clusters.")
        with col_b:
            min_dist = st.slider("UMAP: min_dist", 0.01, 0.50, 0.10, 0.01,
                                 help="Lower = tighter clusters; higher = more spread out.")

        if st.button("Generate 2D Map"):
            with st.spinner("Computing UMAP projection..."):
                # Robust: if embeddings are missing for any reason, recompute now
                emb = st.session_state.get("embeddings")
                if emb is None:
                    emb = embed_df_texts(dfc)
                    st.session_state["embeddings"] = emb

                coords = compute_umap(emb, n_neighbors=n_neighbors, min_dist=min_dist)
                df_map = attach_coords(dfc, coords)
                df_map["_hover"] = build_hover_html(df_map)

            fig_map = px.scatter(
                df_map, x="x", y="y", color="Cluster",
                hover_name="Title",
                hover_data={"x": False, "y": False, "_hover": True, "Cluster": True},
                title="UMAP: hover to inspect posts", opacity=0.85
            )
            fig_map.update_traces(hovertemplate="%{customdata[0]}")
            fig_map.update_traces(customdata=df_map[["_hover"]].to_numpy())
            fig_map.update_layout(legend_title_text="Cluster", template="plotly_white")
            st.plotly_chart(fig_map, use_container_width=True)  # OK to keep; see note in logs


# ---------------------------
# KWIC: keyword-in-context
# ---------------------------
st.subheader("Keyword-in-Context (KWIC)")
kw_col1, kw_col2, kw_col3 = st.columns([2,1,1])
with kw_col1:
    kw_pattern = st.text_input("Keyword or regex (e.g., lawsuit|settlement|class action)", "")
with kw_col2:
    kw_window = st.number_input("Context window (chars)", min_value=20, max_value=200, value=50, step=5)
with kw_col3:
    cluster_filter = st.text_input("Restrict to cluster(s) (e.g., 0 or 0,2)", "")

if st.button("Run KWIC") and kw_pattern.strip():
    # Parse cluster filter
    clusters = None
    if cluster_filter.strip():
        try:
            if "," in cluster_filter:
                clusters = [int(x.strip()) for x in cluster_filter.split(",")]
            else:
                clusters = int(cluster_filter.strip())
        except:
            st.warning("Could not parse cluster filter; searching all clusters.")
            clusters = None

    with st.spinner("Searching..."):
        kwic_df = kwic_search(dfc, pattern=kw_pattern, window=int(kw_window), clusters=clusters)
    st.write(f"KWIC matches: {len(kwic_df)}")

    if not kwic_df.empty:
        # Counts by cluster
        kwic_counts = kwic_counts_by_cluster(kwic_df)
        st.write("Matches by cluster:")
        st.dataframe(kwic_counts, use_container_width=True)

        # Co-occurrence (top terms in KWIC windows)
        cooc_df = kwic_cooccurrence(kwic_df, top_n=20)
        st.write("Top co-occurring terms per cluster (first rows):")
        st.dataframe(cooc_df.head(20), use_container_width=True)

        # Downloads
        kwic_csv = kwic_df.to_csv(index=False).encode("utf-8")
        cooc_csv = cooc_df.to_csv(index=False).encode("utf-8")
        kc_csv   = kwic_counts.to_csv(index=False).encode("utf-8")

        dl1, dl2, dl3 = st.columns(3)
        with dl1:
            st.download_button("Download KWIC matches (CSV)", kwic_csv, "kwic_matches.csv", "text/csv")
        with dl2:
            st.download_button("Download KWIC co-occurrence (CSV)", cooc_csv, "kwic_cooccurrence.csv", "text/csv")
        with dl3:
            st.download_button("Download KWIC counts (CSV)", kc_csv, "kwic_counts_by_cluster.csv", "text/csv")
    else:
        st.info("No matches for that pattern.")

# ---------------------------
# Sentiment (VADER)
# ---------------------------
st.subheader("Sentiment (VADER)")

# Guard: data must be loaded
if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfc = st.session_state["df"]

    # Guard: we summarize by cluster, so require clusters
    if "Cluster" not in dfc.columns:
        st.info("Run clustering first to see per-cluster sentiment.")
    else:
        # Primary action
        compute_sent = st.button("Compute sentiment")

        if compute_sent:
            with st.spinner("Scoring sentiment..."):
                df_sent = add_vader_sentiment(dfc)  # adds df["Sentiment"]
                st.session_state["df"] = df_sent    # keep the new column
                dfc = df_sent

            st.success("Sentiment computed.")

            # Build and show the summary now that Sentiment exists
            sent_tbl = sentiment_by_cluster(dfc)
            st.dataframe(sent_tbl, width="stretch")

            # Downloads (only after we have sent_tbl)
            sent_csv = sent_tbl.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download sentiment summary (CSV)",
                sent_csv,
                "sentiment_by_cluster.csv",
                "text/csv"
            )

        # If sentiment was computed previously in this session, show latest table
        elif "Sentiment" in dfc.columns:
            sent_tbl = sentiment_by_cluster(dfc)
            st.dataframe(sent_tbl, width="stretch")

            sent_csv = sent_tbl.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download sentiment summary (CSV)",
                sent_csv,
                "sentiment_by_cluster.csv",
                "text/csv"
            )
        else:
            st.caption("Click **Compute sentiment** to score and summarize by cluster.")

# ---------------------------
# Timeline (counts or % by time bin)
# ---------------------------
import pandas as pd
import plotly.express as px

st.subheader("Timeline")

# Local guards
if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfc = st.session_state["df"]

    if "Date" not in dfc.columns or dfc["Date"].isna().all():
        st.info("No usable dates found. Ensure your CSV has a Date column (parseable).")
    elif "Cluster" not in dfc.columns:
        st.info("Run clustering in the sidebar to enable the timeline.")
    else:
        # Controls
        col1, col2, col3 = st.columns([1,1,1])
        with col1:
            bin_choice = st.selectbox("Time bin", ["Weekly", "Daily", "Monthly"], index=0)
        with col2:
            normalize_pct = st.checkbox("Normalize to % per bin", value=True,
                                        help="Shows each cluster as a percent of total in that time bin.")
        with col3:
            smooth_window = st.number_input("Rolling window (bins)", min_value=1, max_value=12, value=1,
                                            help="Applies a rolling average over this many bins. 1 = no smoothing.")

        # Optional cluster filter
        clusters_available = sorted(dfc["Cluster"].unique().tolist())
        chosen = st.multiselect("Clusters to include", clusters_available, default=clusters_available)

        # Build time bin
        df_time = dfc.copy()
        df_time["Date"] = pd.to_datetime(df_time["Date"], errors="coerce")
        df_time = df_time.dropna(subset=["Date"])
        if not chosen:
            st.warning("No clusters selected.")
            st.stop()
        df_time = df_time[df_time["Cluster"].isin(chosen)]

        if bin_choice == "Weekly":
            df_time["Bin"] = df_time["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        elif bin_choice == "Daily":
            df_time["Bin"] = df_time["Date"].dt.floor("D")
        else:  # Monthly
            df_time["Bin"] = df_time["Date"].dt.to_period("M").dt.to_timestamp()

        # Counts per bin+cluster
        timeline = (df_time.groupby(["Bin","Cluster"]).size()
                              .reset_index(name="Count")
                              .sort_values(["Bin","Cluster"]))

        # Normalize to % per bin (optional)
        if normalize_pct:
            totals = timeline.groupby("Bin")["Count"].transform("sum").replace(0, pd.NA)
            timeline["Value"] = (timeline["Count"] / totals) * 100
            y_label = "Percent of bin (%)"
        else:
            timeline["Value"] = timeline["Count"].astype(float)
            y_label = "Posts"

        # Rolling smoothing per cluster (optional)
        if smooth_window > 1:
            timeline = (timeline.sort_values(["Cluster","Bin"])
                                 .groupby("Cluster", group_keys=False)
                                 .apply(lambda d: d.assign(
                                     Value=d["Value"].rolling(smooth_window, min_periods=1).mean()
                                 )))

        # Plot
        if timeline.empty:
            st.info("No data after filtering. Try different clusters or bin size.")
        else:
            fig_t = px.line(timeline, x="Bin", y="Value", color="Cluster",
                            title=f"Timeline — {bin_choice} ({'%' if normalize_pct else 'counts'})",
                            markers=True)
            fig_t.update_layout(yaxis_title=y_label, xaxis_title="Time", legend_title_text="Cluster",
                                template="plotly_white")
            st.plotly_chart(fig_t, use_container_width=True)

            # Download
            dl_csv = timeline.to_csv(index=False).encode("utf-8")
            st.download_button("Download timeline data (CSV)", dl_csv, "timeline.csv", "text/csv")
