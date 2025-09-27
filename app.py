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
# Assign to existing ideas (nearest-centroid)
# ---------------------------
from sklearn.metrics.pairwise import cosine_similarity

st.subheader("Assign to existing ideas (baseline)")

if "baseline" not in st.session_state:
    st.info("No baseline saved yet. First: cluster + name ideas, then click 'Save baseline'.")
elif "df" not in st.session_state:
    st.info("Load a dataset to assign.")
else:
    df_in = st.session_state["df"]
    base = st.session_state["baseline"]

    # Controls
    sim_threshold = st.slider(
        "Minimum similarity to assign", 0.20, 0.80, 0.35, 0.01,
        help="Items below this cosine similarity will be marked as Unassigned."
    )
    assign_btn = st.button("Assign to baseline ideas")

    if assign_btn:
        with st.spinner("Embedding and assigning to ideas..."):
            emb = st.session_state.get("embeddings")
            if emb is None or len(emb) != len(df_in):
                emb = embed_df_texts(df_in)
                st.session_state["embeddings"] = emb

            # Compute nearest centroid
            C = base["centroids"]                  # [n_ideas x dim]
            L = np.array(base["labels"])           # [n_ideas]
            sims = cosine_similarity(emb, C)       # [n_items x n_ideas]
            best_idx = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)
            assigned_labels = np.where(best_sim >= sim_threshold, L[best_idx], "Unassigned")

            # Write back; also give numeric codes so downstream sections work
            label_to_int = {lab: i for i, lab in enumerate(L)}
            df_out = df_in.copy()
            df_out["Label"] = assigned_labels.astype(str)
            df_out["Cluster"] = [label_to_int.get(lab, -1) for lab in df_out["Label"]]

            st.session_state["df"] = df_out
            st.session_state["assigned_from_baseline"] = True

        # Quick summary
        counts = (df_out["Label"].value_counts()
                  .rename_axis("Idea").reset_index(name="Count")
                  .sort_values("Idea"))
        st.success("Assigned to existing ideas.")
        st.dataframe(counts, width="stretch")


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
# Name your clusters as "ideas"
# ---------------------------
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

st.subheader("Name your clusters (ideas)")

if "df" in st.session_state and "Cluster" in st.session_state["df"].columns:
    dfc = st.session_state["df"]

    # Create (or reuse) a label map in session state
    if "labels" not in st.session_state:
        st.session_state["labels"] = {}  # {cluster_id: "Idea label"}

    # Ensure normalized text + embeddings so we can show exemplars
    df_norm = ensure_text_column(dfc)
    emb = st.session_state.get("embeddings")
    if emb is None:
        emb = embed_df_texts(dfc)
        st.session_state["embeddings"] = emb

    # Build quick top-terms per cluster (comma-joined)
    terms_df = cluster_terms_dataframe(df_norm, n_terms=8)
    terms_map = {int(r.Cluster): r.TopTerms for _, r in terms_df.iterrows()}

    # Show 3 representative titles (closest to centroid) for each cluster
    for cid in sorted(dfc["Cluster"].unique()):
        idx = np.where(dfc["Cluster"].values == cid)[0]
        if len(idx) == 0:
            continue
        centroid = emb[idx].mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, emb[idx]).ravel()
        top_k = idx[sims.argsort()[::-1][:3]]
        exemplars = " | ".join(dfc.iloc[top_k]["Title"].tolist())

        default_label = st.session_state["labels"].get(cid, f"Idea {cid}")
        st.write(f"**Cluster {cid}** — Top terms: _{terms_map.get(int(cid), '')}_")
        new_label = st.text_input(
            f"Label for cluster {cid}",
            value=default_label,
            key=f"label_{cid}"
        )
        st.caption(f"Examples: {exemplars}")
        st.session_state["labels"][cid] = new_label.strip() or f"Idea {cid}"

    # Apply labels to df and keep it in state
    dfc["Label"] = dfc["Cluster"].map(st.session_state["labels"]).astype(str)
    st.session_state["df"] = dfc
else:
    st.info("Run clustering in the sidebar to name your clusters.")

# ---------------------------
# Freeze current ideas as a baseline (centroids)
# ---------------------------
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import datetime as _dt

st.subheader("Freeze ideas as baseline")

if "df" in st.session_state and "Cluster" in st.session_state["df"].columns:
    dfc = st.session_state["df"]

    # We store: labels[] and centroids [n_ideas x dim]
    if st.button("Save baseline (use these ideas for future datasets)"):
        # Ensure embeddings exist
        emb = st.session_state.get("embeddings")
        if emb is None or len(emb) != len(dfc):
            emb = embed_df_texts(dfc)
            st.session_state["embeddings"] = emb

        # Use human labels if present, else fall back to cluster ids as strings
        if "Label" not in dfc.columns:
            dfc["Label"] = dfc["Cluster"].astype(str)

        labels = []
        centroids = []
        for lab in sorted(dfc["Label"].unique()):
            idx = np.where(dfc["Label"].values == lab)[0]
            if len(idx) == 0:
                continue
            labels.append(lab)
            centroids.append(emb[idx].mean(axis=0))
        if centroids:
            st.session_state["baseline"] = {
                "labels": labels,
                "centroids": np.vstack(centroids),
                "created": _dt.datetime.utcnow().isoformat(timespec="seconds") + "Z",
            }
            st.success(f"Baseline saved with {len(labels)} ideas.")
        else:
            st.warning("Could not compute centroids (no items).")

    # Optional: clear baseline
    if "baseline" in st.session_state:
        if st.button("Clear baseline"):
            st.session_state.pop("baseline", None)
            st.success("Baseline cleared.")
else:
    st.info("Run clustering and name your ideas before saving a baseline.")

# ---------------------------
# Results: cluster counts + top terms (using labels)
# ---------------------------
st.subheader("Cluster results")

if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfc = st.session_state["df"]
    if "Cluster" not in dfc.columns:
        st.info("Run clustering in the sidebar to see cluster results.")
    else:
        # Ensure Label column exists (from the naming section); fallback to cluster id strings
        if "Label" not in dfc.columns:
            dfc["Label"] = dfc["Cluster"].astype(str)

        # Counts by label
        counts = (dfc["Label"].value_counts()
                  .sort_index()
                  .rename_axis("Label").reset_index(name="Count"))

        left, right = st.columns([1, 2])
        with left:
            st.dataframe(counts, width="stretch")
        with right:
            fig = px.bar(counts, x="Label", y="Count", title="Posts per Idea")
            st.plotly_chart(fig, width="stretch")

        # Top terms (TF-IDF) table (still computed by cluster; join label)
        df_norm = ensure_text_column(dfc)
        terms_df = cluster_terms_dataframe(df_norm, n_terms=12)
        terms_df["Label"] = terms_df["Cluster"].map(
            dfc.drop_duplicates("Cluster").set_index("Cluster")["Label"]
        ).astype(str)
        terms_df = terms_df[["Label", "TopTerms"]].sort_values("Label")
        st.subheader("Top terms per idea (TF-IDF)")
        st.dataframe(terms_df, width="stretch")

        # Downloads
        st.download_button("Download idea counts (CSV)",
                           counts.to_csv(index=False).encode("utf-8"),
                           "idea_counts.csv", "text/csv")
        st.download_button("Download idea terms (CSV)",
                           terms_df.to_csv(index=False).encode("utf-8"),
                           "idea_top_terms.csv", "text/csv")

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
# Timeline (counts or % by time bin) — using idea labels and bin filters
# ---------------------------
st.subheader("Timeline")

if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfc = st.session_state["df"]

    if "Date" not in dfc.columns or dfc["Date"].isna().all():
        st.info("No usable dates found. Ensure your CSV has a Date column (parseable).")
    elif "Cluster" not in dfc.columns:
        st.info("Run clustering in the sidebar to enable the timeline.")
    else:
        # Labels fallback
        if "Label" not in dfc.columns:
            dfc["Label"] = dfc["Cluster"].astype(str)

        # Controls
        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            bin_choice = st.selectbox("Time bin", ["Weekly", "Daily", "Monthly"], index=0)
        with c2:
            normalize_pct = st.checkbox("Normalize to % per bin", value=True,
                                        help="Shows each idea as a percent of total in that time bin.")
        with c3:
            smooth_window = st.number_input("Rolling window (bins)", min_value=1, max_value=12, value=3,
                                            help="Applies a rolling average over this many bins. 1 = no smoothing.")

        # Date range + min bin size
        df_time = dfc.copy()
        df_time["Date"] = pd.to_datetime(df_time["Date"], errors="coerce")
        df_time = df_time.dropna(subset=["Date"])
        if df_time.empty:
            st.info("No valid dates after parsing.")
            st.stop()

        min_date = df_time["Date"].min().date()
        max_date = df_time["Date"].max().date()
        default_start = max_date - pd.Timedelta(days=180)
        start, end = st.date_input("Date range", (max(default_start, min_date), max_date),
                                   min_value=min_date, max_value=max_date)

        min_bin_size = st.number_input("Hide bins with < N items", min_value=1, max_value=100, value=5)

        # Filter by date range
        mask = (df_time["Date"].dt.date >= start) & (df_time["Date"].dt.date <= end)
        df_time = df_time[mask].copy()

        # Build bin
        if bin_choice == "Weekly":
            df_time["Bin"] = df_time["Date"].dt.to_period("W").apply(lambda r: r.start_time)
        elif bin_choice == "Daily":
            df_time["Bin"] = df_time["Date"].dt.floor("D")
        else:
            df_time["Bin"] = df_time["Date"].dt.to_period("M").dt.to_timestamp()

        # Counts per bin + Label
        timeline = (df_time.groupby(["Bin","Label"]).size()
                              .reset_index(name="Count")
                              .sort_values(["Bin","Label"]))

        if timeline.empty:
            st.info("No data in the selected range.")
            st.stop()

        # Drop bins with too few total items (prevents 100% spikes from bins of size 1)
        totals = timeline.groupby("Bin")["Count"].transform("sum")
        timeline = timeline[totals >= min_bin_size].copy()

        if timeline.empty:
            st.info("All bins filtered out by the 'Hide bins with < N items' threshold. Lower it or widen the date range.")
            st.stop()

        # Normalize to % per bin (optional)
        if normalize_pct:
            totals = timeline.groupby("Bin")["Count"].transform("sum").replace(0, pd.NA)
            timeline["Value"] = (timeline["Count"] / totals) * 100
            y_label = "Percent of bin (%)"
            title_suffix = "(%)"
        else:
            timeline["Value"] = timeline["Count"].astype(float)
            y_label = "Posts"
            title_suffix = "(counts)"

        # Rolling smoothing per Label (optional)
        if smooth_window > 1:
            timeline = (timeline.sort_values(["Label","Bin"])
                                 .groupby("Label", group_keys=False)
                                 .apply(lambda d: d.assign(
                                     Value=d["Value"].rolling(smooth_window, min_periods=1).mean()
                                 )))

        fig_t = px.line(timeline, x="Bin", y="Value", color="Label",
                        title=f"Timeline — {bin_choice} {title_suffix}",
                        markers=True)
        fig_t.update_layout(yaxis_title=y_label, xaxis_title="Time",
                            legend_title_text="Idea", template="plotly_white")
        st.plotly_chart(fig_t, width="stretch")

        # Download
        st.download_button("Download timeline data (CSV)",
                           timeline.to_csv(index=False).encode("utf-8"),
                           "timeline.csv", "text/csv")

