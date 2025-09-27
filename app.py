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
# 2D Semantic Map (UMAP) — centroid view first
# ---------------------------
import numpy as np
import plotly.express as px

st.subheader("2D Semantic Map (UMAP)")

if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfm = st.session_state["df"].copy()
    if "Cluster" not in dfm.columns:
        st.info("Run clustering or Assign to baseline to enable the map.")
    else:
        if "Label" not in dfm.columns:
            dfm["Label"] = dfm["Cluster"].astype(str)

        view = st.radio("View", ["Idea centroids", "All posts"], horizontal=True)

        n_neighbors = st.slider("UMAP: n_neighbors", 5, 50, 15)
        min_dist = st.slider("UMAP: min_dist", 0.01, 0.50, 0.10, 0.01)

        if st.button("Generate 2D Map"):
            with st.spinner("Computing UMAP projection..."):
                emb = st.session_state.get("embeddings")
                if emb is None or len(emb) != len(dfm):
                    emb = embed_df_texts(dfm)
                    st.session_state["embeddings"] = emb

                coords = compute_umap(emb, n_neighbors=n_neighbors, min_dist=min_dist)
                dfm["x"], dfm["y"] = coords[:,0], coords[:,1]

                if view == "Idea centroids":
                    g = dfm.groupby("Label", as_index=False).agg(
                        x=("x","mean"), y=("y","mean"), Count=("Label","size"))
                    fig = px.scatter(g, x="x", y="y", size="Count", color="Label",
                                     text="Label", title="Ideas (centroids; size = volume)")
                    fig.update_traces(textposition="top center")
                else:
                    fig = px.scatter(dfm, x="x", y="y", color="Label",
                                     hover_name="Title", title="All posts (colored by idea)",
                                     opacity=0.8)
            fig.update_layout(template="plotly_white", legend_title_text="Idea")
            st.plotly_chart(fig, width="stretch")

# ---------------------------
# Keyword-in-Context (KWIC) — counts by idea + highlighted examples
# ---------------------------
import re
import plotly.express as px

st.subheader("Keyword-in-Context (KWIC)")
kw_col1, kw_col2, kw_col3 = st.columns([2,1,1])
with kw_col1:
    kw_pattern = st.text_input("Keyword or regex (e.g., lawsuit|settlement|class action)", "")
with kw_col2:
    kw_window = st.number_input("Context window (chars)", 20, 200, 60, 5)
with kw_col3:
    cluster_filter = st.text_input("Restrict to idea(s) (e.g., 0 or 0,2)", "")

def _hl(s, pat):
    try:
        return re.sub(f"({pat})", r"<mark>\1</mark>", s, flags=re.I)
    except re.error:
        return s  # bad regex → no highlight

if st.button("Run KWIC") and kw_pattern.strip():
    if "df" not in st.session_state:
        st.info("Load data first.")
    else:
        dfk = st.session_state["df"].copy()
        if "Label" not in dfk.columns:
            dfk["Label"] = dfk["Cluster"].astype(str)

        # Cluster filter (by Label text)
        if cluster_filter.strip():
            try:
                labs = [x.strip() for x in cluster_filter.split(",")]
                dfk = dfk[dfk["Label"].isin(labs) | dfk["Cluster"].isin([int(x) for x in labs if x.isdigit()])]
            except Exception:
                pass

        # Build normalized text once
        dfk = ensure_text_column(dfk)

        # Find matches
        rows = []
        pat = re.compile(kw_pattern, re.I)
        for _, r in dfk.iterrows():
            txt = r["_text_norm"]
            for m in pat.finditer(txt):
                start, end = m.start(), m.end()
                left = txt[max(0, start-kw_window): start]
                match = txt[start:end]
                right = txt[end: min(len(txt), end+kw_window)]
                rows.append({
                    "Label": r["Label"],
                    "Title": r["Title"],
                    "Left": left, "Match": match, "Right": right,
                    "URL": r.get("URL","")
                })
        import pandas as pd
        kwic_df = pd.DataFrame(rows)

        st.write(f"KWIC matches: **{len(kwic_df)}**")
        if kwic_df.empty:
            st.info("No matches. Try a broader term or remove cluster filter.")
        else:
            # Counts by idea
            counts = (kwic_df["Label"].value_counts()
                      .rename_axis("Idea").reset_index(name="Matches")
                      .sort_values("Idea"))
            st.dataframe(counts, width="stretch")
            figc = px.bar(counts, x="Idea", y="Matches", title="KWIC matches by idea")
            st.plotly_chart(figc, width="stretch")

            # Top highlighted examples
            st.markdown("**Examples (highlighted):**")
            for _, r in kwic_df.head(12).iterrows():
                snippet = _hl(f"{r['Left']}{r['Match']}{r['Right']}", kw_pattern)
                st.markdown(f"- **{r['Label']}** — {r['Title']}  \n{snippet}", unsafe_allow_html=True)

            # Download
            st.download_button("Download KWIC matches (CSV)",
                               kwic_df.to_csv(index=False).encode("utf-8"),
                               "kwic_matches.csv", "text/csv")


# ---------------------------
# Sentiment (VADER) — color summary + trend
# ---------------------------
import numpy as np
import plotly.express as px
import pandas as pd

st.subheader("Sentiment")

if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfs = st.session_state["df"].copy()
    if "Cluster" not in dfs.columns:
        st.info("Run clustering or Assign to baseline first.")
    else:
        if "Label" not in dfs.columns:
            dfs["Label"] = dfs["Cluster"].astype(str)

        if st.button("Compute sentiment"):
            with st.spinner("Scoring sentiment..."):
                dfs = add_vader_sentiment(dfs)   # adds Sentiment
                st.session_state["df"] = dfs

        if "Sentiment" in dfs.columns:
            # Overall colored bars by idea
            cuts = pd.cut(dfs["Sentiment"],
                          bins=[-1.0,-0.05,0.05,1.0],
                          labels=["Negative","Neutral","Positive"])
            share = (dfs.assign(Bucket=cuts)
                        .groupby(["Label","Bucket"]).size()
                        .groupby(level=0).apply(lambda s: s/s.sum()*100)
                        .rename("Percent").reset_index())
            figb = px.bar(share, x="Label", y="Percent", color="Bucket",
                          title="Sentiment by idea (share)", barmode="stack",
                          color_discrete_map={"Positive":"#2ca02c","Neutral":"#9e9e9e","Negative":"#d62728"})
            st.plotly_chart(figb, width="stretch")

            # Trend over time (weekly mean sentiment)
            if "Date" in dfs.columns and dfs["Date"].notna().any():
                dfs["Date"] = pd.to_datetime(dfs["Date"], errors="coerce")
                w = dfs.dropna(subset=["Date"]).copy()
                w["Week"] = w["Date"].dt.to_period("W").apply(lambda r: r.start_time)
                trend = (w.groupby(["Week","Label"])["Sentiment"]
                           .mean().reset_index())
                figt = px.line(trend, x="Week", y="Sentiment", color="Label",
                               title="Mean sentiment over time (weekly)",
                               markers=True, color_discrete_sequence=px.colors.qualitative.Set2)
                figt.update_yaxes(range=[-1,1])
                st.plotly_chart(figt, width="stretch")
        else:
            st.caption("Click **Compute sentiment** to score and visualize.")


# ---------------------------
# Timeline (smart defaults, labels, smoothing, auto-fallback)
# ---------------------------
import pandas as pd
import plotly.express as px

st.subheader("Timeline")

if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfc = st.session_state["df"].copy()

    if "Date" not in dfc.columns or dfc["Date"].isna().all():
        st.info("No usable dates found. Ensure your CSV has a Date column (parseable).")
    elif "Cluster" not in dfc.columns:
        st.info("Run clustering (or Assign to baseline) to enable the timeline.")
    else:
        if "Label" not in dfc.columns:
            dfc["Label"] = dfc["Cluster"].astype(str)

        # Parse dates and bound range
        dfc["Date"] = pd.to_datetime(dfc["Date"], errors="coerce")
        dfc = dfc.dropna(subset=["Date"])
        if dfc.empty:
            st.info("No valid dates after parsing.")
            st.stop()

        min_date = dfc["Date"].min().date()
        max_date = dfc["Date"].max().date()

        # Heuristic bin default: small range->Daily, medium->Weekly, big->Monthly
        days_span = (max_date - min_date).days or 1
        default_bin = "Daily" if days_span <= 45 else ("Weekly" if days_span <= 400 else "Monthly")

        c1, c2, c3 = st.columns([1,1,1])
        with c1:
            bin_choice = st.selectbox("Time bin", ["Daily","Weekly","Monthly"],
                                      index=["Daily","Weekly","Monthly"].index(default_bin))
        with c2:
            normalize_pct = st.checkbox("Normalize to % per bin", value=True)
        with c3:
            smooth_window = st.number_input("Rolling window (bins)", 1, 12, 3)

        # Date range + min bin size
        default_start = max_date - pd.Timedelta(days=min(180, days_span))
        start, end = st.date_input("Date range",
                                   (max(default_start, min_date), max_date),
                                   min_value=min_date, max_value=max_date)
        min_bin_size = st.number_input("Hide bins with < N items", 1, 100, 4)

        # Filter to date range
        mask = (dfc["Date"].dt.date >= start) & (dfc["Date"].dt.date <= end)
        dfT = dfc.loc[mask].copy()

        # Build bin column
        def make_bin(df, choice):
            if choice == "Daily":
                return df["Date"].dt.floor("D")
            if choice == "Weekly":
                return df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
            return df["Date"].dt.to_period("M").dt.to_timestamp()

        # Try chosen bin; if everything gets filtered out, auto-fallback to looser settings
        tried = []
        for choice in [bin_choice, "Weekly", "Monthly"]:
            if choice in tried: 
                continue
            tried.append(choice)
            dfT["Bin"] = make_bin(dfT, choice)
            tl = (dfT.groupby(["Bin","Label"]).size()
                      .reset_index(name="Count")
                      .sort_values(["Bin","Label"]))
            if tl.empty:
                continue
            totals = tl.groupby("Bin")["Count"].transform("sum")
            tl2 = tl[totals >= min_bin_size].copy()
            if not tl2.empty:
                bin_choice = choice
                timeline = tl2
                break
        else:
            st.info("All bins filtered out. Lower the threshold or widen the date range.")
            st.stop()

        # Normalize or keep counts
        if normalize_pct:
            totals = timeline.groupby("Bin")["Count"].transform("sum").replace(0, pd.NA)
            timeline["Value"] = (timeline["Count"] / totals) * 100
            ylab, suffix = "Percent of bin (%)", "(%)"
        else:
            timeline["Value"] = timeline["Count"].astype(float)
            ylab, suffix = "Posts", "(counts)"

        # Rolling smoothing per label
        if smooth_window > 1:
            timeline = (timeline.sort_values(["Label","Bin"])
                                .groupby("Label", group_keys=False)
                                .apply(lambda d: d.assign(
                                    Value=d["Value"].rolling(smooth_window, min_periods=1).mean()
                                )))

        fig = px.line(timeline, x="Bin", y="Value", color="Label",
                      title=f"Timeline — {bin_choice} {suffix}", markers=True)
        fig.update_layout(yaxis_title=ylab, xaxis_title="Time",
                          legend_title_text="Idea", template="plotly_white")
        st.plotly_chart(fig, width="stretch")

        st.download_button("Download timeline data (CSV)",
                           timeline.to_csv(index=False).encode("utf-8"),
                           "timeline.csv", "text/csv")
