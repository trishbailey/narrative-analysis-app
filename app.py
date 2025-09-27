# app.py — Narrative Analysis (lean, idea-focused)
# Features:
# - Upload & normalize CSV (Title/Snippet/Date/URL)
# - Embeddings (SBERT, cached)
# - KMeans clustering (one click)
# - Name clusters as human "ideas" + save a baseline (centroids)
# - Assign any future dataset to the same ideas (nearest-centroid)
# - One-line narrative summary per idea (SVO over central exemplars)
# - Idea volumes (counts & share)
# - Sentiment (colored bars + weekly trend)
# - Timeline (smart defaults, % or counts, smoothing, bin filters)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import re
import datetime as _dt
from sklearn.metrics.pairwise import cosine_similarity

# --- Reusable modules ---
from narrative.narrative_io import normalize_to_canonical
from narrative.narrative_embed import load_sbert, concat_title_snippet, embed_texts
from narrative.narrative_cluster import run_kmeans, attach_clusters
from narrative.narrative_sentiment import add_vader_sentiment
from narrative.narrative_summarize import summarize_narratives
from narrative.narrative_io import read_csv_auto, normalize_to_canonical


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
    for k in ["df", "embeddings", "data_sig", "clustered", "labels", "baseline", "assigned_from_baseline"]:
        st.session_state.pop(k, None)
    st.success("State cleared. Upload or enable demo again.")
    st.stop()

def _df_signature(d: pd.DataFrame):
    """Only replace state when data actually changes."""
    try:
        sig = (d.shape, pd.util.hash_pandas_object(d, index=True).sum())
    except Exception:
        sig = (d.shape, tuple(sorted(d.columns)))
    return sig

df = None
error = None
try:
    if uploaded is not None:
raw = read_csv_auto(uploaded)   # robust reader (UTF-16/UTF-8-SIG/TSV)
df  = normalize_to_canonical(raw)

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
        st.session_state["df"] = df.reset_index(drop=True)
        st.session_state["data_sig"] = new_sig
        for k in ["embeddings", "clustered", "assigned_from_baseline"]:
            st.session_state.pop(k, None)
        st.success(f"Loaded {len(df)} rows (new dataset).")
    else:
        st.success(f"Loaded {len(st.session_state['df'])} rows (using cached state).")
else:
    if "df" not in st.session_state:
        st.info(
            "Upload a CSV or enable the demo to proceed.\n\n"
            "Required: Title, Snippet. Optional: Date, URL."
        )
        st.stop()

# Show a quick preview (whatever is in state now)
st.dataframe(st.session_state["df"].head(20), width="stretch")

# ---------------------------
# Embeddings (cached)
# ---------------------------
@st.cache_resource
def get_model():
    return load_sbert("all-MiniLM-L6-v2")

@st.cache_data(show_spinner=False)
def embed_df_texts(df_in: pd.DataFrame):
    model = get_model()
    texts = concat_title_snippet(df_in)
    emb = embed_texts(model, texts, show_progress=False)
    return emb

# ---------------------------
# Assign to existing ideas (nearest-centroid)
# ---------------------------
st.subheader("Assign to existing ideas (baseline)")
if "baseline" not in st.session_state:
    st.info("No baseline saved yet. First: run clustering, name ideas, then click 'Save baseline'.")
elif "df" not in st.session_state:
    st.info("Load a dataset to assign.")
else:
    df_in = st.session_state["df"]
    base = st.session_state["baseline"]

    sim_threshold = st.slider(
        "Minimum similarity to assign", 0.20, 0.80, 0.35, 0.01,
        help="Items below this cosine similarity are marked Unassigned."
    )
    assign_btn = st.button("Assign to baseline ideas")

    if assign_btn:
        with st.spinner("Embedding and assigning to ideas..."):
            emb = st.session_state.get("embeddings")
            if emb is None or len(emb) != len(df_in):
                emb = embed_df_texts(df_in)
                st.session_state["embeddings"] = emb

            C = base["centroids"]                    # [n_ideas x dim]
            L = np.array(base["labels"])             # [n_ideas]
            sims = cosine_similarity(emb, C)         # [n_items x n_ideas]
            best_idx = sims.argmax(axis=1)
            best_sim = sims.max(axis=1)
            assigned_labels = np.where(best_sim >= sim_threshold, L[best_idx], "Unassigned")

            label_to_int = {lab: i for i, lab in enumerate(L)}
            df_out = df_in.copy()
            df_out["Label"] = assigned_labels.astype(str)
            df_out["Cluster"] = [label_to_int.get(lab, -1) for lab in df_out["Label"]]

            st.session_state["df"] = df_out
            st.session_state["assigned_from_baseline"] = True

        counts = (df_out["Label"].value_counts()
                  .rename_axis("Idea").reset_index(name="Count")
                  .sort_values("Idea"))
        st.success("Assigned to existing ideas.")
        st.dataframe(counts, width="stretch")

# ---------------------------
# Clustering (one click)
# ---------------------------
st.sidebar.header("Clustering")
k = st.sidebar.slider("Number of clusters (KMeans)", 2, 12, 6, 1)
run_btn = st.sidebar.button("Run clustering")

if run_btn:
    with st.spinner("Embedding and clustering..."):
        embeddings = embed_df_texts(st.session_state["df"])
        labels, _ = run_kmeans(embeddings, n_clusters=k)
        df_clustered = attach_clusters(st.session_state["df"], labels)
        st.session_state["df"] = df_clustered
        st.session_state["embeddings"] = embeddings
        st.session_state["clustered"] = True
    st.success("Clustering complete.")

# ---------------------------
# Auto-label ideas (from narrative one-liners)
# ---------------------------
from narrative.narrative_summarize import summarize_narratives

st.subheader("Auto-label ideas")

if ("df" in st.session_state) and ("Cluster" in st.session_state["df"].columns):
    dfc = st.session_state["df"].copy()

    # Ensure embeddings exist
    emb = st.session_state.get("embeddings")
    if (emb is None) or (len(emb) != len(dfc)):
        emb = embed_df_texts(dfc)
        st.session_state["embeddings"] = emb

    # Build one-liners first (use cluster ids as the grouping key)
    narr_df = summarize_narratives(dfc, emb, label_col="Cluster", topk_central=5)

    # Create short labels from narratives (first few informative words)
    STOP = {
        "the","a","an","and","or","for","to","of","in","on","at","with","by","after","before","from",
        "is","are","was","were","be","been","being","it","its","this","that","these","those","as",
        "over","under","about","into","out","up","down","than","then","but","so","if","while","when",
        "you","we","they","he","she","i"
    }
    def short_label(s: str, max_words: int = 5) -> str:
        s = str(s or "")
        toks = re.sub(r"[^A-Za-z0-9\s]", " ", s).lower().split()
        kept = [w for w in toks if w not in STOP]
        return (" ".join(kept[:max_words]) or "idea").title()

    # Map cluster id -> label via narrative rows
    labels_map = {}
    used = set()
    for _, r in narr_df.iterrows():
        # narr_df["Idea"] is the cluster id as string because we passed label_col="Cluster"
        try:
            cid = int(r["Idea"])
        except Exception:
            continue
        base = short_label(r.get("Narrative", ""))
        label = base or f"Idea {cid}"
        n = 2
        while label in used:
            label = f"{base} #{n}"
            n += 1
        used.add(label)
        labels_map[cid] = label

    # Apply and persist
    dfc["Label"] = dfc["Cluster"].map(labels_map).astype(str)
    st.session_state["labels"] = labels_map
    st.session_state["df"] = dfc

    # Show mapping table
    st.dataframe(
        pd.DataFrame(
            {"Cluster": sorted(labels_map.keys()),
             "Idea": [labels_map[c] for c in sorted(labels_map.keys())]}
        ),
        width="stretch"
    )
else:
    st.info("Run clustering (or Assign to baseline) to create ideas first.")


# ---------------------------
# Freeze current ideas as a baseline (centroids)
# ---------------------------
st.subheader("Freeze ideas as baseline")
if "df" in st.session_state and "Cluster" in st.session_state["df"].columns:
    dfc = st.session_state["df"]

    if st.button("Save baseline (use these ideas for future datasets)"):
        emb = st.session_state.get("embeddings")
        if emb is None or len(emb) != len(dfc):
            emb = embed_df_texts(dfc)
            st.session_state["embeddings"] = emb

        if "Label" not in dfc.columns:
            dfc["Label"] = dfc["Cluster"].astype(str)

        labels, centroids = [], []
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

    if "baseline" in st.session_state:
        if st.button("Clear baseline"):
            st.session_state.pop("baseline", None)
            st.success("Baseline cleared.")
else:
    st.info("Run clustering and name your ideas before saving a baseline.")

# ---------------------------
# Narratives (one-line summaries) — primary view
# ---------------------------
st.subheader("Narratives (one-line summaries)")
if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfn = st.session_state["df"].copy()
    if "Cluster" not in dfn.columns:
        st.info("Run clustering or Assign to baseline first.")
    else:
        if "Label" not in dfn.columns:
            dfn["Label"] = dfn["Cluster"].astype(str)

        topk = st.number_input(
            "Use top-k central posts per idea", 1, 15, 5,
            help="Higher = more robust summary; slower."
        )
        if st.button("Generate narrative summaries"):
            with st.spinner("Summarizing ideas..."):
                emb = st.session_state.get("embeddings")
                if emb is None or len(emb) != len(dfn):
                    emb = embed_df_texts(dfn)
                    st.session_state["embeddings"] = emb
                narr_df = summarize_narratives(dfn, emb, label_col="Label", topk_central=topk)

            st.success("Narratives generated.")
            st.dataframe(narr_df, width="stretch")
            st.download_button(
                "Download narratives (CSV)",
                narr_df.to_csv(index=False).encode("utf-8"),
                "narratives.csv", "text/csv"
            )

# ---------------------------
# Idea volumes (counts + share)
# ---------------------------
st.subheader("Idea volumes")
if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dfv = st.session_state["df"]
    if "Cluster" not in dfv.columns:
        st.info("Run clustering (or Assign to baseline) to see idea volumes.")
    else:
        if "Label" not in dfv.columns:
            dfv["Label"] = dfv["Cluster"].astype(str)

        counts = (dfv["Label"].value_counts()
                  .rename_axis("Idea")
                  .reset_index(name="Count")
                  .sort_values("Idea"))
        counts["Share_%"] = (counts["Count"] / counts["Count"].sum() * 100).round(2)

        left, right = st.columns([1, 2])
        with left:
            st.dataframe(counts, width="stretch")
        with right:
            fig = px.bar(counts, x="Idea", y="Count", title="Posts per Idea")
            st.plotly_chart(fig, width="stretch")

        st.download_button(
            "Download idea volumes (CSV)",
            counts.to_csv(index=False).encode("utf-8"),
            "idea_volumes.csv",
            "text/csv",
        )

# ---------------------------
# Sentiment (VADER) — color summary + trend
# ---------------------------
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
            # Robust percent shares by idea (pivot -> percent -> melt)
            cuts = pd.cut(
                dfs["Sentiment"],
                bins=[-1.0, -0.05, 0.05, 1.0],
                labels=["Negative", "Neutral", "Positive"]
            )
            tab = (dfs.assign(Bucket=cuts)
                     .groupby(["Label","Bucket"], dropna=False)
                     .size()
                     .unstack(fill_value=0)
                     .rename_axis(index="Label", columns="Bucket"))
            pct = (tab.div(tab.sum(axis=1).replace(0, pd.NA), axis=0) * 100).reset_index()
            share = pct.melt(id_vars="Label", var_name="Bucket", value_name="Percent").fillna(0)

            figb = px.bar(
                share, x="Label", y="Percent", color="Bucket",
                title="Sentiment by idea (share)", barmode="stack",
                color_discrete_map={
                    "Positive": "#2ca02c", "Neutral": "#9e9e9e", "Negative": "#d62728"
                }
            )
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
st.subheader("Timeline")
if "df" not in st.session_state:
    st.info("Load data first.")
else:
    dft = st.session_state["df"].copy()
    if "Date" not in dft.columns or dft["Date"].isna().all():
        st.info("No usable dates found. Ensure your CSV has a Date column (parseable).")
    elif "Cluster" not in dft.columns:
        st.info("Run clustering (or Assign to baseline) to enable the timeline.")
    else:
        if "Label" not in dft.columns:
            dft["Label"] = dft["Cluster"].astype(str)

        dft["Date"] = pd.to_datetime(dft["Date"], errors="coerce")
        dft = dft.dropna(subset=["Date"])
        if dft.empty:
            st.info("No valid dates after parsing.")
            st.stop()

        min_date = dft["Date"].min().date()
        max_date = dft["Date"].max().date()
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

        default_start = max_date - pd.Timedelta(days=min(180, days_span))
        start, end = st.date_input("Date range",
                                   (max(default_start, min_date), max_date),
                                   min_value=min_date, max_value=max_date)
        min_bin_size = st.number_input("Hide bins with < N items", 1, 100, 4)

        mask = (dft["Date"].dt.date >= start) & (dft["Date"].dt.date <= end)
        dfT = dft.loc[mask].copy()

        def make_bin(df, choice):
            if choice == "Daily":
                return df["Date"].dt.floor("D")
            if choice == "Weekly":
                return df["Date"].dt.to_period("W").apply(lambda r: r.start_time)
            return df["Date"].dt.to_period("M").dt.to_timestamp()

        # Try chosen bin; if empty after filters, auto-fallback to looser bins
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

        if normalize_pct:
            totals = timeline.groupby("Bin")["Count"].transform("sum").replace(0, pd.NA)
            timeline["Value"] = (timeline["Count"] / totals) * 100
            ylab, suffix = "Percent of bin (%)", "(%)"
        else:
            timeline["Value"] = timeline["Count"].astype(float)
            ylab, suffix = "Posts", "(counts)"

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

        st.download_button(
            "Download timeline data (CSV)",
            timeline.to_csv(index=False).encode("utf-8"),
            "timeline.csv", "text/csv"
        )
