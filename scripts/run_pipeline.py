# scripts/run_pipeline.py
"""
End-to-end pipeline (no Streamlit):
- Load & normalize CSV
- SBERT embeddings
- KMeans clustering
- TF-IDF top terms per cluster
- UMAP 2D coords
- (Optional) VADER sentiment
- Near-duplicate detection + summary
- Timeline counts
Writes outputs to ./out/
Usage:
  python -m scripts.run_pipeline --input path/to.csv --clusters 6
"""
from __future__ import annotations
import os, argparse
import pandas as pd
import numpy as np

from src.narrative_io import load_and_normalize
from src.narrative_embed import load_sbert, concat_title_snippet, embed_texts
from src.narrative_cluster import run_kmeans, attach_clusters
from src.narrative_terms import ensure_text_column, cluster_terms_dataframe
from src.narrative_map import compute_umap, attach_coords
from src.narrative_sentiment import add_vader_sentiment, sentiment_by_cluster
from src.narrative_dedup import find_near_duplicates, dedup_summary
from src.narrative_timeline import add_week, counts_by_week_cluster

def ensure_out(outdir: str) -> str:
    os.makedirs(outdir, exist_ok=True)
    return outdir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input CSV path")
    ap.add_argument("--outdir", default="out", help="Output directory (default: out)")
    ap.add_argument("--clusters", type=int, default=6, help="KMeans cluster count (default: 6)")
    ap.add_argument("--do_sentiment", action="store_true", help="Compute VADER sentiment")
    ap.add_argument("--dedup_threshold", type=float, default=0.90, help="Near-dup cosine sim threshold (default: 0.90)")
    args = ap.parse_args()

    outdir = ensure_out(args.outdir)

    # 1) Load & normalize
    print("[1/8] Loading & normalizing …")
    df = load_and_normalize(args.input)
    df.to_csv(os.path.join(outdir, "clean.csv"), index=False)

    # 2) Embeddings
    print("[2/8] Embedding …")
    model = load_sbert("all-MiniLM-L6-v2")
    texts = concat_title_snippet(df)
    embeddings = embed_texts(model, texts, show_progress=True)
    np.save(os.path.join(outdir, "embeddings.npy"), embeddings)

    # 3) KMeans
    print(f"[3/8] KMeans (k={args.clusters}) …")
    labels, _ = run_kmeans(embeddings, n_clusters=args.clusters)
    df = attach_clusters(df, labels)
    df.to_csv(os.path.join(outdir, "clustered.csv"), index=False)

    # 4) Top TF-IDF terms per cluster
    print("[4/8] Top terms per cluster …")
    df_norm = ensure_text_column(df)
    terms_df = cluster_terms_dataframe(df_norm, n_terms=12)
    terms_df.to_csv(os.path.join(outdir, "cluster_top_terms.csv"), index=False)

    # 5) UMAP coords
    print("[5/8] UMAP 2D projection …")
    coords = compute_umap(embeddings, n_neighbors=15, min_dist=0.10)
    df_map = attach_coords(df, coords)
    df_map.to_csv(os.path.join(outdir, "umap_coords.csv"), index=False)

    # 6) Optional sentiment
    if args.do_sentiment:
        print("[6/8] VADER sentiment …")
        df_sent = add_vader_sentiment(df)
        df_sent.to_csv(os.path.join(outdir, "clustered_with_sentiment.csv"), index=False)
        sent_tbl = sentiment_by_cluster(df_sent)
        sent_tbl.to_csv(os.path.join(outdir, "sentiment_by_cluster.csv"), index=False)

    # 7) Dedup + summary
    print("[7/8] Near-duplicate detection …")
    pairs_df, dup_groups_df, dedup_df = find_near_duplicates(
        embeddings, df, sim_threshold=args.dedup_threshold, min_chars=40
    )
    pairs_df.to_csv(os.path.join(outdir, "duplicate_pairs.csv"), index=False)
    dup_groups_df.to_csv(os.path.join(outdir, "duplicate_groups.csv"), index=False)
    dedup_df.to_csv(os.path.join(outdir, "dedup.csv"), index=False)
    summary = dedup_summary(df, dedup_df)
    summary.to_csv(os.path.join(outdir, "dedup_summary.csv"), index=False)

    # 8) Timeline counts (if dates exist)
    if "Date" in df.columns and df["Date"].notna().any():
        print("[8/8] Timeline …")
        df_w = add_week(df)
        timeline = counts_by_week_cluster(df_w)
        timeline.to_csv(os.path.join(outdir, "timeline_by_week_cluster.csv"), index=False)
    else:
        print("[8/8] Timeline skipped (no Date column).")

    print(f"\nDone. Outputs in: {outdir}")

if __name__ == "__main__":
    main()
