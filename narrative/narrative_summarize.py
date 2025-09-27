# narrative/narrative_summarize.py
# One-line narrative summaries without spaCy: pick the most central posts per idea,
# then choose a clear, single-sentence summary from their titles/snippets.

from __future__ import annotations
from typing import Iterable, List
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def _first_sentence(text: str, min_len: int = 25, max_len: int = 180) -> str:
    """Return the first reasonably long sentence from text; fallback to trimmed text."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    # Split on sentence boundaries ., !, ?
    parts = re.split(r'(?<=[\.!?])\s+', text.strip())
    for s in parts:
        s = s.strip()
        if len(s) >= min_len:
            return s[:max_len].rstrip(".") + "."
    return text[:max_len].rstrip(".") + "." if text else ""

def _best_candidate(title: str, snippet: str) -> str:
    """Prefer a good title; otherwise take first sentence from snippet; ensure one sentence."""
    title = (title or "").strip()
    snippet = (snippet or "").strip()
    # If title is already a good one-liner, use it.
    if len(title) >= 30:
        return title.rstrip(".") + "."
    # Else try the first sentence from snippet.
    if snippet:
        return _first_sentence(snippet)
    # Fallback to title (even if short).
    return (title or "No clear summary").rstrip(".") + "."

def summarize_narratives(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str = "Label",
    text_cols: Iterable[str] = ("Title", "Snippet"),
    topk_central: int = 5,
) -> pd.DataFrame:
    """
    For each idea (Label), select the top-k most central posts (by cosine to the idea centroid),
    then choose a single-sentence summary from their titles/snippets.
    Returns columns: Idea | Narrative | Posts | Share_% | Examples
    """
    # Ensure label column exists
    if label_col not in df.columns:
        df = df.copy()
        df[label_col] = df["Cluster"].astype(str)

    total = max(1, len(df))
    results: List[dict] = []

    for idea, sub in df.groupby(label_col):
        idx = sub.index.to_numpy()
        if idx.size == 0:
            continue

        embs = embeddings[idx]
        # Centrality by similarity to centroid
        centroid = embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, embs).ravel()
        order = sims.argsort()[::-1]
        top_loc = order[: min(topk_central, len(order))]
        tops = sub.iloc[top_loc]

        # Build candidates (one sentence each)
        cand: List[str] = []
        for _, row in tops.iterrows():
            title = str(row.get("Title", "") or "")
            snippet = str(row.get("Snippet", "") or "")
            cand.append(_best_candidate(title, snippet))

        # Pick the most common (case-insensitive), tie-break by shortest
        if cand:
            from collections import Counter
            lowers = [c.lower() for c in cand]
            best_lower, _ = Counter(lowers).most_common(1)[0]
            best = min((c for c in cand if c.lower() == best_lower), key=len)
        else:
            best = "No clear summary."

        examples = " | ".join(tops["Title"].dropna().astype(str).head(3).tolist())
        posts = len(sub)
        share = round(100.0 * posts / total, 2)

        results.append({
            "Idea": str(idea),
            "Narrative": best,
            "Posts": posts,
            "Share_%": share,
            "Examples": examples
        })

    out = pd.DataFrame(results).sort_values(["Posts", "Idea"], ascending=[False, True]).reset_index(drop=True)
    return out
