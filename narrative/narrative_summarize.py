# narrative/narrative_summarize.py
# One-line idea summaries WITHOUT spaCy:
# - Pick top-k central posts (by cosine to centroid).
# - Prefer a clean first sentence from the snippet; fallback to a usable title line.
# - De-duplicate candidate sentences and choose the most representative one.

from __future__ import annotations
from typing import Iterable, List
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def _first_sentence(text: str, min_len: int = 25, max_len: int = 180) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.strip()
    if not text:
        return ""
    # Split on sentence boundaries ., !, ?
    parts = re.split(r'(?<=[\.!?])\s+', text)
    for s in parts:
        s = s.strip()
        if len(s) >= min_len:
            return s[:max_len].rstrip(".") + "."
    # fallback: any non-empty chunk
    for s in parts:
        s = s.strip()
        if s:
            return s[:max_len].rstrip(".") + "."
    return ""

def _clean_title_line(title: str, snippet: str) -> str:
    title = (title or "").strip()
    snippet = (snippet or "").strip()
    # Prefer snippet sentence if it looks like prose
    sent = _first_sentence(snippet)
    if len(sent) >= 25:
        return sent
    # Else try title if it's descriptive (not just a tag line)
    if len(title) >= 30:
        return title.rstrip(".") + "."
    # Last fallback: short sentence from snippet or title
    return (snippet or title or "No clear summary").strip()[:120].rstrip(".") + "."

def _dedupe_keep_order(cands: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in cands:
        key = re.sub(r"\s+", " ", c.strip().lower())
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out

def summarize_narratives(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str = "Label",
    text_cols: Iterable[str] = ("Title","Snippet"),
    topk_central: int = 5,
) -> pd.DataFrame:
    """
    For each idea (Label), select the top-k most central posts (by cosine to the idea centroid),
    build one-line candidates from snippet/title, de-duplicate, and choose the most representative.
    Returns columns: Idea | Narrative | Posts | Share_% | Examples
    """
    if label_col not in df.columns:
        df = df.copy()
        df[label_col] = df["Cluster"].astype(str)

    total = max(1, len(df))
    rows: List[dict] = []

    for idea, sub in df.groupby(label_col):
        idx = sub.index.to_numpy()
        if idx.size == 0:
            continue

        embs = embeddings[idx]
        centroid = embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, embs).ravel()
        order = sims.argsort()[::-1][: min(topk_central, len(sims))]
        tops = sub.iloc[order]

        # Build candidate one-liners (prefer snippet first-sentence)
        candidates: List[str] = []
        for _, r in tops.iterrows():
            one = _clean_title_line(str(r.get("Title","")), str(r.get("Snippet","")))
            if one:
                candidates.append(one)

        candidates = _dedupe_keep_order(candidates)
        if not candidates:
            candidates = ["No clear summary."]

        # Choose the candidate closest to the centroid text-wise:
        # score by average token overlap with the rest (simple, fast proxy)
        def score(c):
            toks = set(re.findall(r"[A-Za-z0-9]+", c.lower()))
            if not toks:
                return 0.0
            overlaps = []
            for d in candidates:
                if d is c:
                    continue
                toks2 = set(re.findall(r"[A-Za-z0-9]+", d.lower()))
                if toks2:
                    overlaps.append(len(toks & toks2) / max(1, len(toks | toks2)))
            return sum(overlaps) / max(1, len(overlaps))

        best = max(candidates, key=score)

        examples = " | ".join(tops["Title"].dropna().astype(str).head(3).tolist())
        posts = len(sub)
        share = round(100.0 * posts / total, 2)

        rows.append({
            "Idea": str(idea),
            "Narrative": best,
            "Posts": posts,
            "Share_%": share,
            "Examples": examples
        })

    out = pd.DataFrame(rows).sort_values(["Posts","Idea"], ascending=[False, True]).reset_index(drop=True)
    return out
