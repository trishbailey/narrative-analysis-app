# src/narrative_kwic.py
# Keyword-in-Context (KWIC) helpers: search, counts by cluster, and co-occurrence.
from __future__ import annotations
from typing import Optional, Union, Iterable
import re
import pandas as pd
from collections import Counter

# Reuse normalization & stopwords from your TF-IDF module
from src.narrative_terms import ensure_text_column, build_stopwords

def kwic_search(
    df: pd.DataFrame,
    pattern: str,
    window: int = 50,
    case: bool = False,
    clusters: Optional[Union[int, Iterable[int]]] = None,
    cluster_col: str = "Cluster",
    text_col: str = "_text_norm",
    title_col: str = "Title",
    date_col: str = "Date",
    url_col: str = "URL",
    max_rows: int = 2000,
) -> pd.DataFrame:
    """
    Run a KWIC search on df[text_col] using regex `pattern`.
    Returns a DataFrame with columns:
      Cluster, Date (if present), Title, URL (if present), Left, Match, Right

    Args:
        df: DataFrame with at least Title/Snippet and cluster assignments.
        pattern: regex pattern (e.g., r"(lawsuit|settlement|class action)")
        window: number of characters on each side to include.
        case: case-sensitive if True; case-insensitive if False.
        clusters: None for all, or int, or iterable of ints to filter by cluster.
        cluster_col: name of cluster column.
        text_col: normalized text column. Created if missing via ensure_text_column.
        max_rows: cap the number of matches to avoid huge tables.

    Note:
        This function does not modify df.
    """
    # Ensure normalized text exists
    df = ensure_text_column(df, out_col=text_col)

    # Optional cluster filtering
    if clusters is None:
        sub = df
    else:
        if isinstance(clusters, int):
            sub = df[df[cluster_col] == clusters]
        else:
            sub = df[df[cluster_col].isin(list(clusters))]

    flags = 0 if case else re.IGNORECASE
    pat = re.compile(pattern, flags)

    rows = []
    count = 0
    has_date = date_col in sub.columns
    has_url = url_col in sub.columns

    for _, row in sub.iterrows():
        text = str(row[text_col])
        for m in pat.finditer(text):
            start, end = m.start(), m.end()
            left = text[max(0, start - window): start]
            match = text[start:end]
            right = text[end: min(len(text), end + window)]

            out = {
                "Cluster": row.get(cluster_col),
                "Title": row.get(title_col),
                "Left": left,
                "Match": match,
                "Right": right,
            }
            if has_date:
                out["Date"] = row.get(date_col)
            if has_url:
                out["URL"] = row.get(url_col)

            rows.append(out)
            count += 1
            if count >= max_rows:
                break
        if count >= max_rows:
            break

    return pd.DataFrame(rows)

def kwic_counts_by_cluster(kwic_df: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """
    Return a table of KWIC match counts per cluster.
    """
    if kwic_df.empty:
        return pd.DataFrame(columns=[cluster_col, "KWIC_Matches"])
    counts = (kwic_df[cluster_col].value_counts()
              .sort_index()
              .rename("KWIC_Matches")
              .reset_index()
              .rename(columns={"index": cluster_col}))
    return counts

def _tokenize_clean(s: str, stops: set[str]) -> list[str]:
    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", str(s).lower())
    return [t for t in tokens if len(t) >= 3 and t not in stops]

def kwic_cooccurrence(
    kwic_df: pd.DataFrame,
    top_n: int = 20,
    cluster_col: str = "Cluster",
) -> pd.DataFrame:
    """
    Compute co-occurring terms around KWIC matches (Left/Right windows), per cluster.
    Excludes the literal matched tokens from the context.
    Returns columns: Cluster, Term, Count, KWIC_Matches_in_Cluster
    """
    if kwic_df.empty:
        return pd.DataFrame(columns=[cluster_col, "Term", "Count", "KWIC_Matches_in_Cluster"])

    stops = set(build_stopwords())

    # Build context tokens for each match
    ctx_rows = []
    for _, r in kwic_df.iterrows():
        left_tokens  = _tokenize_clean(r.get("Left", ""), stops)
        right_tokens = _tokenize_clean(r.get("Right", ""), stops)
        match_tokens = set(_tokenize_clean(r.get("Match", ""), stops))
        ctx = [t for t in (left_tokens + right_tokens) if t not in match_tokens]
        ctx_rows.append((r.get(cluster_col), ctx))

    # Aggregate per cluster
    rows = []
    df_tmp = pd.DataFrame(ctx_rows, columns=[cluster_col, "_ctx"])
    for cid, grp in df_tmp.groupby(cluster_col):
        all_tokens: list[str] = []
        for toks in grp["_ctx"]:
            all_tokens.extend(toks)
        counter = Counter(all_tokens)
        for term, count in counter.most_common(top_n):
            rows.append({
                cluster_col: cid,
                "Term": term,
                "Count": count,
                "KWIC_Matches_in_Cluster": len(grp)
            })

    return pd.DataFrame(rows)
