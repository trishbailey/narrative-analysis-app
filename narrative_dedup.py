# src/narrative_dedup.py
from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Tuple
from sklearn.preprocessing import normalize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

def find_near_duplicates(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    title_col: str = "Title",
    snippet_col: str = "Snippet",
    sim_threshold: float = 0.90,
    min_chars: int = 40,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Detect near-duplicate rows using cosine similarity on embeddings.
    Returns: (pairs_df, dup_groups_df, dedup_df)
    - pairs_df: pairwise links above threshold
    - dup_groups_df: rows grouped by duplicate group id (>1 items)
    - dedup_df: dataframe keeping one representative per group
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)

    combo = (df[title_col].fillna("") + " " + df[snippet_col].fillna("")).astype(str)
    valid_mask = combo.str.len() >= min_chars
    idx = np.where(valid_mask.values)[0]
    if idx.size == 0:
        return (pd.DataFrame(columns=["i","j","similarity"]),
                pd.DataFrame(columns=["DupGroupID","Index",title_col,"URL","Cluster"]),
                df.copy())

    E = normalize(embeddings, norm="l2", axis=1)
    E_sub = E[idx]
    sim = E_sub @ E_sub.T
    np.fill_diagonal(sim, 0.0)

    # collect i<j above threshold
    pairs_i, pairs_j, pairs_s = [], [], []
    n = sim.shape[0]
    for i in range(n):
        row = sim[i, i+1:]
        js = np.where(row >= sim_threshold)[0]
        if js.size:
            for off in js:
                j = i + 1 + off
                pairs_i.append(i); pairs_j.append(j); pairs_s.append(float(row[off]))

    pairs_df = pd.DataFrame({"i_sub": pairs_i, "j_sub": pairs_j, "similarity": pairs_s})
    if pairs_df.empty:
        # nothing to dedup
        return (pd.DataFrame(columns=["i","j","similarity"]),
                pd.DataFrame(columns=["DupGroupID","Index",title_col,"URL","Cluster"]),
                df.copy())

    # map to original indices
    pairs_df["i"] = idx[pairs_df["i_sub"]].astype(int)
    pairs_df["j"] = idx[pairs_df["j_sub"]].astype(int)
    pairs_df.drop(columns=["i_sub","j_sub"], inplace=True)
    pairs_df = pairs_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    # build graph on filtered nodes, find connected components
    inv_map = {orig: pos for pos, orig in enumerate(idx)}
    rows = np.r_[pairs_df["i"].values, pairs_df["j"].values]
    cols = np.r_[pairs_df["j"].values, pairs_df["i"].values]
    rows_rel = np.array([inv_map[r] for r in rows], dtype=int)
    cols_rel = np.array([inv_map[c] for c in cols], dtype=int)
    data = np.ones(rows_rel.shape[0], dtype=np.int8)
    A = csr_matrix((data, (rows_rel, cols_rel)), shape=(idx.size, idx.size))
    n_comp, labels_rel = connected_components(A, directed=False)

    # groups with >1 members
    dup_groups = []
    for gid_rel in np.unique(labels_rel):
        members_rel = np.where(labels_rel == gid_rel)[0]
        if members_rel.size <= 1:
            continue
        members_orig = idx[members_rel]
        for orig in members_orig:
            dup_groups.append({
                "DupGroupID": int(gid_rel),
                "Index": int(orig),
                title_col: df.loc[orig, title_col],
                "URL": df.loc[orig, "URL"] if "URL" in df.columns else "",
                "Cluster": df.loc[orig, "Cluster"] if "Cluster" in df.columns else None
            })
    dup_groups_df = pd.DataFrame(dup_groups).sort_values(["DupGroupID","Index"]).reset_index(drop=True)

    # pick representative: longest text
    reps = []
    for gid_rel in np.unique(labels_rel):
        members_rel = np.where(labels_rel == gid_rel)[0]
        if members_rel.size <= 1:
            continue
        members_orig = idx[members_rel]
        best = max(members_orig, key=lambda k: len(combo.iloc[k]))
        reps.append(best)
    reps = set(reps)

    # build deduped df: keep all non-dups + representatives
    to_drop = set()
    for gid_rel in np.unique(labels_rel):
        members_rel = np.where(labels_rel == gid_rel)[0]
        if members_rel.size <= 1:
            continue
        members_orig = idx[members_rel]
        for m in members_orig:
            if m not in reps:
                to_drop.add(int(m))

    dedup_df = df.drop(index=list(to_drop)).copy().reset_index(drop=True)
    return pairs_df, dup_groups_df, dedup_df

def dedup_summary(df_before: pd.DataFrame, df_after: pd.DataFrame, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Return per-cluster Before/After/Removed/%Removed summary."""
    b = df_before[cluster_col].value_counts().sort_index()
    a = df_after[cluster_col].value_counts().sort_index()
    out = (pd.DataFrame({"Before": b}).join(pd.DataFrame({"After": a}), how="outer")
           .fillna(0).astype(int).reset_index().rename(columns={"index": cluster_col}))
    out["Removed"] = out["Before"] - out["After"]
    out["Pct Removed"] = (out["Removed"] / out["Before"].replace(0, np.nan) * 100).round(2)
    return out
