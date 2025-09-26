# src/narrative_map.py
# UMAP helpers for 2D "semantic map" coordinates and Plotly-friendly hover fields.

from __future__ import annotations
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from umap import UMAP

def compute_umap(
    embeddings: np.ndarray,
    n_neighbors: int = 15,
    min_dist: float = 0.10,
    metric: str = "cosine",
    random_state: int = 42
) -> np.ndarray:
    """
    Reduce embeddings to 2D with UMAP. Returns coords of shape [n_rows, 2].
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    reducer = UMAP(
        n_components=2,
        metric=metric,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    coords = reducer.fit_transform(embeddings)
    return coords

def attach_coords(
    df: pd.DataFrame,
    coords: np.ndarray,
    x_col: str = "x",
    y_col: str = "y"
) -> pd.DataFrame:
    """
    Return a copy of df with 2D coordinates attached as columns x_col, y_col.
    """
    out = df.copy()
    out[x_col] = coords[:, 0]
    out[y_col] = coords[:, 1]
    return out

def build_hover_html(
    df: pd.DataFrame,
    title_col: str = "Title",
    snippet_col: str = "Snippet",
    date_col: str = "Date",
    url_col: str = "URL",
    max_snippet: int = 160
) -> pd.Series:
    """
    Build a Plotly-friendly hover HTML string for each row:
    Title, Date (if present), URL (if present), short Snippet.
    """
    def shorten(s: str, n: int) -> str:
        s = "" if s is None else str(s)
        return (s[:n] + "…") if len(s) > n else s

    has_date = date_col in df.columns
    has_url  = url_col in df.columns

    pieces = []
    for _, row in df.iterrows():
        title = str(row.get(title_col, ""))
        snippet = shorten(row.get(snippet_col, ""), max_snippet)

        parts = [f"Title: {title}"]
        if has_date and pd.notna(row.get(date_col)):
            parts.append(f"Date: {row.get(date_col)}")
        if has_url and pd.notna(row.get(url_col)):
            parts.append(f"URL: {row.get(url_col)}")
        parts.append(f"Snippet: {snippet}")

        pieces.append("<br>".join(parts))
    return pd.Series(pieces, index=df.index, name="_hover")
