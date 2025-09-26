# src/narrative_cluster.py
# Simple KMeans helpers for narrative clustering

from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

def run_kmeans(embeddings: np.ndarray, n_clusters: int = 6,
               random_state: int = 42, n_init: int = 10) -> tuple[np.ndarray, KMeans]:
    """
    Fit KMeans on embedding vectors and return (labels, model).
    embeddings : shape [n_rows, dim]
    """
    if not isinstance(embeddings, np.ndarray):
        embeddings = np.asarray(embeddings)
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
    labels = km.fit_predict(embeddings)
    return labels, km

def attach_clusters(df: pd.DataFrame, labels: np.ndarray, colname: str = "Cluster") -> pd.DataFrame:
    """
    Return a copy of df with a new 'Cluster' column (or custom name).
    """
    out = df.copy()
    out[colname] = labels.astype(int)
    return out

def cluster_counts(df: pd.DataFrame, colname: str = "Cluster") -> pd.DataFrame:
    """
    Count rows per cluster, sorted by cluster id.
    """
    return (df[colname].value_counts()
            .sort_index()
            .rename("Count")
            .reset_index()
            .rename(columns={"index": colname}))
