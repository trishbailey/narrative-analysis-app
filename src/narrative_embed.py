# src/narrative_embed.py
# Utilities for sentence embeddings with SentenceTransformers

from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_sbert(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """
    Load a lightweight, high-quality embedding model.
    Streamlit will cache this, so it loads once.
    """
    return SentenceTransformer(model_name)

def concat_title_snippet(df: pd.DataFrame) -> List[str]:
    """
    Combine Title + Snippet into a single text string per row.
    Downstream clustering expects this.
    """
    return (df["Title"].fillna("") + ". " + df["Snippet"].fillna("")).tolist()

def embed_texts(model: SentenceTransformer, texts: List[str], show_progress: bool = True) -> np.ndarray:
    """
    Generate sentence embeddings (shape: [n_rows, dim]).
    """
    emb = model.encode(texts, show_progress_bar=show_progress)
    return np.asarray(emb)
