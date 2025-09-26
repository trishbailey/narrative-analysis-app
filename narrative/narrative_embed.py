# narrative/narrative_embed.py
from __future__ import annotations
from typing import List
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

def load_sbert(model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformer:
    """Load a light, high-quality sentence embedding model."""
    return SentenceTransformer(model_name)

def concat_title_snippet(df: pd.DataFrame) -> List[str]:
    """Combine Title + Snippet into one text string per row."""
    return (df["Title"].fillna("") + ". " + df["Snippet"].fillna("")).tolist()

def embed_texts(model: SentenceTransformer, texts: List[str], show_progress: bool = True) -> np.ndarray:
    """Generate sentence embeddings as a numpy array [n_rows, dim]."""
    emb = model.encode(texts, show_progress_bar=show_progress)
    return np.asarray(emb)
