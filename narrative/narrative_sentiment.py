# src/narrative_sentiment.py
# VADER sentiment: add per-row sentiment and summarize by cluster.

from __future__ import annotations
import pandas as pd
import re

# Reuse text normalization from your terms module
from narrative.narrative_terms import ensure_text_column

# NLTK VADER
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

def _ensure_vader():
    """Make sure the VADER lexicon is available."""
    try:
        # This will raise a LookupError if missing
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon", quiet=True)

def add_vader_sentiment(
    df: pd.DataFrame,
    text_col: str = "_text_norm",
    out_col: str = "Sentiment",
    title_col: str = "Title",
    snippet_col: str = "Snippet"
) -> pd.DataFrame:
    """
    Ensure normalized text, compute VADER compound sentiment (-1..+1) per row,
    and return a copy with a new Sentiment column.
    """
    _ensure_vader()
    df2 = ensure_text_column(df, title_col=title_col, snippet_col=snippet_col, out_col=text_col).copy()
    sia = SentimentIntensityAnalyzer()
    df2[out_col] = df2[text_col].map(lambda t: sia.polarity_scores(t)["compound"])
    return df2

def sentiment_by_cluster(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    sent_col: str = "Sentiment"
) -> pd.DataFrame:
    """
    Aggregate sentiment stats per cluster.
    Returns columns: Cluster, count, mean, median, min, max
    """
    if sent_col not in df.columns:
        raise ValueError(f"'{sent_col}' column not found. Run add_vader_sentiment(...) first.")
    summ = (
        df.groupby(cluster_col)[sent_col]
          .agg(["count","mean","median","min","max"])
          .reset_index()
    )
    return summ
