# narrative/narrative_terms.py
# TF-IDF "top terms" per cluster with a strong stopword list and text normalization.

from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

# ---------------------------
# Text normalization & stops
# ---------------------------

def normalize_text(s: str) -> str:
    """Strip HTML, entities, URLs, punctuation; collapse whitespace."""
    if not isinstance(s, str):
        s = "" if s is None else str(s)
    s = re.sub(r"<[^>]+>", " ", s)          # HTML tags
    s = re.sub(r"&\w+;", " ", s)            # HTML entities (&nbsp; &quot; ...)
    s = re.sub(r"http\S+", " ", s)          # URLs
    s = re.sub(r"[^\w\s]", " ", s)          # punctuation -> space
    s = re.sub(r"\s+", " ", s).strip()
    return s

def build_stopwords() -> List[str]:
    """English stopwords + custom junk commonly found in social/news text."""
    custom = {
        "amp","nbsp","quot","http","https","www","com","org","net",
        "utm","ref","click","read","share","reply","retweet",
        "with","from","into","about","after","before","between","through",
        "could","would","should","might","also","still","even","like",
        "said","says","say","one","two","new","make","made","get","got",
        "today","yesterday","tomorrow","year","years","day","days",
        "edit","deleted","removed"
    }
    return list(ENGLISH_STOP_WORDS.union(custom))

# -----------------------------------------
# Build a normalized text column in a frame
# -----------------------------------------

def ensure_text_column(
    df: pd.DataFrame,
    title_col: str = "Title",
    snippet_col: str = "Snippet",
    out_col: str = "_text_norm"
) -> pd.DataFrame:
    """Create df[out_col] = normalized(Title + ' ' + Snippet). Returns df (modified copy)."""
    out = df.copy()
    if out_col not in out.columns:
        out[out_col] = (out[title_col].fillna("") + " " + out[snippet_col].fillna("")).map(normalize_text)
    return out

# -----------------------
# TF-IDF top terms logic
# -----------------------

def tfidf_top_terms(
    texts: List[str],
    n_terms: int = 12,
    stopwords: Optional[List[str]] = None,
    ngram_range: Tuple[int,int] = (1, 2),
    min_df: int = 2,
    max_df: float = 0.8
) -> List[str]:
    """
    Return top n_terms by mean TF-IDF for a list of documents.
    """
    if stopwords is None:
        stopwords = build_stopwords()
    vec = TfidfVectorizer(
        stop_words=stopwords,
        ngram_range=ngram_range,
        min_df=min_df,
        max_df=max_df
    )
    X = vec.fit_transform(texts)
    if X.shape[1] == 0:
        return []
    mean_scores = np.asarray(X.mean(axis=0)).ravel()
    terms = np.array(vec.get_feature_names_out())
    top_idx = mean_scores.argsort()[::-1][:n_terms]
    return terms[top_idx].tolist()

def cluster_top_terms(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    text_col: str = "_text_norm",
    n_terms: int = 12,
    min_df_small: int = 1
) -> Dict[int, List[str]]:
    """
    Compute top TF-IDF terms per cluster. For very small clusters, relax min_df to 1.
    Returns: {cluster_id: [term1, term2, ...]}
    """
    stops = build_stopwords()
    result: Dict[int, List[str]] = {}
    for cid, sub in df.groupby(cluster_col):
        docs = sub[text_col].tolist()
        if len(docs) == 0:
            result[int(cid)] = []
            continue
        min_df_val = min_df_small if len(docs) < 5 else 2
        terms = tfidf_top_terms(
            docs, n_terms=n_terms, stopwords=stops, ngram_range=(1,2),
            min_df=min_df_val, max_df=0.8
        )
        result[int(cid)] = terms
    return result

def cluster_terms_dataframe(
    df: pd.DataFrame,
    cluster_col: str = "Cluster",
    text_col: str = "_text_norm",
    n_terms: int = 12
) -> pd.DataFrame:
    """
    Return a tidy DataFrame with Cluster, TopTerms (comma-joined).
    """
    df_norm = ensure_text_column(df, out_col=text_col)
    tops = cluster_top_terms(df_norm, cluster_col=cluster_col, text_col=text_col, n_terms=n_terms)
    rows = [{"Cluster": cid, "TopTerms": ", ".join(terms)} for cid, terms in sorted(tops.items())]
    return pd.DataFrame(rows)
