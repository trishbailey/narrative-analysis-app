# narrative/narrative_io.py
from __future__ import annotations
import re
from typing import Optional, Union
import pandas as pd

ALIASES = {
    "title":   ["title", "headline", "headlines"],
    "snippet": ["snippet", "summary", "description", "dek", "selftext", "selftext_html", "body", "text"],
    "date":    ["date", "published", "pubdate", "time", "created", "created_iso", "created_utc"],
    "url":     ["url", "link", "permalink"]
}

def _pick(colnames_norm: list[str], candidates: list[str]) -> Optional[str]:
    cols = set(colnames_norm)
    for cand in candidates:
        if cand in cols:
            return cand
    return None

def read_csv_auto(path_or_buf: Union[str, bytes]) -> pd.DataFrame:
    try:
        return pd.read_csv(path_or_buf)
    except UnicodeDecodeError:
        return pd.read_csv(path_or_buf, encoding="latin-1")

def normalize_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    norm_to_orig = {}
    cols_norm = []
    for c in df_raw.columns:
        k = re.sub(r"\s+", "", c.strip().lower())
        norm_to_orig[k] = c
        cols_norm.append(k)

    title_col   = _pick(cols_norm, ALIASES["title"])
    snippet_col = _pick(cols_norm, ALIASES["snippet"])
    date_col    = _pick(cols_norm, ALIASES["date"])
    url_col     = _pick(cols_norm, ALIASES["url"])

    missing = []
    if title_col is None:   missing.append("Title")
    if snippet_col is None: missing.append("Snippet")
    if missing:
        raise ValueError(
            f"Required columns missing: {missing}. Found: {list(df_raw.columns)}. "
            f"Expected one of {ALIASES['title']} for Title and {ALIASES['snippet']} for Snippet."
        )

    keep = {
        "Title":   df_raw[norm_to_orig[title_col]].astype(str).str.strip(),
        "Snippet": df_raw[norm_to_orig[snippet_col]].astype(str).str.strip(),
    }
    if date_col is not None:
        keep["Date"] = pd.to_datetime(df_raw[norm_to_orig[date_col]], errors="coerce")
    if url_col is not None:
        keep["URL"] = df_raw[norm_to_orig[url_col]].astype(str).str.strip()

    df = pd.DataFrame(keep)
    df = df[(df["Title"].str.len() > 0) | (df["Snippet"].str.len() > 0)].copy()
    df.drop_duplicates(subset=["Title", "Snippet"], inplace=True)
    return df.reset_index(drop=True)

def load_and_normalize(path_or_buf: Union[str, bytes]) -> pd.DataFrame:
    return normalize_to_canonical(read_csv_auto(path_or_buf))
