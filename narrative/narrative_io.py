# narrative/narrative_io.py
# Normalize any CSV (incl. Meltwater X exports) to the canonical schema:
# Title (req), Snippet (req), Date (opt), URL (opt)

from __future__ import annotations
import re
from typing import Optional, Union
import pandas as pd

# Canonical alias lists
ALIASES = {
    # Title candidates, in priority order (we'll also apply MW-specific logic below)
    "title":   ["title", "headline", "headlines", "inputname", "keywords"],

    # Snippet candidates
    "snippet": ["snippet", "summary", "description", "dek",
                "selftext", "selftext_html", "body", "text",
                "openingtext", "hitsentence"],  # Meltwater X

    # Date candidates
    "date":    ["date", "published", "pubdate", "time", "created", "created_iso", "created_utc",
                "alternatedateformat"],  # MW: Alternate Date Format

    # URL candidates
    "url":     ["url", "link", "permalink", "parenturl"]  # MW: Parent URL
}

def _norm_name(c: str) -> str:
    return re.sub(r"\s+|_", "", c.strip().lower())

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

def _first_nonempty(df: pd.DataFrame, col_keys: list[str], norm2orig: dict) -> pd.Series:
    """Return the first non-empty string across candidate columns."""
    out = pd.Series([""] * len(df), dtype=object)
    for k in col_keys:
        if k in norm2orig:
            s = df[norm2orig[k]].fillna("").astype(str).str.strip()
            out = out.mask(out.astype(bool), out)  # keep existing non-empty
            out = out.mask(~out.astype(bool), s)   # fill only where empty
    return out

def _parse_meltwater_datetime(df: pd.DataFrame, norm2orig: dict) -> pd.Series:
    """
    Parse Meltwater-like date fields.
    Tries 'Date' (e.g., 15-Sep-2025 02:55PM). If fails, tries 'Alternate Date Format' + 'Time'.
    """
    def _coerce(s: pd.Series) -> pd.Series:
        # Insert a space before AM/PM if missing (e.g., '02:55PM' -> '02:55 PM')
        s = s.astype(str).str.replace(r'(?i)(am|pm)$', r' \1', regex=True)
        out = pd.to_datetime(s, errors="coerce", dayfirst=True)  # handles 15-Sep-2025 etc.
        return out

    date_col = norm2orig.get("date")
    alt_col  = norm2orig.get("alternatedateformat")
    time_col = norm2orig.get("time")

    # 1) Try main Date column first
    if date_col:
        d = _coerce(df[date_col])
    else:
        d = pd.Series(pd.NaT, index=df.index)

    # 2) Where NaT, try Alternate Date Format + Time
    need = d.isna()
    if need.any() and (alt_col or time_col):
        alt = df[norm2orig.get("alternatedateformat", "")].astype(str).str.strip() if alt_col else ""
        tim = df[norm2orig.get("time", "")].astype(str).str.strip() if time_col else ""
        combo = (alt + " " + tim).str.strip()
        d2 = _coerce(combo)
        d = d.fillna(d2)

    return d

def normalize_to_canonical(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize a raw DataFrame to the canonical columns:
    Title, Snippet, (optional) Date, URL.
    Handles Meltwater X columns like Opening Text / Hit Sentence / Parent URL / Alternate Date Format / Time.
    """
    # Map normalized -> original names
    norm2orig: dict[str, str] = {}
    cols_norm: list[str] = []
    for c in df_raw.columns:
        k = _norm_name(c)
        norm2orig[k] = c
        cols_norm.append(k)

    # Identify common fields
    title_key   = _pick(cols_norm, ALIASES["title"])
    snippet_key = _pick(cols_norm, ALIASES["snippet"])
    date_key    = _pick(cols_norm, ALIASES["date"])
    url_key     = _pick(cols_norm, ALIASES["url"])

    # --- MELTWATER-SPECIFIC TITLE/SNIPPET LOGIC ---
    # Prefer Headline if present; otherwise use Hit Sentence or Opening Text.
    # For snippet, prefer Opening Text; else Hit Sentence; else Headline.
    headline = norm2orig.get("headline")
    hitsent  = norm2orig.get("hitsentence")
    opentxt  = norm2orig.get("openingtext")
    inputname = norm2orig.get("inputname")
    keywords  = norm2orig.get("keywords")

    # Build Title
    if headline:
        title_series = df_raw[headline].fillna("").astype(str).str.strip()
    elif hitsent:
        title_series = df_raw[hitsent].fillna("").astype(str).str.strip()
    elif opentxt:
        title_series = df_raw[opentxt].fillna("").astype(str).str.strip()
    elif inputname:
        title_series = df_raw[inputname].fillna("").astype(str).str.strip()
    elif keywords:
        title_series = df_raw[keywords].fillna("").astype(str).str.strip()
    else:
        # Fall back to generic title aliases (if any existed)
        title_series = _first_nonempty(df_raw, [k for k in ALIASES["title"] if k in norm2orig], norm2orig)

    # Build Snippet
    if opentxt:
        snippet_series = df_raw[opentxt].fillna("").astype(str).str.strip()
    elif hitsent:
        snippet_series = df_raw[hitsent].fillna("").astype(str).str.strip()
    elif headline:
        snippet_series = df_raw[headline].fillna("").astype(str).str.strip()
    else:
        snippet_series = _first_nonempty(df_raw, [k for k in ALIASES["snippet"] if k in norm2orig], norm2orig)

    # Build Date
    if date_key:
        date_series = _parse_meltwater_datetime(df_raw, norm2orig)
    else:
        # even if 'date' wasn't detected, try MW combo explicitly
        date_series = _parse_meltwater_datetime(df_raw, norm2orig)

    # Build URL (prefer Parent URL if present)
    if "parenturl" in norm2orig:
        url_series = df_raw[norm2orig["parenturl"]].fillna("").astype(str).str.strip()
    elif url_key:
        url_series = df_raw[norm2orig[url_key]].fillna("").astype(str).str.strip()
    else:
        url_series = pd.Series([""] * len(df_raw), dtype=object)

    # Assemble canonical
    df = pd.DataFrame({
        "Title":   title_series,
        "Snippet": snippet_series,
        "Date":    date_series,
        "URL":     url_series
    })

    # Clean rows: require either Title or Snippet
    df = df[(df["Title"].str.len() > 0) | (df["Snippet"].str.len() > 0)].copy()
    df.drop_duplicates(subset=["Title", "Snippet"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def load_and_normalize(path_or_buf: Union[str, bytes]) -> pd.DataFrame:
    return normalize_to_canonical(read_csv_auto(path_or_buf))
