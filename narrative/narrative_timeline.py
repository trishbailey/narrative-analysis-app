# src/narrative_timeline.py
from __future__ import annotations
import pandas as pd

def add_week(df: pd.DataFrame, date_col: str = "Date", out_col: str = "Week") -> pd.DataFrame:
    """Return a copy with a weekly period start column."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    out[out_col] = out[date_col].dt.to_period("W").apply(lambda r: r.start_time)
    return out

def counts_by_week_cluster(df: pd.DataFrame, week_col: str = "Week", cluster_col: str = "Cluster") -> pd.DataFrame:
    """Count rows by week and cluster."""
    return (df.groupby([week_col, cluster_col]).size()
            .reset_index(name="Count")
            .sort_values([week_col, cluster_col]))
