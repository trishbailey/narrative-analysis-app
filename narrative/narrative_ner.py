# src/narrative_ner.py
from __future__ import annotations
import re
import pandas as pd
from typing import Iterable, Tuple
import spacy

from src.narrative_terms import ensure_text_column

def load_spacy(model: str = "en_core_web_sm") -> spacy.Language:
    """Load spaCy model; download if missing when run in a notebook/script."""
    try:
        return spacy.load(model)
    except OSError:
        # Deferred download to runtime (not at import)
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
        return spacy.load(model)

def extract_entities(
    df: pd.DataFrame,
    labels: Iterable[str] = ("PERSON","ORG","GPE"),
    text_col: str = "_text_norm",
    title_col: str = "Title",
    snippet_col: str = "Snippet",
    batch_size: int = 256
) -> pd.DataFrame:
    """Return rows: Cluster, Entity, Label."""
    nlp = load_spacy()
    nlp.disable_pipes(*[p for p in nlp.pipe_names if p != "ner"])
    df2 = ensure_text_column(df, title_col=title_col, snippet_col=snippet_col, out_col=text_col)
    rows = []
    for i, doc in enumerate(nlp.pipe(df2[text_col].tolist(), batch_size=batch_size)):
        cid = int(df2.iloc[i]["Cluster"])
        for ent in doc.ents:
            if ent.label_ in labels:
                t = ent.text.strip()
                if len(t) >= 2 and re.search(r"[A-Za-z]", t):
                    rows.append({"Cluster": cid, "Entity": t, "Label": ent.label_})
    return pd.DataFrame(rows)

def top_entities_by_cluster(ents_df: pd.DataFrame, label: str, top_k: int = 15) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (counts_df, top_df) for a single entity label (e.g., ORG)."""
    sub = ents_df[ents_df["Label"] == label]
    counts = (sub.groupby(["Cluster","Entity"]).size()
              .reset_index(name="Count")
              .sort_values(["Cluster","Count"], ascending=[True,False]))
    top = counts.groupby("Cluster").head(top_k).reset_index(drop=True)
    return counts, top
