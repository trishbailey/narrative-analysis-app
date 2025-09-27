# narrative/narrative_summarize.py
from __future__ import annotations
from typing import Iterable, Tuple, List
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import spacy

def _load_spacy(model: str = "en_core_web_sm"):
    try:
        return spacy.load(model)
    except OSError:
        import subprocess, sys
        subprocess.run([sys.executable, "-m", "spacy", "download", model], check=True)
        return spacy.load(model)

def _extract_svo(doc: "spacy.tokens.Doc") -> Tuple[str,str,str]:
    """Simple SVO extractor from a parsed Doc; falls back gracefully."""
    subj_txt, verb_lemma, obj_txt = "", "", ""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root:
        verb_lemma = root.lemma_
        # subjects
        subs = [c.subtree for c in root.children if c.dep_ in ("nsubj", "nsubjpass")]
        if subs:
            subj_txt = " ".join(w.text for w in list(subs[0]) if not w.is_punct).strip()
        # objects / complements
        objs = [c.subtree for c in root.children if c.dep_ in ("dobj","attr","pobj","oprd")]
        if objs:
            obj_txt = " ".join(w.text for w in list(objs[0]) if not w.is_punct).strip()
    return subj_txt, verb_lemma, obj_txt

def summarize_narratives(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str = "Label",
    text_cols: Iterable[str] = ("Title","Snippet"),
    topk_central: int = 5,
) -> pd.DataFrame:
    """
    For each idea (Label), choose the top-k most central posts (by centroid similarity),
    extract SVOs, and compose a single-sentence narrative. Returns rows:
      Idea | Narrative | Posts | Share_% | Examples
    """
    nlp = _load_spacy()
    results = []

    # Ensure label column exists
    if label_col not in df.columns:
        df = df.copy()
        df[label_col] = df["Cluster"].astype(str)

    total = max(1, len(df))

    for idea, sub in df.groupby(label_col):
        idx = sub.index.to_numpy()
        embs = embeddings[idx]
        if len(embs) == 0:
            continue

        # central items
        centroid = embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, embs).ravel()
        order = sims.argsort()[::-1]
        top_loc = order[: min(topk_central, len(order))]
        tops = sub.iloc[top_loc]

        # candidate sentences from SVO over central items
        candidates: List[str] = []
        for _, row in tops.iterrows():
            raw = " ".join(str(row.get(c,"")) for c in text_cols if pd.notna(row.get(c,""))).strip()
            raw = raw[:400]  # keep it short for parsing
            doc = nlp(raw)
            s, v, o = _extract_svo(doc)
            if s or o:
                sent = " ".join(t for t in (s, v, o) if t).strip().capitalize()
            else:
                # fallback to first reasonably long sentence or the title
                sent = next((s.text.strip() for s in doc.sents if len(s.text.strip()) > 25), (row.get("Title") or raw)[:120])
            candidates.append(sent)

        # pick the most common (case-insensitive); tie-break by shortest
        if candidates:
            from collections import Counter
            lowers = [c.lower() for c in candidates]
            best_lower, _ = Counter(lowers).most_common(1)[0]
            best = min((c for c in candidates if c.lower() == best_lower), key=len)
        else:
            best = "(no clear summary)"

        examples = " | ".join(tops["Title"].dropna().astype(str).head(3).tolist())
        posts = len(sub)
        share = round(100.0 * posts / total, 2)

        # light normalization for readability
        best = best.rstrip(".") + "."
        results.append({"Idea": str(idea), "Narrative": best, "Posts": posts, "Share_%": share, "Examples": examples})

    out = pd.DataFrame(results).sort_values(["Posts","Idea"], ascending=[False, True]).reset_index(drop=True)
    return out
