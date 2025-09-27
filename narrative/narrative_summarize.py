# narrative/narrative_summarize.py
# One-line idea summaries:
# 1) Try spaCy (preinstalled model) to extract consensus Subject–Verb–Object across central posts.
# 2) If spaCy isn't available, fall back to a cleaned first-sentence heuristic.

from __future__ import annotations
from typing import Iterable, List, Tuple
import re
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def _try_load_spacy():
    try:
        import spacy
        import en_core_web_sm
        nlp = en_core_web_sm.load()
        # keep only tagger+parser+ner minimal: small model is already lightweight
        return nlp
    except Exception:
        return None

def _first_sentence(text: str, min_len: int = 25, max_len: int = 180) -> str:
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    parts = re.split(r'(?<=[\.!?])\s+', text.strip())
    for s in parts:
        s = s.strip()
        if len(s) >= min_len:
            return s[:max_len].rstrip(".") + "."
    return text[:max_len].rstrip(".") + "." if text else "No clear summary."

def _best_one_liner(title: str, snippet: str) -> str:
    title = (title or "").strip()
    snippet = (snippet or "").strip()
    if len(title) >= 30:
        return title.rstrip(".") + "."
    if snippet:
        return _first_sentence(snippet)
    return (title or "No clear summary").rstrip(".") + "."

def _svo(doc) -> Tuple[str,str,str]:
    """Extract simple S–V–O (best-effort)."""
    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if not root:
        return "","", ""
    s = next((" ".join(w.text for w in c.subtree if not w.is_punct).strip()
              for c in root.children if c.dep_ in ("nsubj","nsubjpass")), "")
    o = next((" ".join(w.text for w in c.subtree if not w.is_punct).strip()
              for c in root.children if c.dep_ in ("dobj","attr","pobj","oprd")), "")
    v = root.lemma_ or root.text
    return s, v, o

def _consensus_svo(nlp, texts: List[str]) -> str:
    """Aggregate SVO across texts and verbalize a single sentence."""
    from collections import Counter
    Ss, Vs, Os = Counter(), Counter(), Counter()
    for t in texts:
        doc = nlp(t[:400])
        s, v, o = _svo(doc)
        if s: Ss[s.lower()] += 1
        if v: Vs[v.lower()] += 1
        if o: Os[o.lower()] += 1
    if not (Ss or Vs or Os):
        return ""
    s = Ss.most_common(1)[0][0] if Ss else ""
    v = Vs.most_common(1)[0][0] if Vs else ""
    o = Os.most_common(1)[0][0] if Os else ""
    parts = [p for p in [s, v, o] if p]
    sent = " ".join(parts).strip().capitalize()
    return (sent.rstrip(".") + ".") if sent else ""

def summarize_narratives(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    label_col: str = "Label",
    text_cols: Iterable[str] = ("Title","Snippet"),
    topk_central: int = 5,
) -> pd.DataFrame:
    """
    For each idea (Label), pick top-k central posts, synthesize an SVO consensus one-liner.
    Fallback: best first-sentence/title heuristic.
    Returns: Idea | Narrative | Posts | Share_% | Examples
    """
    if label_col not in df.columns:
        df = df.copy()
        df[label_col] = df["Cluster"].astype(str)

    total = max(1, len(df))
    nlp = _try_load_spacy()  # can be None

    rows: List[dict] = []
    for idea, sub in df.groupby(label_col):
        idx = sub.index.to_numpy()
        if idx.size == 0:
            continue
        embs = embeddings[idx]
        centroid = embs.mean(axis=0, keepdims=True)
        sims = cosine_similarity(centroid, embs).ravel()
        order = sims.argsort()[::-1][: min(topk_central, len(sims))]
        tops = sub.iloc[order]

        # Build candidate raw strings for NLP (title + first sentence of snippet)
        raw_texts, one_liners = [], []
        for _, r in tops.iterrows():
            title = str(r.get("Title","") or "")
            snippet = str(r.get("Snippet","") or "")
            one_liners.append(_best_one_liner(title, snippet))
            raw = title
            if snippet:
                raw += " " + _first_sentence(snippet, min_len=15, max_len=120)
            raw_texts.append(raw.strip())

        # Try SVO consensus; fallback to the most common one-liner
        if nlp:
            cons = _consensus_svo(nlp, raw_texts)
        else:
            cons = ""

        if cons:
            narrative = cons
        else:
            from collections import Counter
            lowers = [x.lower() for x in one_liners if x]
            if lowers:
                best_lower, _ = Counter(lowers).most_common(1)[0]
                narrative = min((x for x in one_liners if x.lower()==best_lower), key=len)
            else:
                narrative = "No clear summary."

        examples = " | ".join(tops["Title"].dropna().astype(str).head(3).tolist())
        posts = len(sub)
        share = round(100.0 * posts / total, 2)

        rows.append({
            "Idea": str(idea),
            "Narrative": narrative,
            "Posts": posts,
            "Share_%": share,
            "Examples": examples
        })

    out = pd.DataFrame(rows).sort_values(["Posts","Idea"], ascending=[False, True]).reset_index(drop=True)
    return out
