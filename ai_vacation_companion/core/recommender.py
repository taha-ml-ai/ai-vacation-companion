from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math

@dataclass
class Preference:
    budget: str              # low/medium/high
    climate: str             # warm/cold/mild
    activities: List[str]    # tags
    duration_days: Optional[int] = None
    month: Optional[str] = None

def normalize_tags(text: str) -> List[str]:
    return [t.strip().lower() for t in text.split(",") if t.strip()]

def jaccard(a: List[str], b: List[str]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 0.0
    return len(A & B) / len(A | B)

def rule_score(pref: Preference, pkg: Dict[str, Any], dest: Dict[str, Any]) -> float:
    score = 0.0
    # budget
    if pkg.get("budget") == pref.budget:
        score += 2.0
    # climate
    if dest.get("climate") == pref.climate:
        score += 2.0
    # activities overlap
    wanted = set(pref.activities)
    pkg_tags = normalize_tags(pkg.get("activities") or dest.get("activities") or "")
    overlap = len(wanted & set(pkg_tags))
    score += min(overlap, 3) * 1.0
    # duration closeness (penalty)
    if pref.duration_days and pkg.get("nights"):
        diff = abs(int(pkg["nights"]) - int(pref.duration_days))
        score += max(0.0, 1.5 - 0.3*diff)
    # optional: month-season heuristic (very light)
    # (left simple; could map months to climates later)
    return score

class SemanticHelper:
    def __init__(self):
        self.model = None
        try:
            from sentence_transformers import SentenceTransformer, util
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.util = util
        except Exception:
            self.model = None
            self.util = None

    def available(self) -> bool:
        return self.model is not None

    def rank(self, query: str, items: List[Tuple[str, str]], top_k: int = 10) -> List[int]:
        """Return indices sorted by semantic similarity.

        items: list of (name, description)

        """
        if not self.model:
            return list(range(len(items)))
        descs = [d for _, d in items]
        q_emb = self.model.encode([query], convert_to_tensor=True)
        d_emb = self.model.encode(descs, convert_to_tensor=True)
        scores = self.util.cos_sim(q_emb, d_emb)[0]
        # argsort descending
        idx = scores.argsort(descending=True)
        return [int(i) for i in idx[:top_k]]

def recommend(preference: Preference, destinations: List[Dict[str, Any]], packages: List[Dict[str, Any]], top_k: int = 5) -> List[Dict[str, Any]]:
    # Build semantic query text
    query_text = f"budget {preference.budget}; climate {preference.climate}; activities {', '.join(preference.activities)}; duration {preference.duration_days or ''}; month {preference.month or ''}"
    sem = SemanticHelper()

    # Create item list for semantic ranking using combined dest+pkg text
    items = []
    pairs = []  # map index -> (pkg, dest)
    dest_by_id = {d.get("id"): d for d in destinations}
    for pkg in packages:
        dest = dest_by_id.get(pkg.get("destination_id"))
        if not dest:
            continue
        desc = " ".join([
            dest.get("name",""), dest.get("country",""), dest.get("climate",""),
            dest.get("activities",""), dest.get("description",""),
            pkg.get("name",""), pkg.get("activities",""), pkg.get("highlights","")
        ])
        items.append((pkg.get("name",""), desc))
        pairs.append((pkg, dest))

    # Semantic pre-ranking (or identity if not available)
    ranked_idx = sem.rank(query_text, items, top_k=len(items))

    # Apply rule-based score to top-N semantic (or all if no model)
    finalists = ranked_idx[: min(50, len(ranked_idx))]
    scored: List[Tuple[float, Dict[str, Any], Dict[str, Any]]] = []
    for i in finalists:
        pkg, dest = pairs[i]
        s = rule_score(preference, pkg, dest)
        scored.append((s, pkg, dest))
    scored.sort(key=lambda t: (-t[0], t[1].get("price") or 0.0))

    results = []
    for s, pkg, dest in scored[:top_k]:
        results.append({
            "score": round(s, 2),
            "package": pkg,
            "destination": dest
        })
    return results
