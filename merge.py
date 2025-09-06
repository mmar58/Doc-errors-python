from typing import List, Dict, Tuple, Optional
from models import Finding, SEV_TEXT_TO_INT, SEV_INT_TO_TEXT, sev_rank

def norm(s: str) -> str:
    return " ".join((s or "").strip().lower().split())

def finding_key(f: Finding) -> Tuple[str, int, int]:
    return (norm(f.phrase), f.page, sev_rank(f.severity))

def more_info_score(f: Finding) -> int:
    score = 0
    if f.start_char is not None and f.end_char is not None:
        score += 2
    if f.context:
        score += 1
    if f.suggestion:
        score += 1
    return score

def _sev_int(s: str) -> int:
    return SEV_TEXT_TO_INT.get((s or "High").lower(), 0)

def merge_findings(
    csv_findings: List[Finding],
    model_findings_all: List[List[Finding]],
    weight_by_phrase: Dict[str, int],
    llm_discovered: List[Finding] = None,
) -> List[Finding]:
    merged: Dict[Tuple[str, int], Finding] = {}  # (phrase_norm, page) -> best Finding

    def consider(f: Finding):
        key = (f.phrase.lower(), f.page or 1)
        if key not in merged:
            merged[key] = f
            return
        # Prefer higher severity (lower int), then prefer with suggestion, then earlier start_char
        prev = merged[key]
        if _sev_int(f.severity) < _sev_int(prev.severity):
            merged[key] = f
        elif _sev_int(f.severity) == _sev_int(prev.severity):
            if (f.suggestion and not prev.suggestion):
                merged[key] = f
            elif f.start_char is not None and (prev.start_char is None or f.start_char < prev.start_char):
                merged[key] = f

    for f in csv_findings:
        f.source = "CSV"
        f.phrase = f.phrase.lower()
        consider(f)

    for lst in model_findings_all or []:
        for f in lst:
            f.source = f.source or "LLM"
            f.phrase = (f.phrase or "").lower()
            consider(f)

    if llm_discovered:
        for f in llm_discovered:
            f.source = "LLM"
            f.phrase = (f.phrase or "").lower()
            consider(f)

    out = list(merged.values())
    # Sort: severity desc (High first), then weight (CSV phrases ranked), then page, then phrase
    def weight_of(f: Finding) -> int:
        return weight_by_phrase.get(f.phrase, 0)
    out.sort(key=lambda f: (
        _sev_int(f.severity),
        -weight_of(f),
        f.page or 0,
        f.phrase
    ))
    return out