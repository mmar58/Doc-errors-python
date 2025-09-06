from typing import List, Optional, Tuple, Dict
from pydantic import BaseModel

SEV_LOW = 2
SEV_MED = 1
SEV_HIGH = 0

SEV_TEXT_TO_INT = {
    "high": SEV_HIGH,
    "medium": SEV_MED,
    "low": SEV_LOW,
}
SEV_INT_TO_TEXT = {
    SEV_HIGH: "High",
    SEV_MED: "Medium",
    SEV_LOW: "Low",
}

def normalize_sev_text(s: str) -> str:
    if not s:
        return "High"
    s = s.strip().lower()
    if s in SEV_TEXT_TO_INT:
        return SEV_INT_TO_TEXT[SEV_TEXT_TO_INT[s]]
    return "High"

def parse_severity_to_int(value, default_int: int = SEV_MED) -> int:
    """
    Convert an incoming value to our internal severity int.
    - Accepts strings: "High" | "Medium" | "Low" (case-insensitive).
    - Accepts ints 0/1/2 and clamps to valid range.
    - Returns default_int if parsing fails.
    """
    if value is None:
        return default_int
    # int-like
    if isinstance(value, int):
        if value in (SEV_HIGH, SEV_MED, SEV_LOW):
            return value
        # clamp to valid range
        return max(min(int(value), SEV_LOW), SEV_HIGH)
    # string-like
    try:
        s = str(value).strip().lower()
        if s == "":
            return default_int
        if s.isdigit():
            iv = int(s)
            if iv in (SEV_HIGH, SEV_MED, SEV_LOW):
                return iv
            return max(min(iv, SEV_LOW), SEV_HIGH)
        return SEV_TEXT_TO_INT.get(s, default_int)
    except Exception:
        return default_int

def apply_match_severity(in_csv: bool, overlay_val: Optional[int]) -> int:
    # If overlay provided, use it. Else High for CSV phrases; Medium default otherwise.
    if overlay_val is not None:
        return overlay_val
    return SEV_HIGH if in_csv else SEV_MED

# MetaTuple: (csv_sev_int, feedback, tags_tuple, weight, enabled, source, version, example_context, locale, regex_flag)
MetaTuple = Tuple[int, Optional[str], Tuple[str, ...], int, bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[bool]]

class Finding(BaseModel):
    phrase: str
    severity: str
    suggestion: Optional[str] = None
    page: int
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    context: Optional[str] = None
    source: str = "CSV"  # "CSV" or "LLM"

class ModelOutput(BaseModel):
    doc_title: Optional[str] = None
    summary: Optional[str] = None
    findings: List[Finding] = []

class AnalyzeResponse(BaseModel):
    job_id: str
    report_json: ModelOutput
    download_url: str
class HealthResponse(BaseModel):
    status: str
    local_only: bool
    phrases_loaded: int
    max_prompt_matches: int
    max_prompt_chars: int

# Per-phrase metadata stored from CSV
# compact tuple form for memory:
# (csv_sev:int, feedback:Optional[str], tags:Tuple[str,...], weight:int, enabled:bool, source:Optional[str], version:Optional[str], example:Optional[str], locale:Optional[str], regex:Optional[bool])
MetaTuple = Tuple[int, Optional[str], Tuple[str, ...], int, bool, Optional[str], Optional[str], Optional[str], Optional[str], Optional[bool]]

def sev_rank(s: str) -> int:
    return SEV_TEXT_TO_INT.get(s.lower(), SEV_MED)

def sev_int_rank(i: int) -> int:
    return i  # already ranked small int