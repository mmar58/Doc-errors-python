from typing import List, Dict, Tuple, Optional
import ahocorasick
from models import apply_match_severity
from dataclasses import dataclass

@dataclass(frozen=True)
class MatchResult:
    phrase_norm: str
    page_idx: int  # 0-based
    start_char: int
    end_char: int
    context: str

class GlobalMatcher:
    def __init__(self):
        self._automaton: Optional[ahocorasick.Automaton] = None
        self._ready: bool = False

    def build(self, phrases: List[str]):
        A = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)
        for p in phrases:
            if not p:
                continue
            A.add_word(p, p)
        A.make_automaton()
        self._automaton = A
        self._ready = True

    def is_ready(self) -> bool:
        return self._ready and self._automaton is not None

    def search_pages(
        self,
        pages_text: List[str],
        csv_presence: Dict[str, bool],
        overlay_severity: Optional[Dict[str, int]] = None
    ) -> List[Tuple[str, int, int, int, str, int]]:
        """
        Returns list of tuples: (phrase_norm, page_1_based, start_char, end_char, context_line, applied_sev_int)
        applied severity uses overlay if present; else High by default for CSV phrases.
        """
        if not self.is_ready():
            return []
        A = self._automaton
        results: List[Tuple[str, int, int, int, str, int]] = []

        for page_idx, text in enumerate(pages_text):
            if not text:
                continue
            lower = text.lower()
            # Build line indexes to extract context quickly
            lines = text.splitlines()
            offsets = []
            pos = 0
            for ln in lines:
                offsets.append(pos)
                pos += len(ln) + 1  # +1 for newline approximation

            def context_for(start, end):
                # Map char position to line containing it via binary search
                lo, hi = 0, len(offsets) - 1
                line_idx = 0
                while lo <= hi:
                    mid = (lo + hi) // 2
                    if offsets[mid] <= start:
                        line_idx = mid
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if 0 <= line_idx < len(lines):
                    return lines[line_idx][:1000]
                return ""

            for end_idx, val in A.iter(lower):
                phrase_norm = val
                start_idx = end_idx - len(phrase_norm) + 1
                ctx = context_for(start_idx, end_idx)
                applied = apply_match_severity(
                    csv_presence.get(phrase_norm, False),
                    (overlay_severity or {}).get(phrase_norm)
                )
                results.append((phrase_norm, page_idx + 1, start_idx, end_idx + 1, ctx, applied))

        return results

global_matcher = GlobalMatcher()