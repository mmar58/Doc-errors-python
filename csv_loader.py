import csv
from typing import Dict, Tuple, List
from models import MetaTuple, parse_severity_to_int, SEV_MED
from config import settings

# Normalize: lowercase, trim, collapse spaces, strip quotes
def normalize_phrase(s: str) -> str:
    s = s.strip().strip('"').strip("'").lower()
    parts = s.split()
    return " ".join(parts)

def word_count(s: str) -> int:
    return len(s.split())

def _safe_sniff(sample: str):
    """
    Try to sniff a CSV dialect. Validate fields that commonly break (delimiter, quotechar, escapechar).
    Return (dialect, reason_str). dialect may be None if sniffing fails.
    """
    try:
        sniffer = csv.Sniffer()
        d = sniffer.sniff(sample)
    except Exception as e:
        return None, f"sniffer_exception={type(e).__name__}: {e}"

    # Validate delimiter
    bad = []
    delim = getattr(d, "delimiter", None)
    if not isinstance(delim, str) or len(delim) != 1 or delim in {"", "\x00"}:
        bad.append(f"bad_delimiter={repr(delim)}")

    # Validate quotechar
    qc = getattr(d, "quotechar", None)
    if qc not in (None, "", '"', "'"):
        # allow None/'' or common quote chars
        if not (isinstance(qc, str) and len(qc) == 1):
            bad.append(f"bad_quotechar={repr(qc)}")

    # Validate escapechar
    ec = getattr(d, "escapechar", None)
    if ec not in (None, "", "\\"):
        if not (isinstance(ec, str) and len(ec) == 1):
            bad.append(f"bad_escapechar={repr(ec)}")

    if bad:
        return None, " | ".join(bad)

    return d, "sniff_ok"

def load_csv_streaming(path: str) -> Tuple[Dict[str, MetaTuple], List[str]]:
    phrase_meta: Dict[str, MetaTuple] = {}
    automaton_phrases: List[str] = []

    try:
        with open(path, "r", encoding="utf-8-sig", newline="") as f:
            sample = f.read(4096)
            f.seek(0)

            dialect, sniff_reason = _safe_sniff(sample)
            print(f"[csv_loader] sniff_reason={sniff_reason}")

            rows: List[List[str]] = []
            reader = None

            # 1) Try reader with sniffed dialect (if valid)
            if dialect is not None:
                try:
                    reader = csv.reader(f, dialect)
                    rows = list(reader)
                except Exception as e:
                    print(f"[csv_loader] reader_fail_on_sniffed_dialect: {type(e).__name__}: {e}")
                    f.seek(0)
                    reader = None
                    rows = []

            # 2) Fallback to csv.excel
            if not rows:
                try:
                    f.seek(0)
                    reader = csv.reader(f, csv.excel)
                    rows = list(reader)
                    print("[csv_loader] fallback_used=csv.excel")
                except Exception as e:
                    print(f"[csv_loader] reader_fail_on_csv_excel: {type(e).__name__}: {e}")
                    f.seek(0)
                    reader = None
                    rows = []

            # 3) Final fallback: force comma delimiter minimal dialect
            if not rows:
                try:
                    f.seek(0)
                    class MinimalDialect(csv.Dialect):
                        delimiter = ','
                        quotechar = '"'
                        escapechar = None
                        doublequote = True
                        skipinitialspace = False
                        lineterminator = '\n'
                        quoting = csv.QUOTE_MINIMAL
                    reader = csv.reader(f, MinimalDialect)
                    rows = list(reader)
                    print("[csv_loader] fallback_used=MinimalDialect(delimiter=',')")
                except Exception as e:
                    print(f"[csv_loader] reader_fail_on_minimal: {type(e).__name__}: {e}")
                    return {}, []

    except FileNotFoundError:
        print(f"[csv_loader] file_not_found: {path}")
        return {}, []

    if not rows:
        print("[csv_loader] no_rows_after_all_fallbacks")
        return {}, []

    headers_present = any(cell.strip() for cell in rows[0])
    header_map: Dict[str, int] = {}
    start_idx = 0

    if headers_present:
        hdrs = [h.strip().lower() for h in rows[0]]
        start_idx = 1
        for idx, h in enumerate(hdrs):
            header_map[h] = idx
        print(f"[csv_loader] detected_headers={hdrs}")
    else:
        hdrs = []
        header_map = {}
        print("[csv_loader] no_header_detected; using first column as phrase")

    # Column access helper
    def get_col(row, names, default=None):
        for n in names:
            idx = header_map.get(n)
            if idx is not None and idx < len(row):
                return row[idx]
        return default

    PHRASE_KEYS = ["phrase", "tortured_phrases"]
    SEV_KEYS = ["severity", "Severity"]
    FEEDBACK_KEYS = ["feedback", "Feedback"]
    TAGS_KEYS = ["tags", "Tags"]
    WEIGHT_KEYS = ["weight", "Weight"]
    ENABLED_KEYS = ["enabled", "Enabled"]
    SOURCE_KEYS = ["source", "Source"]
    VERSION_KEYS = ["version", "Version"]
    EXAMPLE_KEYS = ["example_context", "Example_context"]
    LOCALE_KEYS = ["locale", "Locale"]
    REGEX_KEYS = ["regex", "Regex"]

    default_csv_sev = parse_severity_to_int(settings.DEFAULT_CSV_SEVERITY, SEV_MED)

    # Dedup buffer: norm -> (weight, -sev_int_for_preference, text_len, original_phrase, meta)
    best: Dict[str, Tuple[int, int, int, str, MetaTuple]] = {}

    for i in range(start_idx, len(rows)):
        row = rows[i]
        if not row:
            continue

        phrase_raw = get_col(row, PHRASE_KEYS, None)
        if phrase_raw is None:
            if not headers_present and len(row) > 0:
                phrase_raw = row[0]
            else:
                continue

        phrase_norm = normalize_phrase(phrase_raw)
        if not phrase_norm:
            continue
        if word_count(phrase_norm) > 12:
            continue

        enabled_val = get_col(row, ENABLED_KEYS, "")
        enabled = True
        if enabled_val is not None and str(enabled_val).strip() != "":
            v = str(enabled_val).strip().lower()
            enabled = v not in {"false", "0", "no", "n"}

        if not enabled:
            continue

        sev_str = get_col(row, SEV_KEYS, None)
        csv_sev = parse_severity_to_int(sev_str, default_csv_sev)

        tags_raw = get_col(row, TAGS_KEYS, None)
        if tags_raw:
            parts = [t.strip() for t in tags_raw.replace("|", ",").split(",") if t.strip()]
            tags = tuple(parts)
        else:
            tags = tuple()

        weight_raw = get_col(row, WEIGHT_KEYS, None)
        try:
            weight = int(weight_raw) if weight_raw not in (None, "") else settings.DEFAULT_WEIGHT
        except ValueError:
            weight = settings.DEFAULT_WEIGHT

        feedback = get_col(row, FEEDBACK_KEYS, None)
        source = get_col(row, SOURCE_KEYS, None)
        version = get_col(row, VERSION_KEYS, None)
        example_context = get_col(row, EXAMPLE_KEYS, None)
        locale = get_col(row, LOCALE_KEYS, None)
        regex_raw = get_col(row, REGEX_KEYS, None)
        regex = None
        if regex_raw is not None and str(regex_raw).strip() != "":
            v = str(regex_raw).strip().lower()
            regex = v in {"true", "1", "yes", "y"}

        meta: MetaTuple = (
            csv_sev, feedback, tags, weight, True, source, version, example_context, locale, regex
        )

        # Dedup policy: keep max weight, then more severe, then longer original text
        w = weight
        sev_pref = -csv_sev
        text_len = len(phrase_raw)
        prev = best.get(phrase_norm)
        if prev is None or (w, sev_pref, text_len) > (prev[0], prev[1], prev[2]):
            best[phrase_norm] = (w, sev_pref, text_len, phrase_raw, meta)

    # Build final dict/list
    for norm, (_w, _sv, _tl, _orig, meta) in best.items():
        phrase_meta[norm] = meta
        automaton_phrases.append(norm)

    print(f"[csv_loader] loaded_phrases={len(automaton_phrases)} unique={len(phrase_meta)}")
    return phrase_meta, automaton_phrases