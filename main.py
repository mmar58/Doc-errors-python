"""
README (Quickstart)

1) Install deps:
   pip install fastapi uvicorn pydantic pydantic-settings PyPDF2 pyahocorasick reportlab

2) Run the server:
   python main.py

3) Web GUI:
   - GET /                    : Upload form
   - POST /ui/analyze         : Handles form submit and redirects to result
   - GET /ui/result/{job_id}  : Rich HTML cards + download link
   - GET /download/{job_id}   : Download generated PDF
"""

import os
import json
import uuid
from typing import Dict, List, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import AnalyzeResponse, ModelOutput, Finding, SEV_INT_TO_TEXT, SEV_TEXT_TO_INT
from csv_loader import load_csv_streaming, normalize_phrase
from matcher_ac import global_matcher
from pdf_utils import extract_text_per_page, content_hash_bytes
from reporting import build_report_pdf
from llm_clients import query_models, query_models_discover
from merge import merge_findings

app = FastAPI(title="PDF Severity Phrase Analyzer", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
PHRASE_META: Dict[str, tuple] = {}
CSV_PRESENCE: Dict[str, bool] = {}
WEIGHT_BY_PHRASE: Dict[str, int] = {}
LOCAL_ONLY = False
JOBS: Dict[str, str] = {}               # job_id -> generated report PDF path
JOB_DATA: Dict[str, ModelOutput] = {}   # job_id -> report_json for inline UI

def ensure_reports_dir():
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    global PHRASE_META, CSV_PRESENCE, WEIGHT_BY_PHRASE, LOCAL_ONLY
    PHRASE_META, automaton_phrases = load_csv_streaming(settings.SEED_CSV_PATH)
    CSV_PRESENCE = {p: True for p in PHRASE_META.keys()}
    WEIGHT_BY_PHRASE = {p: (PHRASE_META[p][3] if len(PHRASE_META[p]) > 3 else 0) for p in PHRASE_META}
    global_matcher.build(automaton_phrases)
    LOCAL_ONLY = not bool(getattr(settings, "ABACUS_API_KEY", None))
    ensure_reports_dir()

# Home page UI
@app.get("/", response_class=HTMLResponse)
def home_page():
    loaded = len(PHRASE_META)
    mode = "LOCAL" if LOCAL_ONLY else "CLOUD"
    return """
    <html>
      <head>
        <meta charset="utf-8" />
        <title>PDF Severity Phrase Analyzer</title>
        <style>
          body { font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; }
          .card { max-width: 900px; padding: 24px; border: 1px solid #e5e7eb; border-radius: 12px; }
          label { display:block; margin-top: 16px; font-weight: 600; }
          input[type=file] { margin-top: 8px; }
          textarea { width: 100%%; height: 120px; margin-top: 8px; }
          button { margin-top: 20px; padding: 10px 16px; background: #111827; color: white; border: none; border-radius: 8px; cursor: pointer; }
          .note { color: #6b7280; font-size: 14px; }
          .footer { margin-top: 16px; color: #6b7280; font-size: 12px; }
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Upload a PDF</h2>
          <form action="/ui/analyze" method="post" enctype="multipart/form-data">
            <label>PDF file</label>
            <input type="file" name="file" accept="application/pdf" required />
            <label>Optional severity overrides (JSON)</label>
            <textarea name="severity_terms" placeholder='{"some phrase": "High", "another": "Low"}'></textarea>
            <div class="note">Leave blank to use defaults. All CSV phrases are High unless overridden. AI reasoning and discovery are always enabled.</div>
            <button type="submit">Generate Report</button>
          </form>
          <div class="footer">Loaded phrases: %d | Mode: %s</div>
        </div>
      </body>
    </html>
    """ % (loaded, mode)

# Helper: parse override JSON from the form
def parse_overlay_json(s: Optional[str]) -> Dict[str, int]:
    if not s:
        return {}
    try:
        data = json.loads(s)
    except Exception:
        raise HTTPException(status_code=400, detail="Malformed JSON in severity_terms")
    if not isinstance(data, dict):
        raise HTTPException(status_code=400, detail="severity_terms must be a JSON object")
    overlay: Dict[str, int] = {}
    for k, v in data.items():
        if not isinstance(k, str) or not isinstance(v, str):
            raise HTTPException(status_code=400, detail="severity_terms must map phrase -> 'High|Medium|Low'")
        nk = normalize_phrase(k)
        if not nk:
            continue
        sev = SEV_TEXT_TO_INT.get(v.strip().lower())
        if sev is None:
            raise HTTPException(status_code=400, detail=f"Invalid severity for '{k}': {v}")
        overlay[nk] = sev
    return overlay

def compact_snippet(text: str, start: int, end: int, window: int = 120) -> str:
    s = max(0, start - window)
    e = min(len(text), end + window)
    snippet = text[s:e].replace("\n", " ").strip()
    snippet = " ".join(snippet.split())
    if len(snippet) > 400:
        snippet = snippet[:400]
    return snippet

# Submit handler that uses analyze() and redirects to result page
@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...), severity_terms: Optional[str] = Form(default=None)):
    res = await analyze(file=file, severity_terms=severity_terms)
    # res is a Pydantic AnalyzeResponse (not a JSONResponse)
    job_id = res.job_id
    return RedirectResponse(url=f"/ui/result/{job_id}", status_code=303)

# Result page
@app.get("/ui/result/{job_id}", response_class=HTMLResponse)
def ui_result(job_id: str):
    path = JOBS.get(job_id)
    report = JOB_DATA.get(job_id)
    if not path or not os.path.exists(path) or report is None:
        return HTMLResponse("<h3>Report not found</h3>", status_code=404)

    def sev_color(sev: str) -> str:
        return {
            "High": "#ef4444",
            "Medium": "#f59e0b",
            "Low": "#10b981",
        }.get(sev, "#6b7280")

    cards_html = []
    for f in report.findings:
        color = sev_color(f.severity)
        phrase = (f.phrase or "").replace("<", "&lt;").replace(">", "&gt;")
        suggestion = (f.suggestion or "").replace("<", "&lt;").replace(">", "&gt;")
        context = (f.context or "").replace("<", "&lt;").replace(">", "&gt;")
        cards_html.append(f"""
          <div style="border:1px solid #e5e7eb; border-radius:10px; margin:16px 0; overflow:hidden;">
            <div style="height: 8px; background:{color};"></div>
            <div style="padding:16px;">
              <div><strong>Severity:</strong> {f.severity}{' [LLM]' if getattr(f, 'source', 'CSV') == 'LLM' else ''}</div>
              <div><strong>Page:</strong> {f.page}</div>
              <div style="margin-top:8px;"><strong>Exact Matched Phrase:</strong><br/>"<code>{phrase}</code>"</div>
              {"<div style='margin-top:8px;'><strong>Suggested Rewrite:</strong><br/>\"<code>%s</code>\"</div>" % suggestion if suggestion else ""}
              <div style="margin-top:12px; color:#374151;">
                <div style="font-weight:600; margin-bottom:4px;">Context:</div>
                <div style="background:#f9fafb; padding:8px; border-radius:6px; font-family:ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace; white-space:pre-wrap;">
                  {context}
                </div>
              </div>
            </div>
          </div>
        """)

    cards_section = "\n".join(cards_html)
    safe_title = (report.doc_title or "PDF Severity Phrase Analysis").replace("<", "&lt;").replace(">", "&gt;")
    safe_summary = (report.summary or "").replace("<", "&lt;").replace(">", "&gt;")

    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Report</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; }}
          .container {{ max-width: 960px; margin: auto; }}
          .hdr {{ display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }}
          a.btn {{ background:#111827; color:white; text-decoration:none; padding:10px 14px; border-radius:8px; }}
          .summary {{ color:#374151; margin-top: 8px; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="hdr">
            <h2 style="margin:0;">{safe_title}</h2>
            <a class="btn" href="/download/{job_id}" target="_blank">Download PDF</a>
          </div>
          {"<div class='summary'>" + safe_summary + "</div>" if safe_summary else ""}
          <div style="margin-top:20px;">
            {cards_section}
          </div>
          <div style="margin-top:24px;">
            <a href="/">Analyze another PDF</a>
          </div>
        </div>
      </body>
    </html>
    """

# Core analyze logic (also available via API)
@app.post("/analyze")
async def analyze(file: UploadFile = File(...), severity_terms: Optional[str] = Form(default=None)) -> AnalyzeResponse:
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    bytes_data = await file.read()
    if not bytes_data:
        raise HTTPException(status_code=400, detail="Empty file")

    # 1) Extract text per page
    try:
        pages_text, detected_title = extract_text_per_page(bytes_data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"PDF parse failure: {e}")

    # 2) Parse overrides
    overlay = parse_overlay_json(severity_terms)

    # 3) Run AC matcher
    matches = global_matcher.search_pages(pages_text, CSV_PRESENCE, overlay_severity=overlay)

    # 4) Build CSV findings
    local_findings: List[Finding] = []
    phrase_first_occurrence_snippet: Dict[str, str] = {}
    phrase_counts: Dict[str, int] = {}
    applied_severities_text: Dict[str, str] = {}

    for phrase_norm, page1, start_char, end_char, context, sev_int in matches:
        phrase_counts[phrase_norm] = phrase_counts.get(phrase_norm, 0) + 1
        sev_text = SEV_INT_TO_TEXT.get(sev_int, "High")
        applied_severities_text[phrase_norm] = sev_text

        if phrase_norm not in phrase_first_occurrence_snippet:
            text = pages_text[page1 - 1]
            snippet = compact_snippet(text, start_char, end_char)
            phrase_first_occurrence_snippet[phrase_norm] = snippet

        feedback = None
        meta = PHRASE_META.get(phrase_norm)
        if meta:
            feedback = meta[1]

        local_findings.append(Finding(
            phrase=phrase_norm,
            severity=sev_text,
            suggestion=feedback or None,
            page=page1,
            start_char=start_char,
            end_char=end_char,
            context=context or None,
            source="CSV"
        ))

    # Prepare LLM inputs for the original reasoning step
    unique_sorted: List[Tuple[str, str, int]] = []
    for p, cnt in phrase_counts.items():
        sev_text = applied_severities_text.get(p, "High")
        unique_sorted.append((p, sev_text, cnt))
    unique_sorted.sort(key=lambda t: (
        -(PHRASE_META.get(t[0], (None, None, (), 0, True, None, None, None, None, None))[3]),
        {"High": 0, "Medium": 1, "Low": 2}.get(t[1], 0),
        t[0]
    ))
    if len(unique_sorted) > settings.MAX_PROMPT_MATCHES:
        unique_sorted = unique_sorted[:settings.MAX_PROMPT_MATCHES]
    capped_snippets = {p: phrase_first_occurrence_snippet.get(p, "") for p, _, _ in unique_sorted}
    sev_counts = {"High": 0, "Medium": 0, "Low": 0}
    for _, sev_text, cnt in unique_sorted:
        sev_counts[sev_text] = sev_counts.get(sev_text, 0) + cnt

    # 5) Original model call (rewrites/title/summary)
    model_outputs: List[ModelOutput] = []
    try:
        if unique_sorted:
            outs = await query_models(detected_title, unique_sorted, capped_snippets, sev_counts)
            model_outputs = outs or []
    except Exception as e:
        print(f"[analyze] query_models_failed: {type(e).__name__}: {e}")

    # 6) Discovery pass for extra risky phrases
    llm_discovered: List[Finding] = []
    try:
        pages_to_scan: List[Tuple[int, str]] = []
        if local_findings:
            pages_with_hits = sorted({f.page for f in local_findings if f.page})
            for p in pages_with_hits[:settings.DISCOVERY_MAX_PAGES_PER_DOC]:
                pages_to_scan.append((p, pages_text[p - 1]))
        else:
            n = min(settings.DISCOVERY_SCAN_FIRST_N_PAGES_IF_NO_MATCHES, len(pages_text))
            for i in range(n):
                pages_to_scan.append((i + 1, pages_text[i]))

        if settings.DISCOVERY_ENABLED and pages_to_scan:
            llm_discovered = await query_models_discover(
                doc_title_hint=detected_title,
                pages=pages_to_scan,
                max_findings_per_page=settings.DISCOVERY_MAX_FINDINGS_PER_PAGE,
            )
    except Exception as e:
        print(f"[analyze] discovery_failed: {type(e).__name__}: {e}")

    # 7) Merge everything
    model_findings_all: List[List[Finding]] = []
    for mo in model_outputs:
        model_findings_all.append(mo.findings)
    merged = merge_findings(local_findings, model_findings_all, WEIGHT_BY_PHRASE, llm_discovered=llm_discovered)

    # 8) Compose title/summary
    doc_title = detected_title
    summary = None
    for mo in model_outputs:
        if not doc_title and getattr(mo, "doc_title", None):
            doc_title = mo.doc_title
        if getattr(mo, "summary", None):
            summary = mo.summary
    if not summary:
        h = sum(1 for f in merged if f.severity == "High")
        m = sum(1 for f in merged if f.severity == "Medium")
        l = sum(1 for f in merged if f.severity == "Low")
        summary = f"We identified {h} high, {m} medium, and {l} low severity phrase(s)."

    report_json = ModelOutput(
        doc_title=doc_title or None,
        summary=summary or None,
        findings=merged
    )

    # 9) Generate PDF report
    job_id = str(uuid.uuid4())
    content_hash = content_hash_bytes(bytes_data)
    out_name = f"{job_id}_{content_hash}.pdf"
    out_path = os.path.join(settings.REPORTS_DIR, out_name)
    try:
        build_report_pdf(out_path, report_json.doc_title or "PDF Severity Phrase Analysis", report_json.summary or "", report_json.findings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report PDF: {e}")

    # 10) Store artifacts for UI and return response
    JOBS[job_id] = out_path
    JOB_DATA[job_id] = report_json
    download_url = f"/download/{job_id}"
    return AnalyzeResponse(job_id=job_id, report_json=report_json, download_url=download_url)

@app.get("/download/{job_id}")
def download(job_id: str):
    path = JOBS.get(job_id)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Report not found")
    filename = os.path.basename(path)
    return FileResponse(path, filename=filename, media_type="application/pdf")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")