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
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

from config import settings
from models import AnalyzeResponse, ModelOutput, Finding, SEV_INT_TO_TEXT, SEV_TEXT_TO_INT
from csv_loader import load_csv_streaming, normalize_phrase
from matcher_ac import global_matcher
from single_word_matcher import global_single_word_matcher
from pdf_utils import extract_text_per_page, content_hash_bytes
from reporting import build_report_pdf
from llm_clients import query_models, query_models_discover
from merge import merge_findings
from progress_tracker import progress_tracker, update_job_progress, ProgressStatus, ProgressUpdate

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
SINGLE_WORDS_META: Dict[str, tuple] = {}  # Track single words separately
LOCAL_ONLY = False
JOBS: Dict[str, str] = {}               # job_id -> generated report PDF path
JOB_DATA: Dict[str, ModelOutput] = {}   # job_id -> report_json for inline UI

def ensure_reports_dir():
    os.makedirs(settings.REPORTS_DIR, exist_ok=True)

@app.on_event("startup")
def startup_event():
    global PHRASE_META, CSV_PRESENCE, WEIGHT_BY_PHRASE, SINGLE_WORDS_META, LOCAL_ONLY
    PHRASE_META, automaton_phrases, SINGLE_WORDS_META = load_csv_streaming(settings.SEED_CSV_PATH)
    CSV_PRESENCE = {p: True for p in PHRASE_META.keys()}
    WEIGHT_BY_PHRASE = {p: (PHRASE_META[p][3] if len(PHRASE_META[p]) > 3 else 0) for p in PHRASE_META}
    global_matcher.build(automaton_phrases)
    
    # Build single word matcher
    single_words = list(SINGLE_WORDS_META.keys())
    global_single_word_matcher.build(single_words)
    
    LOCAL_ONLY = not bool(getattr(settings, "ABACUS_API_KEY", None))
    ensure_reports_dir()

# Home page UI
@app.get("/", response_class=HTMLResponse)
def home_page():
    loaded = len(PHRASE_META)
    single_words = len(SINGLE_WORDS_META)
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
          .feature { background: #f3f4f6; padding: 12px; border-radius: 6px; margin-top: 12px; }
        </style>
      </head>
      <body>
        <div class="card">
          <h2>Upload a PDF</h2>
          <div class="feature">
            <strong>Enhanced Single Word Analysis:</strong> Now analyzes individual words from the lexicon using AI to identify contextual phrases and provide targeted suggestions.
          </div>
          <form action="/ui/analyze" method="post" enctype="multipart/form-data">
            <label>PDF file</label>
            <input type="file" name="file" accept="application/pdf" required />
            <label>Optional severity overrides (JSON)</label>
            <textarea name="severity_terms" placeholder='{"some phrase": "High", "another": "Low"}'></textarea>
            <div class="note">Leave blank to use defaults. All CSV phrases are High unless overridden. AI reasoning and discovery are always enabled.</div>
            <button type="submit">Generate Report</button>
          </form>
          <div class="footer">Loaded phrases: %d | Single words for context analysis: %d | Mode: %s</div>
        </div>
      </body>
    </html>
    """ % (loaded, single_words, mode)

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

# Submit handler that uses analyze() and redirects to progress page
@app.post("/ui/analyze")
async def ui_analyze(file: UploadFile = File(...), severity_terms: Optional[str] = Form(default=None)):
    # Generate job ID immediately
    job_id = str(uuid.uuid4())
    print(f"[ui_analyze] Starting analysis with job ID: {job_id}")
    
    # Read file data immediately before starting background task
    try:
        print(f"[ui_analyze] Reading file: {file.filename}")
        file_data = await file.read()
        filename = file.filename
        print(f"[ui_analyze] Successfully read {len(file_data)} bytes from {filename}")
    except Exception as e:
        print(f"[ui_analyze] ERROR reading file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to read uploaded file: {str(e)}")
    
    # Start analysis in background - pass the bytes data and filename
    print(f"[ui_analyze] Starting background task for job {job_id}")
    import asyncio
    asyncio.create_task(analyze_with_progress(job_id, file_data, filename, severity_terms))
    
    # Redirect to progress page immediately
    print(f"[ui_analyze] Redirecting to progress page for job {job_id}")
    return RedirectResponse(url=f"/ui/progress/{job_id}", status_code=302)

async def analyze_with_progress(job_id: str, file_data: bytes, filename: str, severity_terms: Optional[str] = None):
    """Run analysis with progress tracking"""
    print(f"[analyze_with_progress] Starting analysis for job {job_id}, file: {filename}")
    try:
        print(f"[analyze_with_progress] Updating status to STARTED for job {job_id}")
        await update_job_progress(job_id, ProgressStatus.STARTED)
        
        # File data is already read, pass it directly to analyze_internal
        print(f"[analyze_with_progress] Calling analyze_internal for job {job_id}")
        result = await analyze_internal(job_id, file_data, filename, severity_terms)
        
        # Store result
        print(f"[analyze_with_progress] Storing results for job {job_id}")
        JOBS[job_id] = result["pdf_path"]
        JOB_DATA[job_id] = result["report_json"]
        
        print(f"[analyze_with_progress] Analysis completed successfully for job {job_id}")
        await update_job_progress(job_id, ProgressStatus.COMPLETED, "Analysis complete - ready to view results")
        
    except Exception as e:
        print(f"[analyze_with_progress] ERROR for job {job_id}: {e}")
        await update_job_progress(job_id, ProgressStatus.FAILED, f"Analysis failed: {str(e)}")
        print(f"[analyze_with_progress] Failed for job {job_id}: {e}")

async def analyze_internal(job_id: str, bytes_data: bytes, filename: str, severity_terms: Optional[str] = None):
    """Internal analysis function with progress tracking"""
    print(f"[analyze_internal] Starting analysis for job {job_id}, filename: {filename}")
    import asyncio
    
    if not filename.lower().endswith(".pdf"):
        print(f"[analyze_internal] ERROR: File {filename} is not a PDF")
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if not bytes_data:
        print(f"[analyze_internal] ERROR: File data is empty")
        raise HTTPException(status_code=400, detail="Empty file")

    print(f"[analyze_internal] File validation passed, data size: {len(bytes_data)} bytes")

    # 1) Extract text per page
    print(f"[analyze_internal] Step 1: Extracting text from PDF")
    await update_job_progress(job_id, ProgressStatus.EXTRACTING_TEXT, f"Extracting text from {filename}")
    try:
        pages_text, detected_title = extract_text_per_page(bytes_data)
        print(f"[analyze_internal] Text extraction successful: {len(pages_text)} pages, title: {detected_title}")
    except Exception as e:
        print(f"[analyze_internal] ERROR during text extraction: {e}")
        raise HTTPException(status_code=400, detail=f"PDF parse failure: {e}")

    # 2) Parse overrides
    print(f"[analyze_internal] Step 2: Parsing severity overrides")
    overlay = parse_overlay_json(severity_terms)
    print(f"[analyze_internal] Severity overrides parsed: {len(overlay)} entries")

    # 3) Run AC matcher
    print(f"[analyze_internal] Step 3: Running phrase matching")
    await update_job_progress(job_id, ProgressStatus.MATCHING_PHRASES, f"Searching for {len(PHRASE_META)} known phrases")
    matches = global_matcher.search_pages(pages_text, CSV_PRESENCE, overlay_severity=overlay)
    print(f"[analyze_internal] Phrase matching complete: {len(matches)} matches found")

    # 4) Build CSV findings
    print(f"[analyze_internal] Step 4: Building CSV findings from matches")
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
    
    print(f"[analyze_internal] CSV findings built: {len(local_findings)} findings")

    # 4.5) Single word analysis - NEW FEATURE
    print(f"[analyze_internal] Step 4.5: Starting single word analysis")
    await update_job_progress(job_id, ProgressStatus.ANALYZING_SINGLE_WORDS, f"Analyzing {len(SINGLE_WORDS_META)} single words with AI")
    single_word_findings: List[Finding] = []
    try:
        if settings.ENABLE_SINGLE_WORD_ANALYSIS and global_single_word_matcher.is_ready():
            print(f"[analyze_internal] Single word analysis enabled, searching for words")
            word_matches = global_single_word_matcher.search_pages(pages_text)
            if word_matches:
                print(f"[analyze_internal] Found {len(word_matches)} single word matches for LLM analysis")
                single_word_findings = await global_single_word_matcher.analyze_words_with_llm(
                    word_matches, detected_title, job_id
                )
                print(f"[analyze_internal] LLM identified {len(single_word_findings)} issues from single words")
            else:
                print(f"[analyze_internal] No single word matches found")
        else:
            print(f"[analyze_internal] Single word analysis disabled or not ready")
    except Exception as e:
        print(f"[analyze_internal] single_word_analysis_failed: {type(e).__name__}: {e}")

    # Prepare LLM inputs for the original reasoning step
    print(f"[analyze_internal] Step 5: Preparing LLM analysis")
    await update_job_progress(job_id, ProgressStatus.QUERYING_MODELS, "Getting AI analysis and suggestions")
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

    # 6) Discovery
    await update_job_progress(job_id, ProgressStatus.DISCOVERING_PHRASES, "Discovering additional issues with AI")
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
    await update_job_progress(job_id, ProgressStatus.MERGING_RESULTS, "Merging and prioritizing all findings")
    model_findings_all: List[List[Finding]] = []
    for mo in model_outputs:
        model_findings_all.append(mo.findings)
    
    # Add single word findings to the LLM discovered findings
    all_llm_discovered = (llm_discovered or []) + single_word_findings
    
    merged = merge_findings(local_findings, model_findings_all, WEIGHT_BY_PHRASE, llm_discovered=all_llm_discovered)

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
    await update_job_progress(job_id, ProgressStatus.GENERATING_REPORT, "Creating downloadable PDF report")
    content_hash = content_hash_bytes(bytes_data)
    out_name = f"{job_id}_{content_hash}.pdf"
    out_path = os.path.join(settings.REPORTS_DIR, out_name)
    try:
        build_report_pdf(out_path, report_json.doc_title or "PDF Severity Phrase Analysis", report_json.summary or "", report_json.findings)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate report PDF: {e}")

    return {
        "pdf_path": out_path,
        "report_json": report_json
    }

# Progress page
@app.get("/ui/progress/{job_id}", response_class=HTMLResponse)
def ui_progress(job_id: str):
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <title>Analyzing PDF - Progress</title>
        <style>
          body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; margin: 40px; background: #f9fafb; }}
          .container {{ max-width: 800px; margin: 0 auto; }}
          .card {{ background: white; padding: 32px; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }}
          .progress-bar {{ width: 100%; height: 24px; background: #e5e7eb; border-radius: 12px; overflow: hidden; margin: 20px 0; }}
          .progress-fill {{ height: 100%; background: linear-gradient(90deg, #3b82f6, #1d4ed8); transition: width 0.3s ease; width: 0%; }}
          .status {{ font-size: 18px; font-weight: 600; margin-bottom: 8px; color: #1f2937; }}
          .step-info {{ color: #6b7280; margin-bottom: 16px; }}
          .details {{ background: #f3f4f6; padding: 12px; border-radius: 6px; margin-top: 12px; font-family: monospace; font-size: 14px; }}
          .spinner {{ display: inline-block; width: 20px; height: 20px; border: 3px solid #e5e7eb; border-top: 3px solid #3b82f6; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 8px; }}
          @keyframes spin {{ 0% {{ transform: rotate(0deg); }} 100% {{ transform: rotate(360deg); }} }}
          .complete {{ color: #059669; }}
          .error {{ color: #dc2626; }}
          .result-button {{ display: none; background: #059669; color: white; padding: 12px 24px; border: none; border-radius: 8px; font-size: 16px; cursor: pointer; margin-top: 20px; }}
          .result-button:hover {{ background: #047857; }}
        </style>
      </head>
      <body>
        <div class="container">
          <div class="card">
            <h2>Analyzing Your PDF</h2>
            <div id="status" class="status">
              <span class="spinner"></span>Starting analysis...
            </div>
            <div class="progress-bar">
              <div id="progress-fill" class="progress-fill"></div>
            </div>
            <div id="step-info" class="step-info">Initializing...</div>
            <div id="details" class="details" style="display: none;"></div>
            <button id="result-button" class="result-button" onclick="window.location.href='/ui/result/{job_id}'">
              View Results
            </button>
          </div>
        </div>
        
        <script>
          const eventSource = new EventSource('/progress-stream/{job_id}');
          const statusEl = document.getElementById('status');
          const progressFill = document.getElementById('progress-fill');
          const stepInfo = document.getElementById('step-info');
          const details = document.getElementById('details');
          const resultButton = document.getElementById('result-button');
          
          eventSource.onmessage = function(event) {{
            const data = JSON.parse(event.data);
            
            // Skip keepalive messages
            if (data.type === 'keepalive') {{
              return;
            }}
            
            // Update progress bar
            progressFill.style.width = data.progress_percent + '%';
            
            // Update status
            if (data.status === 'completed') {{
              statusEl.innerHTML = '<span class="complete">✓</span> ' + data.message;
              stepInfo.textContent = 'Analysis completed successfully!';
              resultButton.style.display = 'inline-block';
              eventSource.close();
            }} else if (data.status === 'failed') {{
              statusEl.innerHTML = '<span class="error">✗</span> Analysis failed';
              stepInfo.textContent = data.details || 'An error occurred during analysis';
              stepInfo.className = 'step-info error';
              eventSource.close();
            }} else {{
              statusEl.innerHTML = '<span class="spinner"></span>' + data.message;
              stepInfo.textContent = `Step ${{data.current_step_number}} of ${{data.total_steps}}: ${{data.current_step}}`;
            }}
            
            // Show details if available
            if (data.details) {{
              details.textContent = data.details;
              details.style.display = 'block';
            }}
          }};
          
          eventSource.onerror = function(event) {{
            console.error('EventSource failed:', event);
            statusEl.innerHTML = '<span class="error">✗</span> Connection lost';
            stepInfo.textContent = 'Lost connection to server. Please refresh to check status.';
            eventSource.close();
          }};
        </script>
      </body>
    </html>
    """

# Server-Sent Events endpoint for progress updates
@app.get("/progress-stream/{job_id}")
async def progress_stream(job_id: str):
    import time
    import asyncio
    
    async def event_stream():
        queue = progress_tracker.subscribe(job_id)
        try:
            # Send current progress if available
            current_progress = progress_tracker.get_progress(job_id)
            if current_progress:
                yield f"data: {json.dumps(current_progress.to_dict())}\n\n"
            
            # Stream updates
            while True:
                try:
                    # Wait for update with timeout
                    update = await asyncio.wait_for(queue.get(), timeout=30.0)
                    yield f"data: {json.dumps(update.to_dict())}\n\n"
                    
                    # Stop streaming if completed or failed
                    if update.status in [ProgressStatus.COMPLETED, ProgressStatus.FAILED]:
                        break
                        
                except asyncio.TimeoutError:
                    # Send keepalive
                    keepalive_data = {"type": "keepalive", "timestamp": time.time()}
                    yield f"data: {json.dumps(keepalive_data)}\n\n"
                    
        except Exception as e:
            print(f"[progress_stream] Error: {e}")
        finally:
            progress_tracker.unsubscribe(job_id, queue)
    
    return StreamingResponse(
        event_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

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
        
        # Add source indicator for enhanced tracking
        source_badge = ""
        if f.source == "LLM-SingleWord-Batch":
            source_badge = " <span style='background: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 12px; font-size: 11px; font-weight: 500; margin-left: 8px;'>AI-Enhanced</span>"
        elif f.source == "LLM-SingleWord":
            source_badge = " <span style='background: #dbeafe; color: #1e40af; padding: 2px 6px; border-radius: 12px; font-size: 11px; font-weight: 500; margin-left: 8px;'>AI-Enhanced</span>"
        elif f.source == "LLM":
            source_badge = " <span style='background: #dcfce7; color: #166534; padding: 2px 6px; border-radius: 12px; font-size: 11px; font-weight: 500; margin-left: 8px;'>AI-Discovered</span>"
        elif f.source == "CSV":
            source_badge = " <span style='background: #fef3c7; color: #92400e; padding: 2px 6px; border-radius: 12px; font-size: 11px; font-weight: 500; margin-left: 8px;'>Lexicon</span>"
        
        cards_html.append(f"""
          <div style="border-left: 4px solid {color}; background: white; margin: 16px 0; padding: 16px; border-radius: 0 8px 8px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
            
            <!-- Row 1: Severity -->
            <div style="margin-bottom: 12px;">
              <strong>Severity:</strong> <span style="color: {color}; font-weight: bold;">{f.severity}</span>{source_badge}
            </div>
            
            <!-- Row 2: Page -->
            <div style="margin-bottom: 12px;">
              <strong>Page:</strong> {f.page}
            </div>
            
            <!-- Row 3: Exact Matched Phrase -->
            <div style="margin-bottom: 12px;">
              <strong>Exact Matched Phrase:</strong> "{phrase}"
            </div>
            
            <!-- Row 4: Suggested Rewrite -->
            <div style="margin-bottom: 12px;">
              <strong>Suggested Rewrite:</strong> "{suggestion}"
            </div>
            
            <!-- Row 5: Context -->
            <div style="margin-bottom: 0;">
              <strong>Context:</strong> {context}
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
    # Generate a job ID for this analysis
    job_id = str(uuid.uuid4())
    
    # Read file data
    bytes_data = await file.read()
    
    # Use the internal analysis function without progress tracking
    result = await analyze_internal(job_id, bytes_data, file.filename, severity_terms)
    
    # Store results
    JOBS[job_id] = result["pdf_path"]
    JOB_DATA[job_id] = result["report_json"]
    
    # Return response
    download_url = f"/download/{job_id}"
    return AnalyzeResponse(job_id=job_id, report_json=result["report_json"], download_url=download_url)

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