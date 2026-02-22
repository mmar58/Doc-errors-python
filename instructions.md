# PDF Severity Phrase Analyzer — Instructions

A FastAPI web application that analyzes academic PDF documents for **"tortured phrases"** — awkward, unnatural, or grammatically problematic expressions commonly found in machine-translated or low-quality academic papers. It combines a curated CSV lexicon, Aho-Corasick pattern matching, and dual AI models (GPT-5 and Claude) to detect and suggest rewrites for problematic phrases.

---

## Prerequisites

- Python 3.10+
- An **Abacus AI / RouteLLM API key** (for AI-powered analysis). The system works in LOCAL mode without one, but AI features are disabled.

---

## Installation

1. **Clone / download** the repository into your working directory.

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   This installs: `fastapi`, `uvicorn`, `pydantic`, `pydantic-settings`, `httpx`, `PyPDF2`, `reportlab`, `python-multipart`, `pyahocorasick`.

---

## Configuration

Settings are controlled via `config.py` and can be overridden with environment variables or a `.env` file in the project root.

| Variable | Default | Description |
|---|---|---|
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |
| `ABACUS_API_KEY` | *(set in config)* | Your Abacus/RouteLLM API key |
| `ABACUS_BASE_URL` | `https://routellm.abacus.ai/v1` | API endpoint |
| `SEED_CSV_PATH` | `Tortured_Phrases_Lexicon_2.csv` | Path to the phrase lexicon CSV |
| `REPORTS_DIR` | `reports` | Folder where generated PDF reports are saved |
| `MAX_PROMPT_MATCHES` | `300` | Max phrases sent to AI per request |
| `LLM_TIMEOUT_SECS` | `18.0` | Timeout for standard AI calls |
| `DISCOVERY_TIMEOUT_SECS` | `45.0` | Timeout for AI discovery calls |
| `ENABLE_SINGLE_WORD_ANALYSIS` | `true` | Enable AI analysis of single-word lexicon entries |
| `DISCOVERY_ENABLED` | `true` | Enable AI-powered phrase discovery |

**Example `.env` file:**
```env
ABACUS_API_KEY=your_api_key_here
PORT=8080
ENABLE_SINGLE_WORD_ANALYSIS=true
```

---

## Running the Server

```bash
python main.py
```

The server starts on `http://localhost:8000` (or your configured port).

---

## Using the Web Interface

### 1. Upload a PDF
Navigate to `http://localhost:8000/` in your browser. You will see:
- A file picker — select a `.pdf` file.
- An optional **severity overrides** textarea (JSON format) to manually assign severities to specific phrases.
- A **Generate Report** button.

### 2. Monitor Progress
After submitting, you are redirected to a live progress page that shows real-time status updates:
- Extracting text from PDF
- Matching phrases from lexicon
- AI single-word analysis
- AI reasoning and suggestions
- AI discovery of additional issues
- Merging findings
- Generating PDF report

### 3. View Results
Once complete, a **View Results** button appears. The results page shows finding cards with:
- **Severity** badge (High / Medium / Low) with color-coding
- **Source** badge: `Lexicon` (CSV match), `AI-Discovered` (LLM-found), or `AI-Enhanced` (single-word AI analysis)
- **Page** number where the phrase was found
- **Exact Matched Phrase**
- **Suggested Rewrite**
- **Context** explaining the issue

### 4. Download the Report
A **Download PDF** button on the results page exports the full findings as a formatted PDF report saved in the `reports/` directory.

---

## Severity Overrides (Optional)

You can override the default severity for specific phrases by providing a JSON object in the form:
```json
{
  "machine learning": "Medium",
  "novel approach": "Low"
}
```
Valid severity values: `"High"`, `"Medium"`, `"Low"`.

---

## REST API Endpoints

The app also exposes a JSON API for programmatic use.

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI home page |
| `POST` | `/ui/analyze` | Submit a PDF via form (redirects to progress page) |
| `GET` | `/ui/progress/{job_id}` | Live progress tracking page (SSE-powered) |
| `GET` | `/ui/result/{job_id}` | HTML results page for a completed job |
| `GET` | `/progress-stream/{job_id}` | Server-Sent Events stream for progress updates |
| `POST` | `/analyze` | **API**: Submit a PDF, returns JSON (`AnalyzeResponse`) |
| `GET` | `/download/{job_id}` | Download the generated PDF report |

### API Example

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@your_paper.pdf" \
  -F 'severity_terms={"some phrase": "Low"}'
```

**Response schema:**
```json
{
  "job_id": "uuid",
  "download_url": "/download/{job_id}",
  "report_json": {
    "doc_title": "Paper Title",
    "summary": "Found X high, Y medium, Z low severity phrase(s).",
    "findings": [
      {
        "phrase": "tortured phrase",
        "severity": "High",
        "suggestion": "improved phrase",
        "page": 1,
        "context": "Explanation of the issue",
        "source": "CSV"
      }
    ]
  }
}
```

---

## How It Works

### Analysis Pipeline

```
PDF Upload
    │
    ▼
1. Text Extraction     (PyPDF2, per-page)
    │
    ▼
2. Lexicon Matching    (Aho-Corasick on multi-word phrases from CSV)
    │
    ▼
3. Single-Word AI      (Regex matching + GPT-5 / Claude contextual analysis)
    │
    ▼
4. AI Reasoning        (GPT-5 + Claude: rewrites, severity, context)
    │
    ▼
5. AI Discovery        (GPT-5 scans pages for additional unlisted issues)
    │
    ▼
6. Merge & Deduplicate (Priority-weighted merge of all findings)
    │
    ▼
7. PDF Report          (ReportLab-generated downloadable report)
```

### Key Modules

| File | Role |
|---|---|
| `main.py` | FastAPI app, routes, analysis orchestration |
| `config.py` | Settings (env-configurable) |
| `models.py` | Pydantic models: `Finding`, `ModelOutput`, `AnalyzeResponse` |
| `csv_loader.py` | Loads and normalizes the phrase lexicon CSV |
| `matcher_ac.py` | Aho-Corasick multi-word phrase matcher |
| `single_word_matcher.py` | Regex-based single-word detection + LLM analysis |
| `word_expansion.py` | Context extraction around single words |
| `llm_clients.py` | RouteLLM/Abacus API calls (GPT-5, Claude) |
| `merge.py` | Deduplication and weighted merge of findings |
| `reporting.py` | PDF report generation via ReportLab |
| `progress_tracker.py` | SSE-based real-time progress updates |
| `pdf_utils.py` | PDF text extraction helpers |

### The Lexicon CSV (`Tortured_Phrases_Lexicon_2.csv`)
This is the core database of known problematic phrases. It contains multi-word phrases and single words with associated metadata (severity, feedback/suggestion, weight, tags). The app loads it at startup and separates single words from multi-word phrases for different processing strategies.

---

## Operating Modes

| Mode | Condition | Behavior |
|---|---|---|
| **CLOUD** | `ABACUS_API_KEY` is set | Full AI analysis: phrase reasoning, discovery, single-word AI |
| **LOCAL** | No API key | Lexicon matching only; no AI suggestions or discovery |

The current mode is displayed at the bottom of the home page.

---

## Test Scripts

Several test scripts are included for development and debugging:

| Script | Purpose |
|---|---|
| `minimal_test.py` | Basic connectivity test |
| `simple_test.py` | Simple end-to-end test |
| `test_discovery.py` | Tests AI discovery pipeline |
| `test_single_word.py` | Tests single-word detection and LLM prompting |
| `test_improvements.py` | Tests suggestion quality improvements |
| `test_variations.py` | Tests phrase variation handling |
| `test_logging.py` | Tests logging output |
| `debug_requests.py` | Debug raw API requests |

Run any test with:
```bash
python test_single_word.py
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| Server shows **LOCAL** mode | Set `ABACUS_API_KEY` in `.env` or `config.py` |
| PDF upload fails | Ensure the file is a valid, non-empty PDF |
| Analysis times out | Increase `LLM_TIMEOUT_SECS` / `DISCOVERY_TIMEOUT_SECS` in config |
| No findings returned | The PDF text may not be extractable (scanned image PDFs are not supported) |
| Report not found after completion | Check the `reports/` directory exists and is writable |
| Slow analysis | Set `ENABLE_SINGLE_WORD_ANALYSIS=false` to skip the AI single-word step |
