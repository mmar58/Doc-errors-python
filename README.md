# 📄 PDF Severity Phrase Analyzer

> **Detect and fix "tortured phrases" in academic PDFs using AI.**

A FastAPI web application that scans academic PDF documents for problematic, unnatural, or grammatically flawed expressions ("tortured phrases") — common in machine-translated or low-quality papers. It combines a curated phrase lexicon, high-speed pattern matching, and dual AI models (GPT-5 & Claude) to surface issues and suggest rewrites.

---

## ✨ Features

- 🔍 **Lexicon matching** — Aho-Corasick search across a 200k+ entry phrase database
- 🤖 **Dual AI analysis** — GPT-5 and Claude run in parallel to reason about severity and rewrites
- 🧠 **AI phrase discovery** — Finds additional issues beyond the lexicon
- 🔤 **Single-word AI analysis** — Contextual analysis of single-word lexicon entries
- 📊 **Progress tracking** — Real-time Server-Sent Events progress bar in the browser
- 📥 **PDF report export** — Download a formatted PDF of all findings
- 🌐 **REST API** — Programmatic access alongside the web UI

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Set your API key for AI features
echo "ABACUS_API_KEY=your_key_here" > .env

# 3. Run the server
python main.py
```

Open **http://localhost:8000** in your browser, upload a PDF, and click **Generate Report**.

> Without an API key the app runs in **LOCAL** mode — lexicon matching only, no AI analysis.

---

## 📁 Project Structure

```
├── main.py                        # FastAPI app & analysis orchestration
├── config.py                      # Settings (env-configurable)
├── models.py                      # Pydantic data models
├── csv_loader.py                  # Phrase lexicon loader
├── matcher_ac.py                  # Aho-Corasick phrase matcher
├── single_word_matcher.py         # Single-word AI detection
├── word_expansion.py              # Context extraction for single words
├── llm_clients.py                 # GPT-5 / Claude API client
├── merge.py                       # Finding deduplication & merging
├── reporting.py                   # PDF report generator
├── progress_tracker.py            # SSE-based progress updates
├── pdf_utils.py                   # PDF text extraction
├── Tortured_Phrases_Lexicon_2.csv # Phrase database
├── reports/                       # Generated report PDFs (auto-created)
├── requirements.txt
├── instructions.md                # Full usage documentation
└── SINGLE_WORD_ENHANCEMENT.md     # Feature implementation notes
```

---

## ⚙️ Key Configuration

| Variable | Default | Description |
|---|---|---|
| `ABACUS_API_KEY` | — | Abacus/RouteLLM API key (enables AI mode) |
| `PORT` | `8000` | Server port |
| `ENABLE_SINGLE_WORD_ANALYSIS` | `true` | AI analysis of single-word entries |
| `DISCOVERY_ENABLED` | `true` | AI discovery of unlisted phrases |

Set via `.env` file or environment variables. See [`instructions.md`](instructions.md) for the full reference.

---

## 🔗 API

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Web UI |
| `POST` | `/analyze` | Submit PDF, get JSON results |
| `GET` | `/download/{job_id}` | Download PDF report |

```bash
curl -X POST http://localhost:8000/analyze \
  -F "file=@paper.pdf"
```

---

## 📋 Requirements

- Python 3.10+
- Dependencies: `fastapi`, `uvicorn`, `PyPDF2`, `reportlab`, `pyahocorasick`, `httpx`, `pydantic-settings`, `python-multipart`

---

## 📖 Documentation

See **[instructions.md](instructions.md)** for full setup, configuration, API reference, and troubleshooting.
