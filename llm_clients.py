import json
import asyncio
from typing import Dict, List, Tuple, Optional, Any
import httpx
from models import ModelOutput, Finding, SEV_INT_TO_TEXT, SEV_TEXT_TO_INT, normalize_sev_text
from config import settings

def truncate_to_chars(s: str, max_chars: int) -> str:
    if len(s) <= max_chars:
        return s
    return s[:max_chars]

def build_prompt(doc_title: str,
                 unique_matches_sorted: List[Tuple[str, str, int]],
                 snippets: Dict[str, str],
                 severity_counts: Dict[str, int],
                 max_chars: int) -> str:
    """
    unique_matches_sorted: list of (phrase, severity_text, count)
    snippets: phrase -> compact snippet
    severity_counts: {"High":x,"Medium":y,"Low":z}
    """
    lines = []
    lines.append("You are an expert editor. Only respond with strict JSON per schema. Do not include extra keys.")
    lines.append(f"Document title: {doc_title}" if doc_title else "Document title: (unknown)")
    lines.append("Matched phrases (with applied severities this request):")
    for phrase, sev, cnt in unique_matches_sorted:
        lines.append(f"- {phrase} | severity={sev} | count={cnt}")
    lines.append("Compact context snippets (first occurrence per phrase):")
    for phrase, snippet in snippets.items():
        lines.append(f"[{phrase}] ... {snippet} ...")
    lines.append("Summary of counts by severity:")
    lines.append(f"High={severity_counts.get('High',0)}, Medium={severity_counts.get('Medium',0)}, Low={severity_counts.get('Low',0)}")
    lines.append("Task:")
    lines.append("1) For the listed matched phrases, provide brief explanations and concrete rewrite suggestions.")
    lines.append("2) Identify additional high/medium/low severity phrases not in the list discovered from the snippets, with page if inferable.")
    lines.append("Rules:")
    lines.append('- Only include phrases <= 12 words.')
    lines.append('- If positions are unknown, set start_char and end_char to null.')
    lines.append('- Use severities: High, Medium, Low.')
    lines.append('- Output must be a JSON object with keys: doc_title (optional), summary (optional), findings (array).')
    lines.append('- findings items must have: phrase, severity, suggestion, page (1-based integer), start_char (int|null), end_char (int|null), context (string|null).')
    text = "\n".join(lines)
    return truncate_to_chars(text, max_chars)

async def call_routellm(model_name: str, prompt: str) -> Optional[ModelOutput]:
    if not settings.ABACUS_API_KEY:
        return None
    headers = {
        "Authorization": f"Bearer {settings.ABACUS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "input": prompt,
        "response_format": {"type": "json_object"},
    }
    timeout = httpx.Timeout(settings.LLM_TIMEOUT_SECS, connect=5.0)
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ABACUS_BASE_URL}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            # RouteLLM returns a 'choices' style; try to parse content
            content = None
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    content = choices[0].get("message", {}).get("content")
                if not content and "output" in data:
                    content = data["output"]
            if not content:
                return None
            try:
                return ModelOutput.model_validate_json(content)
            except Exception:
                # Try to coerce if minor issues
                return ModelOutput(**json.loads(content))
    except Exception:
        return None

async def query_models(doc_title: str,
                       unique_matches_sorted: List[Tuple[str, str, int]],
                       snippets: Dict[str, str],
                       severity_counts: Dict[str, int]) -> List[ModelOutput]:
    prompt = build_prompt(doc_title, unique_matches_sorted, snippets, severity_counts, settings.MAX_PROMPT_CHARS)
    tasks = [
        call_routellm(settings.ROUTELLM_MODEL_GPT5, prompt),
        call_routellm(settings.ROUTELLM_MODEL_CLAUDE, prompt),
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs: List[ModelOutput] = []
    for r in results:
        if isinstance(r, ModelOutput):
            outputs.append(r)
    return outputs
async def query_models_discover(
    doc_title_hint: Optional[str],
    pages: List[Tuple[int, str]],  # list of (page_number_1_based, page_text)
    max_findings_per_page: int,
) -> List[Finding]:
    """
    Ask the model to identify risky phrases/wording with severity and suggested rewrites.
    Returns Finding list with source="LLM".
    """
    findings: List[Finding] = []

    if not pages:
        return findings

    # Simple per-page loop to keep token sizes bounded.
    for page_num, page_text in pages:
        if not page_text or not page_text.strip():
            continue

        prompt = f"""
You are auditing scientific/technical prose for "tortured phrases" or risky wording that should be rewritten.
From the provided page text, extract up to {max_findings_per_page} findings.
Return a clean JSON array with objects having exactly these keys:
- phrase: the exact risky wording span as appears
- severity: High/Medium/Low (High if strongly problematic, Medium otherwise, Low for minor)
- suggestion: a concise rewrite
- context: short context line (<= 240 chars) that includes the phrase

Do not include any keys other than those four. Use the page text verbatim for the phrase.

Page {page_num} text:
\"\"\"{page_text[:6000]}\"\"\"  # hard cap to avoid oversized prompts
"""
        try:
            # Use our existing call_routellm function instead of route_llm_call
            result = await call_routellm(settings.DISCOVERY_MODEL_NAME, prompt)
            if result and result.findings:
                for finding in result.findings:
                    findings.append(Finding(
                        phrase=finding.phrase.lower(),  # normalize to lowercase for dedup/merge
                        severity=finding.severity,
                        suggestion=finding.suggestion or None,
                        page=page_num,
                        start_char=finding.start_char,
                        end_char=finding.end_char,
                        context=finding.context or None,
                        source="LLM",
                    ))
        except Exception as e:
            print(f"[discover] page={page_num} failed: {type(e).__name__}: {e}")
            continue

    return findings