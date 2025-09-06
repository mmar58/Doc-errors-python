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

def _convert_new_format_to_model_output(parsed_data: dict) -> Optional[ModelOutput]:
    """
    Convert new JSON format to ModelOutput for backward compatibility.
    """
    try:
        findings_list = []
        findings_data = parsed_data.get('findings', [])
        
        for finding_data in findings_data:
            # Extract fields from new format
            category = finding_data.get('category', 'tortured_phrase')
            page = finding_data.get('page', 1)
            section = finding_data.get('section', '')
            span = finding_data.get('span', '')
            exact = finding_data.get('exact', '')
            suggestion = finding_data.get('suggestion', '')
            rationale = finding_data.get('rationale', '')
            severity = finding_data.get('severity', 'medium')
            
            # Convert severity to title case for compatibility
            severity_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
            severity_title = severity_map.get(severity.lower(), 'Medium')
            
            # Create context from section and rationale
            context_parts = []
            if section:
                context_parts.append(f"Section: {section}")
            if rationale:
                context_parts.append(f"Issue: {rationale}")
            if span:
                context_parts.append(f"Span: {span}")
            
            context = " | ".join(context_parts) if context_parts else rationale
            
            finding = Finding(
                phrase=exact,
                severity=severity_title,
                suggestion=suggestion,
                page=page,
                start_char=None,
                end_char=None,
                context=context,
                source="LLM"
            )
            findings_list.append(finding)
        
        return ModelOutput(
            doc_title=parsed_data.get('doc_title'),
            summary=parsed_data.get('summary'),
            findings=findings_list
        )
        
    except Exception as e:
        print(f"[llm_clients] Error converting new format: {e}")
        return None

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
    lines.append('- Use severities: high, medium, low.')
    lines.append('- Output must be a JSON object matching this exact schema:')
    lines.append('{')
    lines.append('  "type": "object",')
    lines.append('  "properties": {')
    lines.append('    "findings": {')
    lines.append('      "type": "array",')
    lines.append('      "items": {')
    lines.append('        "type": "object",')
    lines.append('        "required": ["category","page","section","span","exact","suggestion","rationale","severity"],')
    lines.append('        "properties": {')
    lines.append('          "category": {"enum": ["tortured_phrase","ai_fingerprint","awkward_sentence","grammar_formatting"]},')
    lines.append('          "page": {"type": "integer", "minimum": 1},')
    lines.append('          "section": {"type": "string"},')
    lines.append('          "span": {"type": "string", "description": "span_id from sentences.jsonl"},')
    lines.append('          "exact": {"type": "string"},')
    lines.append('          "suggestion": {"type": "string"},')
    lines.append('          "rationale": {"type": "string", "maxLength": 240},')
    lines.append('          "severity": {"enum": ["low","medium","high"]}')
    lines.append('        }')
    lines.append('      }')
    lines.append('    }')
    lines.append('  },')
    lines.append('  "required": ["findings"],')
    lines.append('  "additionalProperties": false')
    lines.append('}')
    lines.append('For each finding:')
    lines.append('- category: Use "tortured_phrase" for problematic academic phrases')
    lines.append('- page: The page number where the issue was found')
    lines.append('- section: Brief description of document section (e.g., "Introduction", "Methods")')
    lines.append('- span: Use format "page_X_span_Y" where X is page number and Y is a unique span ID')
    lines.append('- exact: The exact problematic text you identified')
    lines.append('- suggestion: Your concrete rewrite suggestion')
    lines.append('- rationale: Brief explanation of why this is problematic (max 240 chars)')
    lines.append('- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor')
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
                # Parse new JSON format
                parsed_data = json.loads(content)
                return _convert_new_format_to_model_output(parsed_data)
            except Exception as e:
                print(f"[llm_clients] Error parsing LLM response: {e}")
                return None
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

        prompt = f"""You are auditing scientific/technical prose for "tortured phrases" or risky wording that should be rewritten.
From the provided page text, extract up to {max_findings_per_page} findings.

Output must be a JSON object matching this exact schema:
{{
  "type": "object",
  "properties": {{
    "findings": {{
      "type": "array",
      "items": {{
        "type": "object",
        "required": ["category","page","section","span","exact","suggestion","rationale","severity"],
        "properties": {{
          "category": {{"enum": ["tortured_phrase","ai_fingerprint","awkward_sentence","grammar_formatting"]}},
          "page": {{"type": "integer", "minimum": 1}},
          "section": {{"type": "string"}},
          "span": {{"type": "string", "description": "span_id from sentences.jsonl"}},
          "exact": {{"type": "string"}},
          "suggestion": {{"type": "string"}},
          "rationale": {{"type": "string", "maxLength": 240}},
          "severity": {{"enum": ["low","medium","high"]}}
        }}
      }}
    }}
  }},
  "required": ["findings"],
  "additionalProperties": false
}}

For each finding:
- category: Use "tortured_phrase" for problematic academic phrases
- page: {page_num}
- section: Brief description of document section
- span: Use format "page_{page_num}_span_X" where X is a unique span ID
- exact: The exact problematic text you identified
- suggestion: Your concrete rewrite suggestion
- rationale: Brief explanation of why this is problematic (max 240 chars)
- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor

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