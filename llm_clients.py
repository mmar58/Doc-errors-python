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
    Convert simplified JSON format to ModelOutput for backward compatibility.
    """
    try:
        findings_list = []
        findings_data = parsed_data.get('findings', [])
        
        for finding_data in findings_data:
            # Extract fields from simplified format
            phrase = finding_data.get('phrase', '')
            severity = finding_data.get('severity', 'medium')
            suggestion = finding_data.get('suggestion', '')
            context = finding_data.get('context', '')
            
            if not phrase:
                continue
            
            # Convert severity to title case for compatibility
            severity_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
            severity_title = severity_map.get(severity.lower(), 'Medium')
            
            finding = Finding(
                phrase=phrase,
                severity=severity_title,
                suggestion=suggestion,
                page=1,  # Will be updated with actual page number from context
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
        print(f"[llm_clients] Error converting simplified format: {e}")
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
    lines.append('        "required": ["phrase","severity","suggestion","context"],')
    lines.append('        "properties": {')
    lines.append('          "phrase": {"type": "string", "description": "The exact problematic text you identified"},')
    lines.append('          "severity": {"enum": ["low","medium","high"]},')
    lines.append('          "suggestion": {"type": "string", "description": "Your concrete rewrite suggestion"},')
    lines.append('          "context": {"type": "string", "description": "Detailed explanation of why this is problematic and context information"}')
    lines.append('        }')
    lines.append('      }')
    lines.append('    }')
    lines.append('  },')
    lines.append('  "required": ["findings"],')
    lines.append('  "additionalProperties": false')
    lines.append('}')
    lines.append('')
    lines.append('For each finding:')
    lines.append('- phrase: The exact problematic text you identified')
    lines.append('- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor')
    lines.append('- suggestion: Your concrete rewrite suggestion')
    lines.append('- context: Detailed explanation of why this is problematic, what type of issue it is, and any relevant context')
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
    print(f"[call_routellm] === CALLING AI MODEL: {model_name} ===")
    if not settings.ABACUS_API_KEY:
        print(f"[call_routellm] No API key available for {model_name}")
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
        print(f"[call_routellm] Sending request to {model_name}...")
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ABACUS_BASE_URL}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            print(f"[call_routellm] Raw HTTP response from {model_name}: {json.dumps(data, indent=2)[:1000]}...")
            
            # RouteLLM returns a 'choices' style; try to parse content
            content = None
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    content = choices[0].get("message", {}).get("content")
                if not content and "output" in data:
                    content = data["output"]
            if not content:
                print(f"[call_routellm] ✗ No content found in response from {model_name}")
                return None
                
            print(f"[call_routellm] Raw AI content from {model_name}: {content[:800]}...")
            try:
                # Parse new JSON format
                parsed_data = json.loads(content)
                print(f"[call_routellm] Parsed JSON from {model_name}: {json.dumps(parsed_data, indent=2)}")
                result = _convert_new_format_to_model_output(parsed_data)
                if result:
                    print(f"[call_routellm] ✓ {model_name} SUCCESS: Converted to {len(result.findings)} findings")
                    for i, finding in enumerate(result.findings[:3]):  # Show first 3
                        print(f"[call_routellm]   {model_name} Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}' (severity: {finding.severity}, source: {finding.source})")
                    if len(result.findings) > 3:
                        print(f"[call_routellm]   {model_name} ... and {len(result.findings) - 3} more findings")
                else:
                    print(f"[call_routellm] ✗ {model_name} FAILED: Conversion returned None")
                return result
            except Exception as e:
                print(f"[call_routellm] ✗ {model_name} JSON PARSE ERROR: {e}")
                print(f"[call_routellm] Raw content that failed to parse: {content}")
                return None
    except Exception as e:
        print(f"[call_routellm] ✗ {model_name} REQUEST FAILED: {e}")
        return None

async def query_models(doc_title: str,
                       unique_matches_sorted: List[Tuple[str, str, int]],
                       snippets: Dict[str, str],
                       severity_counts: Dict[str, int]) -> List[ModelOutput]:
    print(f"[query_models] === STARTING MODEL QUERY FOR DOCUMENT: {doc_title} ===")
    print(f"[query_models] Processing {len(unique_matches_sorted)} unique matches")
    prompt = build_prompt(doc_title, unique_matches_sorted, snippets, severity_counts, settings.MAX_PROMPT_CHARS)
    print(f"[query_models] Built prompt with {len(prompt)} characters")
    tasks = [
        call_routellm(settings.ROUTELLM_MODEL_GPT5, prompt),
        call_routellm(settings.ROUTELLM_MODEL_CLAUDE, prompt),
    ]
    print(f"[query_models] Running {len(tasks)} models in parallel: {settings.ROUTELLM_MODEL_GPT5}, {settings.ROUTELLM_MODEL_CLAUDE}")
    results = await asyncio.gather(*tasks, return_exceptions=True)
    outputs: List[ModelOutput] = []
    for i, r in enumerate(results):
        model_name = [settings.ROUTELLM_MODEL_GPT5, settings.ROUTELLM_MODEL_CLAUDE][i]
        if isinstance(r, ModelOutput):
            print(f"[query_models] ✓ {model_name} returned {len(r.findings)} findings")
            outputs.append(r)
        elif isinstance(r, Exception):
            print(f"[query_models] ✗ {model_name} EXCEPTION: {r}")
        else:
            print(f"[query_models] ✗ {model_name} returned unexpected type: {type(r)}")
    print(f"[query_models] === COMPLETED: {len(outputs)} successful model responses out of {len(tasks)} ===")
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
    print(f"[query_models_discover] === STARTING DISCOVERY ANALYSIS ===")
    print(f"[query_models_discover] Document: {doc_title_hint}")
    print(f"[query_models_discover] Pages to analyze: {len(pages)}")
    print(f"[query_models_discover] Max findings per page: {max_findings_per_page}")
    print(f"[query_models_discover] Discovery model: {settings.DISCOVERY_MODEL_NAME}")
    
    findings: List[Finding] = []

    if not pages:
        print(f"[query_models_discover] No pages provided, returning empty findings")
        return findings

    # Simple per-page loop to keep token sizes bounded.
    for page_num, page_text in pages:
        if not page_text or not page_text.strip():
            print(f"[query_models_discover] Skipping empty page {page_num}")
            continue

        print(f"[query_models_discover] Analyzing page {page_num} ({len(page_text)} chars)")
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
        "required": ["phrase","severity","suggestion","context"],
        "properties": {{
          "phrase": {{"type": "string", "description": "The exact problematic text you identified"}},
          "severity": {{"enum": ["low","medium","high"]}},
          "suggestion": {{"type": "string", "description": "Your concrete rewrite suggestion"}},
          "context": {{"type": "string", "description": "Detailed explanation of why this is problematic and context information"}}
        }}
      }}
    }}
  }},
  "required": ["findings"],
  "additionalProperties": false
}}

For each finding:
- phrase: The exact problematic text you identified
- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor
- suggestion: Your concrete rewrite suggestion
- context: Detailed explanation of why this is problematic, what type of issue it is, and any relevant context

Page {page_num} text:
\"\"\"{page_text[:6000]}\"\"\"  # hard cap to avoid oversized prompts
"""
        try:
            # Use our existing call_routellm function instead of route_llm_call
            result = await call_routellm(settings.DISCOVERY_MODEL_NAME, prompt)
            if result and result.findings:
                print(f"[query_models_discover] ✓ Page {page_num}: Found {len(result.findings)} findings from {settings.DISCOVERY_MODEL_NAME}")
                for i, finding in enumerate(result.findings):
                    print(f"[query_models_discover]   Page {page_num} Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}' (severity: {finding.severity})")
                    # Create Finding object and set the correct page number from our context
                    new_finding = Finding(
                        phrase=finding.phrase.lower(),  # normalize to lowercase for dedup/merge
                        severity=finding.severity,
                        suggestion=finding.suggestion or None,
                        page=page_num,  # Use the page number from our loop context, not from AI
                        start_char=finding.start_char,
                        end_char=finding.end_char,
                        context=finding.context or None,
                        source="LLM",
                    )
                    findings.append(new_finding)
            else:
                print(f"[query_models_discover] ✗ Page {page_num}: No findings returned from {settings.DISCOVERY_MODEL_NAME}")
        except Exception as e:
            print(f"[query_models_discover] ✗ Page {page_num} FAILED: {type(e).__name__}: {e}")
            continue

    print(f"[query_models_discover] === DISCOVERY COMPLETED: {len(findings)} total findings across {len(pages)} pages ===")
    return findings