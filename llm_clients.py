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

async def generate_smart_suggestion(phrase: str, context: str = "") -> str:
    """
    Generate intelligent suggestion by asking AI models for the improved phrase only.
    """
    # First try quick pattern-based fixes for common issues
    phrase_lower = phrase.lower().strip()
    
    # Quick fixes for very common patterns
    quick_fixes = {
        'in order to': 'to',
        'due to the fact that': 'because',
        'with regard to': 'regarding',
        'in the event that': 'if',
        'by means of': 'by',
    }
    
    for pattern, fix in quick_fixes.items():
        if pattern in phrase_lower:
            return phrase.replace(pattern, fix).replace(pattern.title(), fix.title())
    
    # For more complex issues, ask the AI model for just the improved phrase
    prompt = f"""You are an expert editor. The following phrase has been identified as problematic in academic writing:

PROBLEMATIC PHRASE: "{phrase}"

CONTEXT: {context if context else "This phrase needs improvement for academic writing standards."}

Your task: Provide ONLY the improved version of this exact phrase. Do not include explanations, quotes, or additional text - just return the corrected phrase that should replace the original.

Requirements:
- Keep the same meaning and intent
- Fix grammar, add missing articles, improve word order
- Make it sound natural and professional
- Return only the improved phrase, nothing else

Improved phrase:"""

    try:
        # Try route-llm for the suggestion using correct RouteLLM format
        headers = {
            "Authorization": f"Bearer {settings.ABACUS_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": "gpt-5",  # Use the standard route-llm model
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "stream": False
        }
        timeout = httpx.Timeout(15.0, connect=5.0)  # Shorter timeout for suggestions
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ABACUS_BASE_URL}/chat/completions", headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
            
            content = None
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content")
            
            if content:
                suggestion = content.strip().strip('"').strip("'").strip()
                # Basic validation - should be similar length and not just explanation
                if suggestion and len(suggestion) < len(phrase) * 3 and suggestion.lower() != phrase.lower():
                    print(f"[generate_smart_suggestion] AI suggestion for '{phrase}': '{suggestion}'")
                    return suggestion
                    
    except Exception as e:
        print(f"[generate_smart_suggestion] AI call failed for '{phrase}': {e}")
    
    # Fallback to pattern-based fixes if AI fails
    if 'serves' in phrase_lower and 'tool' in phrase_lower and 'as' not in phrase_lower:
        return phrase.replace('serves', 'serves as a', 1)
    
    if 'provides' in phrase_lower and ('overview' in phrase_lower or 'analysis' in phrase_lower) and ' a ' not in phrase_lower:
        return phrase.replace('provides', 'provides a', 1)
    
    if 'offers' in phrase_lower and ' a ' not in phrase_lower and ('perspective' in phrase_lower or 'solution' in phrase_lower):
        return phrase.replace('offers', 'offers a', 1)
    
    if 'represents' in phrase_lower and ' a ' not in phrase_lower and ('advance' in phrase_lower or 'improvement' in phrase_lower):
        return phrase.replace('represents', 'represents a', 1)
    
    # Last resort - simple suggestion
    if len(phrase.split()) <= 3:
        return f"[Improve: {phrase}]"
    else:
        return f"[Rewrite: {phrase}]"

async def _convert_new_format_to_model_output(parsed_data: dict) -> Optional[ModelOutput]:
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
                print(f"[llm_clients] Skipping finding with empty phrase")
                continue
                
            # Validate suggestion is not empty - this is now mandatory
            if not suggestion or suggestion.strip() == '':
                print(f"[llm_clients] Warning: Empty suggestion for phrase '{phrase}' - generating context-aware fallback")
                # Generate a more meaningful suggestion based on AI with context
                suggestion = await generate_smart_suggestion(phrase, context)
            
            # Validate context is not empty
            if not context or context.strip() == '':
                print(f"[llm_clients] Warning: Empty context for phrase '{phrase}' - generating fallback")
                context = f"Grammatical or stylistic issue identified in '{phrase}' that requires improvement for academic writing standards"
            
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
                context=context if context else "Issue identified requiring attention",
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
    lines.append("1) For EVERY matched phrase, provide detailed explanations and MANDATORY concrete rewrite suggestions.")
    lines.append("2) Identify additional high/medium/low severity phrases not in the list discovered from the snippets, with page if inferable.")
    lines.append("Rules:")
    lines.append('- Only include phrases <= 12 words.')
    lines.append('- Use severities: high, medium, low.')
    lines.append('- EVERY finding MUST have a suggestion - no exceptions!')
    lines.append('- Context must include: phrase type, specific issue, why it\'s problematic, and improvement rationale.')
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
    lines.append('          "suggestion": {"type": "string", "description": "MANDATORY concrete rewrite suggestion - NEVER leave empty"},')
    lines.append('          "context": {"type": "string", "description": "Detailed explanation including: phrase type, specific issue, why problematic, improvement rationale"}')
    lines.append('        }')
    lines.append('      }')
    lines.append('    }')
    lines.append('  },')
    lines.append('  "required": ["findings"],')
    lines.append('  "additionalProperties": false')
    lines.append('}')
    lines.append('')
    lines.append('MANDATORY REQUIREMENTS for each finding:')
    lines.append('- phrase: The exact problematic text you identified')
    lines.append('- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor')
    lines.append('- suggestion: MANDATORY concrete rewrite suggestion - provide specific improved wording for EVERY finding')
    lines.append('- context: Must include 4 parts: 1) Type of phrase issue 2) Specific problem 3) Why it\'s problematic 4) How suggestion improves it')
    lines.append('')
    lines.append('EXAMPLE CONTEXT FORMAT: "Tortured phrase - missing article. This phrase lacks grammatical completeness and sounds unnatural in academic writing. The suggestion adds the missing \'as\' to make it grammatically correct and more professional."')
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
                result = await _convert_new_format_to_model_output(parsed_data)
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

CRITICAL REQUIREMENTS:
- EVERY finding MUST have a concrete suggestion - no exceptions!
- Context must be detailed and include specific reasoning
- Focus on tortured phrases, AI fingerprints, awkward sentences, and grammar issues

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
          "suggestion": {{"type": "string", "description": "MANDATORY concrete rewrite suggestion - NEVER leave empty"}},
          "context": {{"type": "string", "description": "Detailed explanation including phrase type, specific issue, why problematic, improvement rationale"}}
        }}
      }}
    }}
  }},
  "required": ["findings"],
  "additionalProperties": false
}}

MANDATORY REQUIREMENTS for each finding:
- phrase: The exact problematic text you identified
- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor  
- suggestion: MANDATORY concrete rewrite suggestion with specific improved wording
- context: Must include 4 parts: 1) Type of issue (tortured phrase/AI fingerprint/grammar) 2) Specific problem 3) Why it's problematic 4) How suggestion improves it

EXAMPLE CONTEXT: "Tortured phrase - grammatical incompleteness. This phrase lacks proper article usage and sounds unnatural in academic writing. The suggestion adds the missing 'as' to create grammatically correct and professional phrasing."

Only include findings where you can provide BOTH a clear problem identification AND a concrete improvement suggestion.

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
                    
                    # Ensure suggestion is never empty
                    suggestion = finding.suggestion
                    if not suggestion or suggestion.strip() == '':
                        suggestion = f"Rewrite '{finding.phrase}' for better clarity and academic tone"
                        print(f"[query_models_discover]   Warning: Empty suggestion for '{finding.phrase}' - using fallback")
                    
                    # Create Finding object and set the correct page number from our context
                    new_finding = Finding(
                        phrase=finding.phrase.lower(),  # normalize to lowercase for dedup/merge
                        severity=finding.severity,
                        suggestion=suggestion,
                        page=page_num,  # Use the page number from our loop context, not from AI
                        start_char=finding.start_char,
                        end_char=finding.end_char,
                        context=finding.context or "Issue identified requiring attention",
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