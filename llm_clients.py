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

async def test_routellm_canary() -> bool:
    """
    Test canary to verify RouteLLM API connectivity and response parsing.
    Returns True if the basic API call works, False otherwise.
    """
    print(f"[test_routellm_canary] === TESTING ROUTELLM API CONNECTIVITY ===")
    
    if not settings.ABACUS_API_KEY:
        print(f"[test_routellm_canary] ✗ No API key available")
        return False
        
    headers = {
        "Authorization": f"Bearer {settings.ABACUS_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Simple test payload with JSON response format
    test_payload = {
        "model": "gpt-5",
        "messages": [
            {
                "role": "user",
                "content": "Respond with a simple JSON object containing just {\"test\": \"success\", \"status\": \"ok\"}. Nothing else."
            }
        ],
        "response_format": {"type": "json_object"},
        "stream": False
    }
    
    timeout = httpx.Timeout(10.0, connect=5.0)
    
    try:
        print(f"[test_routellm_canary] Sending test request...")
        print(f"[test_routellm_canary] Test payload: {json.dumps(test_payload, indent=2)}")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ABACUS_BASE_URL}/chat/completions", headers=headers, json=test_payload)
            
            print(f"[test_routellm_canary] Response status: {resp.status_code}")
            print(f"[test_routellm_canary] Response headers: {dict(resp.headers)}")
            
            resp.raise_for_status()
            data = resp.json()
            
            print(f"[test_routellm_canary] === RAW CANARY RESPONSE ===")
            print(json.dumps(data, indent=2))
            print(f"[test_routellm_canary] === END RAW CANARY RESPONSE ===")
            
            # Try to extract content using the same logic as call_routellm
            content = None
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices:
                    content = choices[0].get("message", {}).get("content")
                if not content and "output" in data:
                    content = data["output"]
                    
            print(f"[test_routellm_canary] Extracted content: '{content}'")
            
            if content:
                try:
                    parsed = json.loads(content)
                    print(f"[test_routellm_canary] ✓ Successfully parsed JSON: {parsed}")
                    return True
                except json.JSONDecodeError as e:
                    print(f"[test_routellm_canary] ✗ JSON parse failed: {e}")
                    print(f"[test_routellm_canary] Raw content: {content}")
                    return False
            else:
                print(f"[test_routellm_canary] ✗ No content extracted from response")
                return False
                
    except Exception as e:
        print(f"[test_routellm_canary] ✗ Test request failed: {type(e).__name__}: {e}")
        return False

async def test_full_pipeline() -> None:
    """
    Test the full pipeline with enhanced logging to verify all components work.
    """
    print(f"[test_full_pipeline] === TESTING FULL PIPELINE WITH ENHANCED LOGGING ===")
    
    # Step 1: Test basic API connectivity
    canary_result = await test_routellm_canary()
    print(f"[test_full_pipeline] Canary test result: {'✓ PASS' if canary_result else '✗ FAIL'}")
    
    if not canary_result:
        print(f"[test_full_pipeline] ✗ Canary failed, skipping further tests")
        return
    
    # Step 2: Test smart suggestion with detailed logging
    print(f"[test_full_pipeline] Testing generate_smart_suggestion...")
    test_phrase = "serves powerful tool"
    test_context = "This phrase appears in academic writing and needs improvement."
    suggestion = await generate_smart_suggestion(test_phrase, test_context)
    print(f"[test_full_pipeline] Smart suggestion result: '{test_phrase}' -> '{suggestion}'")
    
    # Step 3: Test call_routellm with a simple finding request
    print(f"[test_full_pipeline] Testing call_routellm with simple prompt...")
    simple_prompt = """You are an expert editor. Respond with strict JSON per schema.

Find issues in this text: "The system serves powerful tool for analysis."

Output must be a JSON object:
{
  "findings": [
    {
      "phrase": "serves powerful tool",
      "severity": "medium", 
      "suggestion": "serves as a powerful tool",
      "context": "Missing preposition 'as' makes this phrase grammatically incomplete."
    }
  ]
}"""
    
    result = await call_routellm("gpt-5", simple_prompt)
    if result:
        print(f"[test_full_pipeline] ✓ call_routellm succeeded with {len(result.findings)} findings")
        for i, finding in enumerate(result.findings):
            print(f"[test_full_pipeline]   Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}'")
    else:
        print(f"[test_full_pipeline] ✗ call_routellm failed")
    
    print(f"[test_full_pipeline] === TESTING COMPLETED ===")

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
            
            print(f"[generate_smart_suggestion] ===== RAW RESPONSE FOR '{phrase}' =====")
            print(json.dumps(data, indent=2))
            print(f"[generate_smart_suggestion] ===== END RAW RESPONSE =====")
            
            content = None
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                    print(f"[generate_smart_suggestion] Extracted content for '{phrase}': '{content}'")
            
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
    Convert AI JSON response to ModelOutput for backward compatibility.
    Expected format: {"findings": [{"phrase": "...", "severity": "...", "suggestion": "...", "context": "..."}]}
    """
    try:
        findings_list = []
        findings_data = parsed_data.get('findings', [])
        
        print(f"[_convert_new_format_to_model_output] Processing {len(findings_data)} findings from AI response")
        
        for i, finding_data in enumerate(findings_data):
            # Extract required fields
            phrase = finding_data.get('phrase', '').strip()
            severity = finding_data.get('severity', 'medium').lower()
            suggestion = finding_data.get('suggestion', '').strip()
            context = finding_data.get('context', '').strip()
            
            # Validate required fields
            if not phrase:
                print(f"[_convert_new_format_to_model_output] Skipping finding {i+1}: empty phrase")
                continue
                
            # Ensure suggestion is never empty (this should not happen with our new prompts)
            if not suggestion:
                print(f"[_convert_new_format_to_model_output] Warning: Empty suggestion for '{phrase}' - generating AI fallback")
                suggestion = await generate_smart_suggestion(phrase, context)
                
            # Ensure context is descriptive
            if not context:
                print(f"[_convert_new_format_to_model_output] Warning: Empty context for '{phrase}' - generating fallback")
                context = f"Issue identified in phrase '{phrase}' requiring improvement for academic writing standards"
            
            # Convert severity to title case for compatibility with existing code
            severity_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
            severity_title = severity_map.get(severity, 'Medium')
            
            # Create Finding object
            finding = Finding(
                phrase=phrase,
                severity=severity_title,
                suggestion=suggestion,
                page=1,  # Will be updated with actual page number during processing
                start_char=None,
                end_char=None,
                context=context,
                source="LLM"
            )
            findings_list.append(finding)
            
            print(f"[_convert_new_format_to_model_output] ✓ Converted finding {i+1}: '{phrase}' -> '{suggestion}' (severity: {severity_title})")
        
        print(f"[_convert_new_format_to_model_output] ✓ Successfully converted {len(findings_list)} findings")
        return ModelOutput(
            doc_title=parsed_data.get('doc_title'),
            summary=parsed_data.get('summary'),
            findings=findings_list
        )
        
    except Exception as e:
        print(f"[_convert_new_format_to_model_output] ✗ Error converting response: {type(e).__name__}: {e}")
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
    lines.append('- In suggestions, provide ONLY the improved phrase - no explanatory text!')
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
    lines.append('          "suggestion": {"type": "string", "description": "ONLY the improved phrase - no explanations or instructions"},')
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
    lines.append('- suggestion: ONLY the improved phrase that replaces the problematic one - NO explanatory text')
    lines.append('- context: Must include 4 parts: 1) Type of phrase issue 2) Specific problem 3) Why it\'s problematic 4) How suggestion improves it')
    lines.append('')
    lines.append('EXAMPLE OUTPUT:')
    lines.append('- phrase: "serves powerful tool"')
    lines.append('- suggestion: "serves as a powerful tool"')
    lines.append('- context: "Tortured phrase - missing preposition. This phrase lacks grammatical completeness and sounds unnatural in academic writing. The suggestion adds the missing \'as\' to make it grammatically correct and more professional."')
    text = "\n".join(lines)
    return truncate_to_chars(text, max_chars)

async def call_routellm(model_name: str, prompt: str) -> Optional[ModelOutput]:
    """Call RouteLLM with standard timeout"""
    return await call_routellm_with_timeout(model_name, prompt, settings.LLM_TIMEOUT_SECS)

async def call_routellm_with_timeout(model_name: str, prompt: str, timeout_secs: float) -> Optional[ModelOutput]:
    print(f"[call_routellm_with_timeout] === CALLING AI MODEL: {model_name} (timeout: {timeout_secs}s) ===")
    if not settings.ABACUS_API_KEY:
        print(f"[call_routellm_with_timeout] No API key available for {model_name}")
        return None
    headers = {
        "Authorization": f"Bearer {settings.ABACUS_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "response_format": {"type": "json_object"},
        "stream": False
    }
    timeout = httpx.Timeout(timeout_secs, connect=5.0)  # Use custom timeout
    try:
        print(f"[call_routellm] === SENDING REQUEST TO {model_name} ===")
        print(f"[call_routellm] Payload size: {len(json.dumps(payload))} characters")
        print(f"[call_routellm] Prompt length: {len(payload['messages'][0]['content'])} characters")
        print(f"[call_routellm] Full payload: {json.dumps(payload, indent=2)[:2000]}{'...(truncated)' if len(json.dumps(payload)) > 2000 else ''}")
        print(f"[call_routellm] Headers: {headers}")
        print(f"[call_routellm] URL: {settings.ABACUS_BASE_URL}/chat/completions")
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ABACUS_BASE_URL}/chat/completions", headers=headers, json=payload)
            
            print(f"[call_routellm] Response status code: {resp.status_code}")
            print(f"[call_routellm] Response headers: {dict(resp.headers)}")
            
            resp.raise_for_status()
            data = resp.json()
            print(f"[call_routellm] === FULL RAW HTTP RESPONSE FROM {model_name} ===")
            print(json.dumps(data, indent=2))
            print(f"[call_routellm] === END RAW HTTP RESPONSE FROM {model_name} ===")
            
            # RouteLLM returns a 'choices' style; try to parse content
            content = None
            if isinstance(data, dict):
                choices = data.get("choices") or []
                if choices and len(choices) > 0:
                    message = choices[0].get("message", {})
                    content = message.get("content")
                # Fallback for alternative response format
                if not content and "output" in data:
                    content = data["output"]
                    
            if not content:
                print(f"[call_routellm] ✗ No content found in response from {model_name}")
                print(f"[call_routellm] Response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}")
                return None
                
            print(f"[call_routellm] ✓ Successfully extracted content from {model_name} ({len(content)} chars)")
            try:
                # Parse the JSON content from AI
                parsed_data = json.loads(content)
                print(f"[call_routellm] ===== COMPLETE JSON RESPONSE FROM {model_name} =====")
                print(json.dumps(parsed_data, indent=2))
                print(f"[call_routellm] ===== END JSON RESPONSE FROM {model_name} =====")
                
                # Validate the expected structure
                if not isinstance(parsed_data, dict):
                    print(f"[call_routellm] ✗ {model_name} returned non-dict: {type(parsed_data)}")
                    return None
                    
                if "findings" not in parsed_data:
                    print(f"[call_routellm] ✗ {model_name} missing 'findings' key. Keys: {list(parsed_data.keys())}")
                    return None
                
                findings_list = parsed_data["findings"]
                if not isinstance(findings_list, list):
                    print(f"[call_routellm] ✗ {model_name} 'findings' is not a list: {type(findings_list)}")
                    return None
                
                # Show individual findings with suggestions for debugging
                print(f"[call_routellm] ✓ {model_name} returned {len(findings_list)} findings:")
                for i, finding in enumerate(findings_list):
                    phrase = finding.get("phrase", "N/A")
                    suggestion = finding.get("suggestion", "N/A") 
                    severity = finding.get("severity", "N/A")
                    context = finding.get("context", "N/A")
                    print(f"[call_routellm]   {model_name} Finding {i+1}:")
                    print(f"[call_routellm]     phrase: '{phrase}'")
                    print(f"[call_routellm]     suggestion: '{suggestion}'")
                    print(f"[call_routellm]     severity: '{severity}'")
                    print(f"[call_routellm]     context: '{context[:100]}...' " if len(context) > 100 else f"[call_routellm]     context: '{context}'")
                
                # Convert to our internal format
                result = await _convert_new_format_to_model_output(parsed_data)
                if result:
                    print(f"[call_routellm] ✓ {model_name} SUCCESS: Converted to {len(result.findings)} findings")
                    return result
                else:
                    print(f"[call_routellm] ✗ {model_name} FAILED: Conversion returned None")
                    return None
                    
            except json.JSONDecodeError as e:
                print(f"[call_routellm] ✗ {model_name} JSON PARSE ERROR: {e}")
                print(f"[call_routellm] Raw content that failed to parse (first 500 chars): {content[:500]}")
                return None
            except Exception as e:
                print(f"[call_routellm] ✗ {model_name} UNEXPECTED ERROR: {type(e).__name__}: {e}")
                return None
    except httpx.HTTPStatusError as e:
        print(f"[call_routellm] ✗ {model_name} HTTP ERROR: {e.response.status_code}")
        print(f"[call_routellm] Response headers: {dict(e.response.headers)}")
        try:
            error_body = e.response.json()
            print(f"[call_routellm] Error response body: {json.dumps(error_body, indent=2)}")
        except:
            print(f"[call_routellm] Error response text: {e.response.text}")
        return None
    except httpx.TimeoutException as e:
        print(f"[call_routellm] ✗ {model_name} TIMEOUT: {e}")
        return None
    except httpx.RequestError as e:
        print(f"[call_routellm] ✗ {model_name} CONNECTION ERROR: {type(e).__name__}: {e}")
        return None
    except Exception as e:
        print(f"[call_routellm] ✗ {model_name} UNEXPECTED ERROR: {type(e).__name__}: {e}")
        import traceback
        print(f"[call_routellm] Full traceback: {traceback.format_exc()}")
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
async def check_api_health() -> bool:
    """
    Quick health check for the RouteLLM API before running discovery.
    Returns True if API is responsive, False otherwise.
    """
    print(f"[check_api_health] Testing API responsiveness...")
    
    if not settings.ABACUS_API_KEY:
        print(f"[check_api_health] ✗ No API key available")
        return False
        
    headers = {
        "Authorization": f"Bearer {settings.ABACUS_API_KEY}",
        "Content-Type": "application/json",
    }
    
    # Ultra-simple test payload
    test_payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Test"}],
        "stream": False
    }
    
    timeout = httpx.Timeout(10.0, connect=3.0)
    
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(f"{settings.ABACUS_BASE_URL}/chat/completions", headers=headers, json=test_payload)
            
            if resp.status_code == 200:
                print(f"[check_api_health] ✓ API is responsive")
                return True
            else:
                print(f"[check_api_health] ✗ API returned status {resp.status_code}")
                return False
                
    except httpx.TimeoutException:
        print(f"[check_api_health] ✗ API timeout - may be overloaded")
        return False
    except Exception as e:
        print(f"[check_api_health] ✗ API error: {type(e).__name__}: {e}")
        return False

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
    
    # Check API health before proceeding with discovery
    api_healthy = await check_api_health()
    if not api_healthy:
        print(f"[query_models_discover] ⚠️  API health check failed - discovery may have issues")
        # Continue anyway, but user is warned
    
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
        
        # Truncate page text if too long to avoid API limits
        max_page_chars = 3000  # Conservative limit
        page_text_truncated = page_text[:max_page_chars]
        if len(page_text) > max_page_chars:
            print(f"[query_models_discover] Truncated page text from {len(page_text)} to {max_page_chars} chars")
        
        # Use a simpler, more robust prompt format
        prompt = f"""You are an expert editor. Find tortured phrases and grammar issues in the text below.

Respond with strict JSON format:

{{
  "findings": [
    {{
      "phrase": "exact problematic text",
      "severity": "high|medium|low",
      "suggestion": "improved phrase only",
      "context": "explanation of the issue"
    }}
  ]
}}

Requirements:
- Extract up to {max_findings_per_page} findings
- Focus on tortured phrases and grammar issues
- Provide ONLY the improved phrase in suggestion field
- Include detailed context explaining the issue

Text to analyze:
{page_text_truncated}"""
        
        print(f"[query_models_discover] Prompt length: {len(prompt)} characters")
        
        # Try the request with longer timeout and retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"[query_models_discover] Attempt {attempt + 1}/{max_retries} for page {page_num}")
                
                # Add a small delay between retries (exponential backoff)
                if attempt > 0:
                    delay = 2 ** attempt  # 2, 4, 8 seconds
                    print(f"[query_models_discover] Waiting {delay}s before retry...")
                    await asyncio.sleep(delay)
                
                # Use call_routellm with discovery-specific timeout
                result = await call_routellm_with_timeout(settings.DISCOVERY_MODEL_NAME, prompt, settings.DISCOVERY_TIMEOUT_SECS)
                
                if result and result.findings:
                    print(f"[query_models_discover] ✓ Page {page_num}: {settings.DISCOVERY_MODEL_NAME} found {len(result.findings)} findings")
                    
                    # Log the quality of findings (based on your excellent test results)
                    high_severity_count = sum(1 for f in result.findings if f.severity.lower() == 'high')
                    medium_severity_count = sum(1 for f in result.findings if f.severity.lower() == 'medium') 
                    low_severity_count = sum(1 for f in result.findings if f.severity.lower() == 'low')
                    print(f"[query_models_discover] Severity breakdown: {high_severity_count} high, {medium_severity_count} medium, {low_severity_count} low")
                    
                    # Process each finding and set correct page number
                    for i, finding in enumerate(result.findings):
                        # Truncate long context for cleaner logging
                        context_preview = finding.context[:100] + "..." if len(finding.context) > 100 else finding.context
                        print(f"[query_models_discover]   Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}' ({finding.severity})")
                        print(f"[query_models_discover]     Context: {context_preview}")
                        
                        # Validate suggestion is never empty
                        suggestion = finding.suggestion.strip() if finding.suggestion else ''
                        if not suggestion:
                            suggestion = await generate_smart_suggestion(finding.phrase, finding.context)
                            print(f"[query_models_discover]     Generated fallback suggestion: '{suggestion}'")
                        
                        # Create Finding with correct page number from our loop context
                        new_finding = Finding(
                            phrase=finding.phrase.lower(),  # normalize for deduplication
                            severity=finding.severity,
                            suggestion=suggestion,
                            page=page_num,  # Use actual page number from our context
                            start_char=finding.start_char,
                            end_char=finding.end_char,
                            context=finding.context or "Issue identified requiring attention",
                            source="LLM",
                        )
                        findings.append(new_finding)
                    break  # Success, exit retry loop
                    
                else:
                    print(f"[query_models_discover] ✗ Page {page_num}: {settings.DISCOVERY_MODEL_NAME} returned no findings (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        print(f"[query_models_discover] Retrying page {page_num} with simpler prompt...")
                        # Use a much simpler prompt for retry
                        simple_prompt = f"""Find grammar and phrase issues in this text. Respond with JSON:

{{
  "findings": [
    {{
      "phrase": "problematic text",
      "severity": "high",
      "suggestion": "improved text",
      "context": "explanation"
    }}
  ]
}}

Text: {page_text_truncated[:1000]}"""  # Even shorter for retry
                        prompt = simple_prompt
                        continue
                        
            except Exception as e:
                print(f"[query_models_discover] ✗ Page {page_num} attempt {attempt + 1} FAILED: {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    print(f"[query_models_discover] Retrying page {page_num}...")
                    continue
                else:
                    print(f"[query_models_discover] Page {page_num} failed after {max_retries} attempts")
                    print(f"[query_models_discover] Skipping page {page_num} - will continue with other pages")

    print(f"[query_models_discover] === DISCOVERY COMPLETED: {len(findings)} total findings across {len(pages)} pages ===")
    return findings