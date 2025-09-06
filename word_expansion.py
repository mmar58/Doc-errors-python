"""
Module for expanding single words into contextual phrases for analysis
"""
from typing import List, Tuple, Optional, Dict
import re
import json
from models import Finding

async def generate_smart_suggestion(phrase: str, context: str = "") -> str:
    """
    Generate intelligent suggestion by asking AI models for the improved phrase only.
    """
    # Import here to avoid circular imports
    from llm_clients import generate_smart_suggestion as ai_suggestion
    try:
        return await ai_suggestion(phrase, context)
    except Exception as e:
        print(f"[word_expansion] AI suggestion failed for '{phrase}': {e}")
        # Quick pattern-based fallbacks
        phrase_lower = phrase.lower().strip()
        if 'serves' in phrase_lower and 'tool' in phrase_lower and 'as' not in phrase_lower:
            return phrase.replace('serves', 'serves as a', 1)
        if 'provides' in phrase_lower and ('overview' in phrase_lower or 'analysis' in phrase_lower) and ' a ' not in phrase_lower:
            return phrase.replace('provides', 'provides a', 1)
        return f"[Improve: {phrase}]"

def extract_phrase_around_word(text: str, word: str, start_pos: int, context_window: int = 50) -> Tuple[str, int, int]:
    """
    Extract a meaningful phrase around a single word match.
    
    Args:
        text: The full text where the word was found
        word: The single word that was matched
        start_pos: Starting position of the word in text
        context_window: Number of characters to expand around the word
    
    Returns:
        Tuple of (expanded_phrase, new_start_pos, new_end_pos)
    """
    word_end = start_pos + len(word)
    
    # Define word boundaries and sentence boundaries
    sentence_boundaries = r'[.!?;:]'
    
    # Find sentence boundaries before and after the word
    text_before = text[:start_pos]
    text_after = text[word_end:]
    
    # Look for sentence start (go back from word position)
    sentence_start = 0
    last_sentence_end = -1
    for match in re.finditer(sentence_boundaries, text_before):
        last_sentence_end = match.end()
    
    if last_sentence_end >= 0:
        sentence_start = last_sentence_end
    
    # Look for sentence end (go forward from word position)
    sentence_end = len(text)
    first_sentence_end = re.search(sentence_boundaries, text_after)
    if first_sentence_end:
        sentence_end = word_end + first_sentence_end.start()
    
    # Extract the sentence containing the word
    sentence = text[sentence_start:sentence_end].strip()
    
    # If sentence is too long, use a smaller context window
    if len(sentence) > 200:
        context_start = max(0, start_pos - context_window)
        context_end = min(len(text), word_end + context_window)
        
        # Try to break at word boundaries
        context_text = text[context_start:context_end]
        
        # Find better boundaries at word breaks
        if context_start > 0:
            space_before = context_text.find(' ')
            if space_before > 0:
                context_start += space_before + 1
        
        if context_end < len(text):
            space_after = context_text.rfind(' ')
            if space_after > 0:
                context_end = context_start + space_after
        
        expanded_phrase = text[context_start:context_end].strip()
        new_start = context_start
        new_end = context_end
    else:
        expanded_phrase = sentence
        new_start = sentence_start
        new_end = sentence_end
    
    # Clean up the phrase
    expanded_phrase = ' '.join(expanded_phrase.split())
    
    return expanded_phrase, new_start, new_end

def get_phrase_candidates_for_word(text: str, word: str, start_pos: int) -> List[Tuple[str, int, int]]:
    """
    Generate multiple phrase candidates around a single word for LLM analysis.
    
    Args:
        text: The full text
        word: The matched single word
        start_pos: Position where word was found
        
    Returns:
        List of (phrase, start_pos, end_pos) tuples
    """
    candidates = []
    
    # 1. Sentence-level context
    sentence_phrase, sent_start, sent_end = extract_phrase_around_word(text, word, start_pos, 100)
    if sentence_phrase and len(sentence_phrase) > len(word):
        candidates.append((sentence_phrase, sent_start, sent_end))
    
    # 2. Smaller context (noun phrases, etc.)
    small_phrase, small_start, small_end = extract_phrase_around_word(text, word, start_pos, 30)
    if small_phrase and len(small_phrase) > len(word) and small_phrase != sentence_phrase:
        candidates.append((small_phrase, small_start, small_end))
    
    # 3. Very focused context (immediate surrounding words)
    focused_phrase, focused_start, focused_end = extract_phrase_around_word(text, word, start_pos, 15)
    if focused_phrase and len(focused_phrase) > len(word) and focused_phrase not in [sentence_phrase, small_phrase]:
        candidates.append((focused_phrase, focused_start, focused_end))
    
    return candidates

async def analyze_single_word_with_llm(
    word: str,
    phrase_candidates: List[Tuple[str, int, int]],
    page_num: int,
    doc_title: Optional[str] = None
) -> List[Finding]:
    """
    Send single word contexts to LLM for analysis.
    
    Args:
        word: The original single word from CSV
        phrase_candidates: List of (phrase, start, end) tuples around the word
        page_num: Page number where word was found
        doc_title: Document title if available
        
    Returns:
        List of Finding objects from LLM analysis
    """
    from llm_clients import call_routellm
    from config import settings
    import json
    
    if not phrase_candidates:
        return []
    
    # Build prompt for single word analysis
    prompt_lines = [
        "You are analyzing a single word that may be part of a 'tortured phrase' in academic writing.",
        f"Word found: '{word}'",
        f"Document: {doc_title or 'Unknown'}, Page: {page_num}",
        "",
        "Context phrases containing this word:",
    ]
    
    for i, (phrase, _, _) in enumerate(phrase_candidates, 1):
        prompt_lines.append(f"{i}. \"{phrase}\"")
    
    prompt_lines.extend([
        "",
        "Task: Analyze if this word appears in problematic 'tortured phrases' within the given contexts.",
        "For each context where you identify issues:",
        "1. Extract the specific problematic phrase containing the word",
        "2. Assign severity: high, medium, or low",
        "3. Provide a concrete rewrite suggestion",
        "4. Categorize the type of issue",
        "",
        "Output must be a JSON object matching this exact schema:",
        '{',
        '  "type": "object",',
        '  "properties": {',
        '    "findings": {',
        '      "type": "array",',
        '      "items": {',
        '        "type": "object",',
        '        "required": ["phrase","severity","suggestion","context"],',
        '        "properties": {',
        '          "phrase": {"type": "string", "description": "The exact problematic text you identified"},',
        '          "severity": {"enum": ["low","medium","high"]},',
        '          "suggestion": {"type": "string", "description": "Your concrete rewrite suggestion"},',
        '          "context": {"type": "string", "description": "Detailed explanation of why this is problematic and context information"}',
        '        }',
        '      }',
        '    }',
        '  },',
        '  "required": ["findings"],',
        '  "additionalProperties": false',
        '}',
        "",
        "MANDATORY REQUIREMENTS for each finding:",
        '- phrase: The exact problematic text you identified',
        '- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor',
        '- suggestion: MANDATORY concrete rewrite suggestion with specific improved wording - NEVER leave empty',
        '- context: Must include 4 parts: 1) Type of issue (tortured phrase/AI fingerprint/grammar) 2) Specific problem 3) Why it\'s problematic 4) How suggestion improves it',
        "",
        'EXAMPLE CONTEXT: "Tortured phrase - grammatical incompleteness. This phrase lacks proper structure and sounds unnatural in academic writing. The suggestion provides grammatically correct and professional phrasing."',
        "",
        "CRITICAL: Every finding MUST have both a clear problem identification AND a concrete improvement suggestion.",
        "Only include findings where you're confident there's an actual issue."
    ])
    
    prompt = "\n".join(prompt_lines)
    
    # Try both models
    findings = []
    
    try:
        # Try GPT-5 first
        result = await call_routellm(settings.ROUTELLM_MODEL_GPT5, prompt)
        if result and result.findings:
            for finding in result.findings:
                # Validate and fix empty suggestions
                if not finding.suggestion or finding.suggestion.strip() == '':
                    print(f"[word_expansion] Warning: Empty suggestion for '{finding.phrase}' - generating smart fallback")
                    finding.suggestion = await generate_smart_suggestion(finding.phrase, finding.context)
                
                # Validate and fix empty context
                if not finding.context or finding.context.strip() == '':
                    print(f"[word_expansion] Warning: Empty context for '{finding.phrase}' - generating fallback")
                    finding.context = "Single word analysis - issue identified requiring attention and improvement"
                
                finding.source = "LLM-SingleWord"
                finding.page = page_num
                findings.append(finding)
    except Exception as e:
        print(f"[word_expansion] GPT-5 call failed: {e}")
    
    try:
        # Try Claude as backup/additional analysis
        result = await call_routellm(settings.ROUTELLM_MODEL_CLAUDE, prompt)
        if result and result.findings:
            for finding in result.findings:
                # Validate and fix empty suggestions
                if not finding.suggestion or finding.suggestion.strip() == '':
                    print(f"[word_expansion] Warning: Empty suggestion for '{finding.phrase}' - generating smart fallback")
                    finding.suggestion = await generate_smart_suggestion(finding.phrase, finding.context)
                
                # Validate and fix empty context
                if not finding.context or finding.context.strip() == '':
                    print(f"[word_expansion] Warning: Empty context for '{finding.phrase}' - generating fallback")
                    finding.context = "Single word analysis - issue identified requiring attention and improvement"
                
                finding.source = "LLM-SingleWord"
                finding.page = page_num
                findings.append(finding)
    except Exception as e:
        print(f"[word_expansion] Claude call failed: {e}")
    
    return findings
