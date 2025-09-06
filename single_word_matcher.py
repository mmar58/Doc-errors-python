"""
Single word matcher using simple string search for words that need context expansion
"""
from typing import List, Tuple, Dict, Optional
import re
import json
from word_expansion import get_phrase_candidates_for_word, analyze_single_word_with_llm
from models import Finding

class SingleWordMatcher:
    def __init__(self):
        self._words: List[str] = []
        self._ready: bool = False

    def build(self, single_words: List[str]):
        """Build matcher with list of single words to search for"""
        self._words = [w for w in single_words if w and len(w.strip()) > 0]
        self._ready = True
        print(f"[SingleWordMatcher] Built matcher with {len(self._words)} single words")

    def is_ready(self) -> bool:
        return self._ready

    def search_pages(self, pages_text: List[str]) -> List[Tuple[str, int, int, int, List[Tuple[str, int, int]]]]:
        """
        Search for single words and return contexts for LLM analysis.
        
        Returns:
            List of (word, page_1_based, start_char, end_char, phrase_candidates)
            where phrase_candidates is [(phrase, phrase_start, phrase_end), ...]
        """
        if not self.is_ready():
            return []
        
        results = []
        
        for page_idx, text in enumerate(pages_text):
            if not text:
                continue
                
            text_lower = text.lower()
            
            for word in self._words:
                # Use word boundary regex to find whole word matches
                pattern = r'\b' + re.escape(word.lower()) + r'\b'
                
                for match in re.finditer(pattern, text_lower):
                    start_pos = match.start()
                    end_pos = match.end()
                    
                    # Get phrase candidates around this word
                    candidates = get_phrase_candidates_for_word(text, word, start_pos)
                    
                    if candidates:  # Only include if we found meaningful context
                        results.append((
                            word,
                            page_idx + 1,  # 1-based page number
                            start_pos,
                            end_pos,
                            candidates
                        ))
        
        return results

    async def analyze_words_with_llm(
        self, 
        word_matches: List[Tuple[str, int, int, int, List[Tuple[str, int, int]]]],
        doc_title: Optional[str] = None,
        job_id: Optional[str] = None
    ) -> List[Finding]:
        """
        Analyze single word matches using LLM in batches to determine if they're part of problematic phrases.
        
        Args:
            word_matches: Output from search_pages()
            doc_title: Document title for context
            job_id: Job ID for progress tracking
            
        Returns:
            List of Finding objects from LLM analysis
        """
        from progress_tracker import update_job_progress, ProgressStatus
        
        if not word_matches:
            return []
        
        print(f"[SingleWordMatcher] Starting batch analysis of {len(word_matches)} word matches")
        
        # Group word matches by batches for efficient processing
        batch_size = 10  # Process 10 words at once
        all_findings = []
        
        for batch_start in range(0, len(word_matches), batch_size):
            batch_end = min(batch_start + batch_size, len(word_matches))
            batch = word_matches[batch_start:batch_end]
            
            batch_num = (batch_start // batch_size) + 1
            total_batches = (len(word_matches) + batch_size - 1) // batch_size
            
            print(f"[SingleWordMatcher] Processing batch {batch_num}/{total_batches} ({len(batch)} words)")
            
            # Update progress if job_id provided
            if job_id:
                progress_msg = f"Analyzing words batch {batch_num}/{total_batches} (words {batch_start+1}-{batch_end})"
                await update_job_progress(job_id, ProgressStatus.ANALYZING_SINGLE_WORDS, progress_msg)
            
            try:
                # Use batch analysis function
                batch_findings = await self._analyze_word_batch_with_llm(batch, doc_title)
                all_findings.extend(batch_findings)
                
                print(f"[SingleWordMatcher] Batch {batch_num} completed: {len(batch_findings)} findings")
                
            except Exception as e:
                print(f"[SingleWordMatcher] Error analyzing batch {batch_num}: {e}")
                continue
        
        print(f"[SingleWordMatcher] Completed all batches: {len(all_findings)} total findings")
        return all_findings
    
    async def _analyze_word_batch_with_llm(
        self,
        word_batch: List[Tuple[str, int, int, int, List[Tuple[str, int, int]]]],
        doc_title: Optional[str] = None
    ) -> List[Finding]:
        """
        Analyze a batch of words with LLM in a single call for efficiency.
        """
        from llm_clients import call_routellm
        from config import settings
        import json
        
        if not word_batch:
            return []
        
        # Build comprehensive prompt for batch analysis
        prompt_lines = [
            "You are analyzing multiple single words that may be part of 'tortured phrases' in academic writing.",
            f"Document: {doc_title or 'Unknown'}",
            "",
            "WORDS TO ANALYZE:",
        ]
        
        # Add each word with its contexts
        for i, (word, page_num, start_char, end_char, phrase_candidates) in enumerate(word_batch, 1):
            prompt_lines.append(f"\n{i}. WORD: '{word}' (Page {page_num})")
            prompt_lines.append("   Contexts:")
            
            for j, (phrase, _, _) in enumerate(phrase_candidates, 1):
                prompt_lines.append(f"   {j}. \"{phrase}\"")
        
        prompt_lines.extend([
            "",
            "TASK: Analyze ALL words above for problematic 'tortured phrases' in their given contexts.",
            "For each word where you identify issues in ANY of its contexts:",
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
            '        "required": ["category","page","section","span","exact","suggestion","rationale","severity"],',
            '        "properties": {',
            '          "category": {"enum": ["tortured_phrase","ai_fingerprint","awkward_sentence","grammar_formatting"]},',
            '          "page": {"type": "integer", "minimum": 1},',
            '          "section": {"type": "string"},',
            '          "span": {"type": "string", "description": "span_id from sentences.jsonl"},',
            '          "exact": {"type": "string"},',
            '          "suggestion": {"type": "string"},',
            '          "rationale": {"type": "string", "maxLength": 240},',
            '          "severity": {"enum": ["low","medium","high"]}',
            '        }',
            '      }',
            '    }',
            '  },',
            '  "required": ["findings"],',
            '  "additionalProperties": false',
            '}',
            "",
            "For each finding:",
            '- category: Use "tortured_phrase" for problematic academic phrases',
            '- page: The page number where the issue was found',
            '- section: Brief description of document section (e.g., "Introduction", "Methods")',
            '- span: Use format "page_X_span_Y" where X is page number and Y is a unique span ID',
            '- exact: The exact problematic text you identified',
            '- suggestion: Your concrete rewrite suggestion',
            '- rationale: Brief explanation of why this is problematic (max 240 chars)',
            '- severity: "high" for clearly problematic, "medium" for questionable, "low" for minor',
            "",
            "Only include findings where you're confident there's an actual issue."
        ])
        
        prompt = "\n".join(prompt_lines)
        
        # Try both models for comprehensive analysis
        all_findings = []
        
        try:
            # Try GPT-5 first
            print(f"[SingleWordMatcher] Calling GPT-5 for batch of {len(word_batch)} words")
            result = await call_routellm(settings.ROUTELLM_MODEL_GPT5, prompt)
            if result and hasattr(result, 'findings') and result.findings:
                for finding_data in result.findings:
                    # Convert new format to Finding object
                    finding = self._convert_to_finding(finding_data)
                    if finding:
                        finding.source = "LLM-SingleWord-Batch"
                        all_findings.append(finding)
                print(f"[SingleWordMatcher] GPT-5 returned {len(result.findings)} findings")
        except Exception as e:
            print(f"[SingleWordMatcher] GPT-5 batch call failed: {e}")
        
        try:
            # Try Claude as backup/additional analysis
            print(f"[SingleWordMatcher] Calling Claude for batch of {len(word_batch)} words")
            result = await call_routellm(settings.ROUTELLM_MODEL_CLAUDE, prompt)
            if result and hasattr(result, 'findings') and result.findings:
                for finding_data in result.findings:
                    # Convert new format to Finding object
                    finding = self._convert_to_finding(finding_data)
                    if finding:
                        finding.source = "LLM-SingleWord-Batch"
                        all_findings.append(finding)
                print(f"[SingleWordMatcher] Claude returned {len(result.findings)} findings")
        except Exception as e:
            print(f"[SingleWordMatcher] Claude batch call failed: {e}")
        
        # Add position information from the original word matches
        for finding in all_findings:
            # Try to match finding to original word position
            for word, page_num, start_char, end_char, phrase_candidates in word_batch:
                if finding.page == page_num and word.lower() in (finding.phrase or "").lower():
                    if finding.start_char is None:
                        finding.start_char = start_char
                    if finding.end_char is None:
                        finding.end_char = end_char
                    break
        
        return all_findings
    
    def _convert_to_finding(self, finding_data: dict) -> Optional[Finding]:
        """
        Convert new JSON format to Finding object.
        """
        try:
            # Extract required fields
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
            
            return Finding(
                phrase=exact,
                severity=severity_title,
                suggestion=suggestion,
                page=page,
                start_char=None,
                end_char=None,
                context=context,
                source="LLM"
            )
            
        except Exception as e:
            print(f"[SingleWordMatcher] Error converting finding: {e}")
            return None

# Global single word matcher instance
global_single_word_matcher = SingleWordMatcher()
