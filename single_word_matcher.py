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
        print(f"[SingleWordMatcher] === STARTING SINGLE WORD BATCH ANALYSIS ===")
        print(f"[SingleWordMatcher] Document: {doc_title}")
        print(f"[SingleWordMatcher] Batch size: {len(word_batch)} words")
        
        from llm_clients import call_routellm
        from config import settings
        import json
        
        if not word_batch:
            print(f"[SingleWordMatcher] Empty word batch, returning no findings")
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
            "CRITICAL REQUIREMENTS:",
            "- EVERY finding MUST have a concrete suggestion - no exceptions!",
            "- Context must include detailed reasoning with 4 parts",
            "- Only include findings where you can provide BOTH clear problem AND concrete solution",
            "",
            "TASK: Analyze ALL words above for problematic 'tortured phrases' in their given contexts.",
            "For each word where you identify issues in ANY of its contexts:",
            "1. Extract the specific problematic phrase containing the word",
            "2. Assign severity: high, medium, or low",
            "3. Provide a MANDATORY concrete rewrite suggestion",
            "4. Provide detailed 4-part context explanation",
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
            '          "suggestion": {"type": "string", "description": "MANDATORY concrete rewrite suggestion - NEVER leave empty"},',
            '          "context": {"type": "string", "description": "Detailed explanation including phrase type, specific issue, why problematic, improvement rationale"}',
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
            '- suggestion: MANDATORY concrete rewrite suggestion with specific improved wording',
            '- context: Must include 4 parts: 1) Type of issue (tortured phrase/grammar/awkward) 2) Specific problem 3) Why it\'s problematic 4) How suggestion improves it',
            "",
            'EXAMPLE CONTEXT: "Tortured phrase - missing preposition. The phrase lacks grammatical completeness which is common in AI-generated content. The suggestion adds the missing \'as\' to create proper grammar and natural academic tone."',
            "",
            "Only include findings where you can provide BOTH a clear problem identification AND a concrete improvement suggestion."
        ])
        
        prompt = "\n".join(prompt_lines)
        
        # Try both models for comprehensive analysis
        all_findings = []
        
        try:
            # Try GPT-5 first
            print(f"[SingleWordMatcher] === CALLING GPT-5 FOR SINGLE WORD BATCH ===")
            print(f"[SingleWordMatcher] Batch size: {len(word_batch)} words")
            print(f"[SingleWordMatcher] Words in batch: {[word[0] for word in word_batch[:5]]}{'...' if len(word_batch) > 5 else ''}")
            result = await call_routellm(settings.ROUTELLM_MODEL_GPT5, prompt)
            if result and hasattr(result, 'findings') and result.findings:
                print(f"[SingleWordMatcher] ✓ GPT-5 SINGLE WORD BATCH SUCCESS: {len(result.findings)} findings")
                for i, finding_data in enumerate(result.findings):
                    # Convert new format to Finding object
                    finding = self._convert_to_finding(finding_data)
                    if finding:
                        finding.source = "LLM-SingleWord-Batch"
                        all_findings.append(finding)
                        print(f"[SingleWordMatcher]   GPT-5 SingleWord Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}' (severity: {finding.severity})")
            else:
                print(f"[SingleWordMatcher] ✗ GPT-5 SINGLE WORD BATCH: No findings returned")
        except Exception as e:
            print(f"[SingleWordMatcher] ✗ GPT-5 SINGLE WORD BATCH FAILED: {e}")
        
        try:
            # Try Claude as backup/additional analysis
            print(f"[SingleWordMatcher] === CALLING CLAUDE FOR SINGLE WORD BATCH ===")
            print(f"[SingleWordMatcher] Batch size: {len(word_batch)} words")
            result = await call_routellm(settings.ROUTELLM_MODEL_CLAUDE, prompt)
            if result and hasattr(result, 'findings') and result.findings:
                print(f"[SingleWordMatcher] ✓ CLAUDE SINGLE WORD BATCH SUCCESS: {len(result.findings)} findings")
                for i, finding_data in enumerate(result.findings):
                    # Convert new format to Finding object
                    finding = self._convert_to_finding(finding_data)
                    if finding:
                        finding.source = "LLM-SingleWord-Batch"
                        all_findings.append(finding)
                        print(f"[SingleWordMatcher]   Claude SingleWord Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}' (severity: {finding.severity})")
            else:
                print(f"[SingleWordMatcher] ✗ CLAUDE SINGLE WORD BATCH: No findings returned")
        except Exception as e:
            print(f"[SingleWordMatcher] ✗ CLAUDE SINGLE WORD BATCH FAILED: {e}")
        
        print(f"[SingleWordMatcher] === SINGLE WORD BATCH ANALYSIS COMPLETED ===")
        print(f"[SingleWordMatcher] Total findings from both models: {len(all_findings)}")
        
        # Add position information from the original word matches
        for finding in all_findings:
            # Try to match finding to original word position and set correct page number
            for word, page_num, start_char, end_char, phrase_candidates in word_batch:
                if word.lower() in (finding.phrase or "").lower():
                    finding.page = page_num  # Set the actual page number from our context
                    if finding.start_char is None:
                        finding.start_char = start_char
                    if finding.end_char is None:
                        finding.end_char = end_char
                    break
        
        return all_findings
    
    def _convert_to_finding(self, finding_data: dict) -> Optional[Finding]:
        """
        Convert simplified JSON format to Finding object.
        """
        try:
            # Extract required fields from simplified format
            phrase = finding_data.get('phrase', '')
            severity = finding_data.get('severity', 'medium')
            suggestion = finding_data.get('suggestion', '')
            context = finding_data.get('context', '')
            
            if not phrase:
                print(f"[SingleWordMatcher] Skipping finding with empty phrase")
                return None
            
            # Validate suggestion is not empty - this is now mandatory
            if not suggestion or suggestion.strip() == '':
                print(f"[SingleWordMatcher] Warning: Empty suggestion for phrase '{phrase}' - generating fallback")
                suggestion = f"Rewrite '{phrase}' for better clarity and academic tone"
            
            # Convert severity to title case for compatibility
            severity_map = {'low': 'Low', 'medium': 'Medium', 'high': 'High'}
            severity_title = severity_map.get(severity.lower(), 'Medium')
            
            return Finding(
                phrase=phrase,
                severity=severity_title,
                suggestion=suggestion,
                page=1,  # Will be updated with actual page number from batch context
                start_char=None,
                end_char=None,
                context=context if context else "Single word analysis - issue identified requiring attention",
                source="LLM"
            )
            
        except Exception as e:
            print(f"[SingleWordMatcher] Error converting finding: {e}")
            return None

# Global single word matcher instance
global_single_word_matcher = SingleWordMatcher()
