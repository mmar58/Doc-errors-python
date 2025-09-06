"""
Single word matcher using simple string search for words that need context expansion
"""
from typing import List, Tuple, Dict, Optional
import re
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
        doc_title: Optional[str] = None
    ) -> List[Finding]:
        """
        Analyze single word matches using LLM to determine if they're part of problematic phrases.
        
        Args:
            word_matches: Output from search_pages()
            doc_title: Document title for context
            
        Returns:
            List of Finding objects from LLM analysis
        """
        all_findings = []
        
        for word, page_num, start_char, end_char, phrase_candidates in word_matches:
            try:
                findings = await analyze_single_word_with_llm(
                    word=word,
                    phrase_candidates=phrase_candidates,
                    page_num=page_num,
                    doc_title=doc_title
                )
                
                # Add position information from the original word match
                for finding in findings:
                    if finding.start_char is None:
                        finding.start_char = start_char
                    if finding.end_char is None:
                        finding.end_char = end_char
                
                all_findings.extend(findings)
                
            except Exception as e:
                print(f"[SingleWordMatcher] Error analyzing word '{word}' on page {page_num}: {e}")
                continue
        
        return all_findings

# Global single word matcher instance
global_single_word_matcher = SingleWordMatcher()
