"""
Test script for single word analysis functionality
"""
import asyncio
from word_expansion import get_phrase_candidates_for_word, analyze_single_word_with_llm

def test_phrase_extraction():
    """Test the phrase extraction around single words"""
    test_text = """
    The characterization of neural networks shows promising results. 
    This research demonstrates how classification algorithms can be improved.
    Data analysis reveals significant patterns in the preparation phase.
    """
    
    # Test extracting phrases around the word "characterization"
    word = "characterization"
    start_pos = test_text.find(word)
    
    candidates = get_phrase_candidates_for_word(test_text, word, start_pos)
    
    print(f"Word: {word}")
    print(f"Position: {start_pos}")
    print("Phrase candidates:")
    for i, (phrase, start, end) in enumerate(candidates, 1):
        print(f"  {i}. '{phrase}' [{start}:{end}]")
    
    return candidates

async def test_llm_analysis():
    """Test LLM analysis of single words (mock version)"""
    # This would normally call the LLM, but we'll just show the structure
    word = "characterization"
    candidates = [
        ("The characterization of neural networks shows promising results", 5, 65),
        ("characterization of neural networks", 9, 44)
    ]
    
    print(f"\nTesting LLM analysis for word: {word}")
    print("This would send the following to the LLM:")
    print(f"Word: {word}")
    print("Context phrases:")
    for i, (phrase, _, _) in enumerate(candidates, 1):
        print(f"  {i}. \"{phrase}\"")
    
    # Mock result
    print("\nExpected LLM response format:")
    print("""{
  "doc_title": null,
  "summary": null,
  "findings": [
    {
      "phrase": "characterization of neural networks",
      "severity": "Medium",
      "suggestion": "description of neural networks",
      "page": 1,
      "start_char": null,
      "end_char": null,
      "context": "The word 'characterization' is academic jargon that could be simplified to 'description' or 'analysis'"
    }
  ]
}""")

if __name__ == "__main__":
    print("=== Testing Single Word Analysis ===")
    
    # Test phrase extraction
    candidates = test_phrase_extraction()
    
    # Test LLM structure (async)
    asyncio.run(test_llm_analysis())
    
    print("\n=== Test Complete ===")
    print("The system should now:")
    print("1. Detect single words from the CSV")
    print("2. Find contextual phrases around those words")
    print("3. Send contexts to LLM for analysis")
    print("4. Receive structured feedback with suggestions")
    print("5. Include results in the final report")
