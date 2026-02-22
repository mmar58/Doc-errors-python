#!/usr/bin/env python3
"""
Quick test to verify the improved llm_clients functionality
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_clients import call_routellm, generate_smart_suggestion

async def test_improvements():
    print("=== TESTING IMPROVED LLM_CLIENTS ===\n")
    
    # Test 1: Smart suggestion
    print("1. Testing smart suggestion...")
    suggestion = await generate_smart_suggestion("serves powerful tool", "Found in academic paper")
    print(f"   Result: 'serves powerful tool' -> '{suggestion}'\n")
    
    # Test 2: Call routellm with simple prompt
    print("2. Testing call_routellm with enhanced logging...")
    prompt = '''You are an expert editor. Respond with strict JSON.

Find issues in: "The system provides comprehensive analysis and serves powerful tool."

Output JSON format:
{
  "findings": [
    {
      "phrase": "provides comprehensive analysis",
      "severity": "low", 
      "suggestion": "provides a comprehensive analysis",
      "context": "Missing article 'a' before comprehensive analysis."
    },
    {
      "phrase": "serves powerful tool",
      "severity": "medium",
      "suggestion": "serves as a powerful tool", 
      "context": "Missing preposition 'as' makes this phrase grammatically incomplete."
    }
  ]
}'''
    
    result = await call_routellm("gpt-5", prompt)
    
    if result and result.findings:
        print(f"   ✓ SUCCESS: {len(result.findings)} findings")
        for i, finding in enumerate(result.findings):
            print(f"     Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}' ({finding.severity})")
    else:
        print("   ✗ FAILED: No results")
    
    print("\n=== TEST COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(test_improvements())
