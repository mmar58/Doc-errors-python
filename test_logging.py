#!/usr/bin/env python3
"""
Test script to verify enhanced logging in llm_clients.py
"""
import asyncio
import sys
import os

# Add the current directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_clients import test_routellm_canary, test_full_pipeline, generate_smart_suggestion, call_routellm

async def main():
    print("=== TESTING ENHANCED LOGGING IN LLM_CLIENTS ===\n")
    
    # Test 1: Basic canary test
    print("1. Testing basic API connectivity...")
    canary_success = await test_routellm_canary()
    print(f"   Result: {'✓ PASS' if canary_success else '✗ FAIL'}\n")
    
    # Test 2: Smart suggestion with logging
    print("2. Testing smart suggestion with enhanced logging...")
    suggestion = await generate_smart_suggestion("serves powerful tool", "Academic writing context")
    print(f"   Suggestion: 'serves powerful tool' -> '{suggestion}'\n")
    
    # Test 3: Simple call_routellm test
    print("3. Testing call_routellm with enhanced logging...")
    simple_prompt = """You are an expert editor. Respond with strict JSON.

Find issues in: "The method provides comprehensive analysis."

Output JSON:
{
  "findings": [
    {
      "phrase": "provides comprehensive analysis", 
      "severity": "low",
      "suggestion": "provides a comprehensive analysis",
      "context": "Missing article 'a' before 'comprehensive analysis'."
    }
  ]
}"""
    
    result = await call_routellm("gpt-5", simple_prompt)
    if result:
        print(f"   ✓ SUCCESS: {len(result.findings)} findings returned")
        for finding in result.findings:
            print(f"     - '{finding.phrase}' -> '{finding.suggestion}' (severity: {finding.severity})")
    else:
        print(f"   ✗ FAILED: No result returned")
    
    print("\n=== ENHANCED LOGGING TEST COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(main())
