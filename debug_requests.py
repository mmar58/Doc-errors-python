#!/usr/bin/env python3
"""
Debug script to compare working vs failing requests
"""
import asyncio
import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_clients import call_routellm

async def test_simple_vs_complex():
    print("=== TESTING SIMPLE VS COMPLEX REQUESTS ===\n")
    
    # Test 1: Simple working case (like our successful test)
    print("1. Testing SIMPLE request (known to work)...")
    simple_prompt = '''You are an expert editor. Respond with strict JSON.

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
}'''
    
    print(f"Simple prompt length: {len(simple_prompt)} characters")
    result1 = await call_routellm("gpt-5", simple_prompt)
    print(f"Simple result: {'✓ SUCCESS' if result1 else '✗ FAILED'}\n")
    
    # Test 2: Medium complexity
    print("2. Testing MEDIUM complexity request...")
    medium_prompt = '''You are auditing scientific prose for tortured phrases.

Find issues in this text: "The strategies for human-made consciousness has been invited by an ever-increasing number of residents. The human-made consciousness technique that speaks to exploring a neural organization has been created."

Output JSON format:
{
  "findings": [
    {
      "phrase": "human-made consciousness",
      "severity": "high", 
      "suggestion": "artificial intelligence",
      "context": "Tortured phrase - AI terminology replacement. This phrase is an awkward substitute for the standard term artificial intelligence."
    }
  ]
}'''
    
    print(f"Medium prompt length: {len(medium_prompt)} characters")
    result2 = await call_routellm("gpt-5", medium_prompt)
    print(f"Medium result: {'✓ SUCCESS' if result2 else '✗ FAILED'}\n")
    
    # Test 3: Check if it's related to the discovery prompt format
    print("3. Testing DISCOVERY-style request (minimal)...")
    discovery_prompt = '''You are auditing scientific/technical prose for "tortured phrases" or risky wording that should be rewritten.
From the provided page text, extract up to 3 findings.

CRITICAL REQUIREMENTS:
- EVERY finding MUST have a concrete suggestion - no exceptions!
- In suggestions, provide ONLY the improved phrase - no explanatory text!

Output must be a JSON object:
{
  "findings": [
    {
      "phrase": "human-made consciousness",
      "severity": "high",
      "suggestion": "artificial intelligence", 
      "context": "Tortured phrase replacement."
    }
  ]
}

Text: "The human-made consciousness technique has been created."'''
    
    print(f"Discovery prompt length: {len(discovery_prompt)} characters")
    result3 = await call_routellm("gpt-5", discovery_prompt)
    print(f"Discovery result: {'✓ SUCCESS' if result3 else '✗ FAILED'}\n")
    
    print("=== COMPARISON TEST COMPLETED ===")

if __name__ == "__main__":
    asyncio.run(test_simple_vs_complex())
