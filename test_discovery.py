#!/usr/bin/env python3
"""
Test the improved discovery function
"""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from llm_clients import query_models_discover

async def test_discovery():
    print("=== TESTING IMPROVED DISCOVERY FUNCTION ===")
    
    # Test with a sample page that contains tortured phrases
    test_pages = [
        (1, """The strategies for human-made consciousness has been invited by an ever-increasing number of residents. The human-made consciousness technique that speaks to exploring a neural organization has been created at an exceptional rate. Such a business expectation, Assessment of scores, business misfortune forecast, these fields, for example vision and control framework, has been generally utilized.""")
    ]
    
    print(f"Testing discovery with {len(test_pages)} pages...")
    print(f"Sample text length: {len(test_pages[0][1])} characters")
    
    try:
        findings = await query_models_discover("Test Document", test_pages, 5)
        
        if findings:
            print(f"✓ SUCCESS: Found {len(findings)} findings")
            for i, finding in enumerate(findings):
                print(f"  Finding {i+1}: '{finding.phrase}' -> '{finding.suggestion}' (severity: {finding.severity})")
        else:
            print("✗ No findings returned")
            
    except Exception as e:
        print(f"✗ ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_discovery())
