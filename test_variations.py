#!/usr/bin/env python3
"""
Test different API variations to identify the issue
"""
import asyncio
import httpx
import json

async def test_variations():
    print("=== TESTING API VARIATIONS ===")
    
    headers = {
        "Authorization": "Bearer s2_8fe4b2ba5e984913af78aea198072d70",
        "Content-Type": "application/json",
    }
    
    # Test 1: Without response_format
    print("\n1. Testing without response_format...")
    payload1 = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False
    }
    await test_request(headers, payload1, "without response_format")
    
    # Test 2: Different model
    print("\n2. Testing with claude model...")
    payload2 = {
        "model": "claude-sonnet-4-20250514",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False
    }
    await test_request(headers, payload2, "claude model")
    
    # Test 3: Very minimal gpt-5
    print("\n3. Testing minimal gpt-5...")
    payload3 = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Hi"}]
    }
    await test_request(headers, payload3, "minimal gpt-5")

async def test_request(headers, payload, description):
    try:
        print(f"   Testing {description}...")
        timeout = httpx.Timeout(10.0, connect=3.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post("https://routellm.abacus.ai/v1/chat/completions", headers=headers, json=payload)
            print(f"   ✓ Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"   ✓ Response: {str(data)[:200]}...")
            else:
                print(f"   ❌ Error: {resp.text[:200]}")
                
    except httpx.TimeoutException:
        print(f"   ❌ TIMEOUT for {description}")
    except Exception as e:
        print(f"   ❌ Error for {description}: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(test_variations())
