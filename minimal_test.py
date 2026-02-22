#!/usr/bin/env python3
"""
Ultra-simple API test to isolate connectivity issues
"""
import asyncio
import httpx
import json

async def minimal_test():
    print("=== MINIMAL API TEST ===")
    
    headers = {
        "Authorization": "Bearer s2_8fe4b2ba5e984913af78aea198072d70",
        "Content-Type": "application/json",
    }
    
    # Minimal payload
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Say 'test' as JSON: {\"result\": \"test\"}"}],
        "response_format": {"type": "json_object"},
        "stream": False
    }
    
    print(f"Payload size: {len(json.dumps(payload))} chars")
    
    try:
        print("Testing with 5-second timeout...")
        timeout = httpx.Timeout(5.0, connect=2.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post("https://routellm.abacus.ai/v1/chat/completions", headers=headers, json=payload)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                print(f"Success! Response: {json.dumps(data, indent=2)}")
            else:
                print(f"Error response: {resp.text}")
                
    except httpx.TimeoutException:
        print("❌ TIMEOUT - API is taking too long to respond")
    except httpx.HTTPStatusError as e:
        print(f"❌ HTTP Error {e.response.status_code}: {e.response.text}")
    except Exception as e:
        print(f"❌ Other error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(minimal_test())
