#!/usr/bin/env python3
"""
Simple test to check basic connectivity
"""
import asyncio
import httpx
import json

async def simple_api_test():
    print("=== SIMPLE API TEST ===")
    
    headers = {
        "Authorization": "Bearer s2_8fe4b2ba5e984913af78aea198072d70",
        "Content-Type": "application/json",
    }
    
    payload = {
        "model": "gpt-5",
        "messages": [{"role": "user", "content": "Say hello"}],
        "stream": False
    }
    
    try:
        print("Sending simple request...")
        timeout = httpx.Timeout(10.0, connect=5.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post("https://routellm.abacus.ai/v1/chat/completions", headers=headers, json=payload)
            print(f"Status: {resp.status_code}")
            print(f"Response: {resp.text[:500]}")
            
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")

if __name__ == "__main__":
    asyncio.run(simple_api_test())
