#!/usr/bin/env python3
"""Test MSI GenAI session initialization using genai_client.py credentials"""
import requests
import json

API_BASE_URL = "https://genai-service.stage.commandcentral.com/app-gateway/api/v2"
API_KEY = ".ux9wjXnNd8ZX7A;(g0QSSkshAX5y@*7w):EuE9v"
MODEL_NAME = "Claude-Sonnet-4"
USER_ID = "wqm764@motorolasolutions.com"
DATASTORE_ID = "1579319e-2b48-4bad-9825-4a7dd10ac0ef"

def init_chat_session():
    """Initialize session using exact genai_client.py code"""
    url = f"{API_BASE_URL}/chat"
    headers = {"x-msi-genai-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "userId": USER_ID,
        "model": MODEL_NAME,
        "datastoreId": DATASTORE_ID,
        "prompt": "init"
    }
    
    print(f"[TEST] URL: {url}")
    print(f"[TEST] Headers: {headers}")
    print(f"[TEST] Payload: {json.dumps(payload, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"[TEST] Status: {response.status_code}")
        print(f"[TEST] Response: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            if response_data.get("status") and "sessionId" in response_data:
                session_id = response_data["sessionId"]
                print(f"[TEST] ✓ Session initialized: {session_id}")
                return session_id
            else:
                print(f"[TEST] ✗ Invalid response format: {response_data}")
        else:
            print(f"[TEST] ✗ Failed with status {response.status_code}")
            
    except Exception as e:
        print(f"[TEST] ✗ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    init_chat_session()
