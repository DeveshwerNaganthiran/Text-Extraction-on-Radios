#!/usr/bin/env python3
"""Test MSI GenAI session initialization using .env credentials"""
import requests
import os
from dotenv import load_dotenv

load_dotenv()

API_BASE_URL = os.getenv('MSI_HOST', "https://genai-service.stage.commandcentral.com/app-gateway/api/v2")
API_KEY = os.getenv('MSI_API_KEY')
MODEL_NAME = os.getenv('MSI_MODEL', "Claude-Sonnet-4")
USER_ID = os.getenv('MSI_USER_ID')
DATASTORE_ID = os.getenv('MSI_DATASTORE_ID')

def init_chat_session():
    """Initialize session using .env credentials"""
    url = f"{API_BASE_URL}/chat"
    headers = {"x-msi-genai-api-key": API_KEY, "Content-Type": "application/json"}
    payload = {
        "userId": USER_ID,
        "model": MODEL_NAME,
        "datastoreId": DATASTORE_ID,
        "prompt": "init"
    }
    
    print(f"[TEST] URL: {url}")
    print(f"[TEST] API Key: {API_KEY[:20]}...")
    print(f"[TEST] User ID: {USER_ID}")
    print(f"[TEST] Datastore ID: {DATASTORE_ID[:10]}...")
    
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        print(f"[TEST] Status: {response.status_code}")
        print(f"[TEST] Response: {response.text}")
        
        if response.status_code == 200:
            response_data = response.json()
            print(f"[TEST] ✓ Success!")
            return response_data
        else:
            print(f"[TEST] ✗ Failed")
            
    except Exception as e:
        print(f"[TEST] ✗ Exception: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    init_chat_session()
