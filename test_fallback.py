#!/usr/bin/env python3
"""
Quick test script to verify API functionality works
Run this to test if OpenRouter and HuggingFace APIs are working correctly
"""
import os
import json
import requests
from dotenv import load_dotenv
from src.config.models import AppConfig, ModelCapability
from src.services.openrouter import OpenRouterClient

def test_huggingface_direct(model_id: str, prompt: str, hf_token: str):
    """Test HuggingFace Inference Providers API directly"""
    # Use the new Inference Providers endpoint
    url = "https://router.huggingface.co/v1/chat/completions"
    
    headers = {
        'Authorization': f'Bearer {hf_token}',
        'Content-Type': 'application/json'
    }
    
    body = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "max_tokens": 100,
        "temperature": 0.0,
        "top_p": 1.0
    }
    
    print(f"HF API call to: {url}")
    print(f"Headers: {headers}")
    print(f"Body: {json.dumps(body, indent=2)}")
    
    try:
        response = requests.post(url, headers=headers, json=body, timeout=30)
        print(f"HF Response status: {response.status_code}")
        print(f"HF Response text: {response.text}")
        
        if response.status_code == 200:
            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                generated_text = data["choices"][0]["message"]["content"]
                return {"ok": True, "text": generated_text}
            else:
                return {"ok": False, "error": f"Unexpected response format: {data}"}
        else:
            return {"ok": False, "error": response.text}
            
    except Exception as e:
        return {"ok": False, "error": str(e)}

def test_apis():
    load_dotenv()
    
    # Get API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    hf_key = os.getenv("HUGGINGFACE_API_KEY")
    
    print("=== API Key Status ===")
    print(f"OpenRouter API Key: {'✓ Found' if openrouter_key else '✗ Missing'}")
    print(f"HuggingFace API Key: {'✓ Found' if hf_key else '✗ Missing'}")
    
    if openrouter_key:
        print(f"OpenRouter Key Preview: {openrouter_key[:10]}...")
    if hf_key:
        print(f"HuggingFace Key Preview: {hf_key[:10]}...")
    
    print("\n" + "="*50)
    
    # Load config and models
    config = AppConfig(
        openrouter_api_key=openrouter_key or "dummy",
        huggingface_api_key=hf_key
    )
    
    with open('models.json', 'r') as f:
        models_data = json.load(f)
    
    # Test with first model
    test_model_data = models_data[0]
    test_model = ModelCapability(**test_model_data)
    
    print(f"Testing model: {test_model.name}")
    print(f"OpenRouter ID: {test_model.id}")
    print("-" * 50)
    
    test_prompt = "Write a simple hello world function in Python"
    
    # Test with ClientManager (includes fallback)
    if openrouter_key and hf_key:
        print("Testing ClientManager with fallback...")
        from src.services.client_manager import ClientManager
        
        client_manager = ClientManager(openrouter_key, hf_key)
        
        fallback_result = client_manager.call_with_fallback(
            test_model,
            test_prompt,
            max_tokens=100,
            temperature=0.0
        )
        
        print(f"Fallback Test OK: {fallback_result.get('ok')}")
        print(f"Fallback used: {fallback_result.get('fallback_used', False)}")
        print(f"Fallback attempted: {fallback_result.get('fallback_attempted', False)}")
        
        if fallback_result.get('ok'):
            print("Fallback Test Success!")
            if fallback_result.get('fallback_used'):
                print(f"Used fallback model: {fallback_result.get('fallback_model')}")
            print(f"Response preview: {fallback_result.get('text', '')[:200]}")
        else:
            print(f"Fallback Test Failed: {fallback_result.get('text', 'Unknown error')}")
    
    print("\n" + "="*50)
    
    # Test OpenRouter
    if openrouter_key:
        print("Testing OpenRouter...")
        client = OpenRouterClient(openrouter_key)
        
        or_result = client.call(
            test_model.id,
            test_prompt,
            max_tokens=100,
            temperature=0.0
        )
        
        print(f"OpenRouter OK: {or_result.get('ok')}")
        if or_result.get('ok'):
            print("OpenRouter Success!")
            print(f"Response preview: {or_result.get('text', '')[:200]}")
        else:
            print(f"OpenRouter Error: {or_result.get('text', 'Unknown error')}")
            if 'status' in or_result:
                print(f"Status Code: {or_result['status']}")
    else:
        print("Skipping OpenRouter test (no API key)")
    
    print("\n" + "="*50)
    
    # Test HuggingFace
    if hf_key:
        print("Testing HuggingFace...")
        # Try a smaller model first
        test_hf_model = "microsoft/phi-4"
        
        hf_result = test_huggingface_direct(test_hf_model, test_prompt, hf_key)
        
        print(f"HuggingFace OK: {hf_result.get('ok')}")
        if hf_result.get('ok'):
            print("HuggingFace Success!")
            print(f"Response preview: {hf_result.get('text', '')[:200]}")
        else:
            print(f"HuggingFace Error: {hf_result.get('error')}")
    else:
        print("Skipping HuggingFace test (no API key)")
    
    print("\n" + "="*50)
    print("Test Summary:")
    print("1. Check that your .env file has the correct API keys")
    print("2. For OpenRouter: Add credits if you see rate limit errors")
    print("3. For HuggingFace: Verify your token has the right permissions")
    print("4. Make sure your HF token starts with 'hf_'")

if __name__ == "__main__":
    test_apis()