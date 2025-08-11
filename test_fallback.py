# test_fallback.py
#!/usr/bin/env python3
"""
Quick test script to verify API functionality works
Run this to test if OpenRouter and Together AI APIs are working correctly
"""
import os
import json
from dotenv import load_dotenv
from together import Together
from src.config.models import AppConfig, ModelCapability
from src.services.openrouter import OpenRouterClient

def test_together_direct(model_id: str, prompt: str, together_token: str):
    """Test Together AI API using the official SDK"""
    try:
        # Create Together client
        client = Together(api_key=together_token)
        
        print(f"Testing Together AI with model: {model_id}")
        
        response = client.chat.completions.create(
            model=model_id,
            messages=[{
                "role": "user",
                "content": prompt
            }],
            max_tokens=100,
            temperature=0.0,
            top_p=1.0
        )
        
        if response.choices and len(response.choices) > 0:
            generated_text = response.choices[0].message.content
            return {"ok": True, "text": generated_text}
        else:
            return {"ok": False, "error": "No choices in response"}
            
    except Exception as e:
        return {"ok": False, "error": str(e)}

def test_apis():
    load_dotenv()
    
    # Get API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    
    print("=== API Key Status ===")
    print(f"OpenRouter API Key: {'✓ Found' if openrouter_key else '✗ Missing'}")
    print(f"Together AI API Key: {'✓ Found' if together_key else '✗ Missing'}")
    
    if openrouter_key:
        print(f"OpenRouter Key Preview: {openrouter_key[:10]}...")
    if together_key:
        print(f"Together Key Preview: {together_key[:10]}...")
    
    print("\n" + "="*50)
    
    # Load config and models
    config = AppConfig(
        openrouter_api_key=openrouter_key or "dummy",
        together_api_key=together_key
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
    if openrouter_key and together_key:
        print("Testing ClientManager with fallback...")
        from src.services.client_manager import ClientManager
        
        client_manager = ClientManager(openrouter_key, together_key)
        
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
    
    # Test Together AI with SDK
    if together_key:
        print("Testing Together AI with SDK...")
        # Try a free model
        test_together_model = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"
        
        together_result = test_together_direct(test_together_model, test_prompt, together_key)
        
        print(f"Together AI OK: {together_result.get('ok')}")
        if together_result.get('ok'):
            print("Together AI Success!")
            print(f"Response preview: {together_result.get('text', '')[:200]}")
        else:
            print(f"Together AI Error: {together_result.get('error')}")
    else:
        print("Skipping Together AI test (no API key)")
    
    print("\n" + "="*50)
    print("Test Summary:")
    print("1. Check that your .env file has the correct API keys")
    print("2. For OpenRouter: Add credits if you see rate limit errors")
    print("3. For Together AI: Verify your token has the right permissions")
    print("4. Install the Together SDK: pip install together")
    print("5. Use free Together models to avoid costs")

if __name__ == "__main__":
    test_apis()