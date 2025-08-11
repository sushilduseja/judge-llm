# src/services/together_ai.py
import time
from typing import Dict, Generator, Optional, Any
from together import Together

class TogetherAIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        # The Together SDK will use TOGETHER_API_KEY from environment if api_key is None
        self.client = Together(api_key=api_key) if api_key else Together()

    def call(
        self, 
        model_id: str, 
        prompt_text: str, 
        max_tokens: int = 256, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        timeout: int = 120, 
        retries: int = 3
    ) -> Dict[str, Any]:
        backoff = 1.0
        
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                
                response = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": prompt_text
                    }],
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p)
                )
                
                elapsed = time.time() - start
                
                # Extract the response content
                if response.choices and len(response.choices) > 0:
                    text_out = response.choices[0].message.content.strip()
                    
                    # Extract usage info if available
                    usage_info = {}
                    if hasattr(response, 'usage') and response.usage:
                        usage_info = {
                            "prompt_tokens": getattr(response.usage, 'prompt_tokens', 0),
                            "completion_tokens": getattr(response.usage, 'completion_tokens', 0),
                            "total_tokens": getattr(response.usage, 'total_tokens', 0)
                        }
                    
                    return {
                        "ok": True, 
                        "text": text_out, 
                        "usage": usage_info, 
                        "elapsed": elapsed,
                        "raw": response
                    }
                else:
                    return {
                        "ok": False, 
                        "status": "parse_error", 
                        "text": "No choices in response", 
                        "elapsed": elapsed
                    }
            
            except Exception as e:
                error_msg = str(e)
                
                # Handle specific error types
                if "404" in error_msg or "not found" in error_msg.lower():
                    return {
                        "ok": False, 
                        "status": 404, 
                        "text": f"Model '{model_id}' not found. Please verify the model ID is correct.", 
                        "elapsed": time.time() - start if 'start' in locals() else None
                    }
                
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    return {
                        "ok": False, 
                        "status": 401, 
                        "text": "Authentication failed - check your Together AI API key", 
                        "elapsed": time.time() - start if 'start' in locals() else None
                    }
                
                if "rate limit" in error_msg.lower() or "429" in error_msg:
                    return {
                        "ok": False, 
                        "status": 429, 
                        "text": "Rate limited", 
                        "elapsed": time.time() - start if 'start' in locals() else None
                    }
                
                # For retriable errors, retry with backoff
                if attempt < retries and any(keyword in error_msg.lower() for keyword in ["timeout", "connection", "network", "503"]):
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                
                # Final attempt or non-retriable error
                return {
                    "ok": False, 
                    "status": "exception", 
                    "text": error_msg, 
                    "elapsed": time.time() - start if 'start' in locals() else None
                }
        
        return {
            "ok": False, 
            "status": "unknown", 
            "text": "Exceeded retries", 
            "elapsed": None
        }

    def stream(
        self, 
        model_id: str, 
        prompt_text: str, 
        max_tokens: int = 256, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        timeout: int = 120, 
        retries: int = 3
    ) -> Generator[Dict[str, Any], None, None]:
        backoff = 1.0
        
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                
                # Create streaming response
                stream = self.client.chat.completions.create(
                    model=model_id,
                    messages=[{
                        "role": "user",
                        "content": prompt_text
                    }],
                    max_tokens=int(max_tokens),
                    temperature=float(temperature),
                    top_p=float(top_p),
                    stream=True
                )
                
                usage_info = {}
                
                # Process streaming chunks
                for chunk in stream:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        if hasattr(delta, 'content') and delta.content:
                            yield {"ok": True, "text": delta.content}
                
                elapsed = time.time() - start
                
                # Send final completion signal
                yield {
                    "ok": True, 
                    "final": True, 
                    "usage": usage_info, 
                    "elapsed": elapsed
                }
                return
            
            except Exception as e:
                error_msg = str(e)
                
                # Handle specific errors
                if "404" in error_msg or "not found" in error_msg.lower():
                    yield {
                        "ok": False, 
                        "status": 404, 
                        "text": f"Model '{model_id}' not found"
                    }
                    return
                
                if "401" in error_msg or "unauthorized" in error_msg.lower():
                    yield {
                        "ok": False, 
                        "status": 401, 
                        "text": "Authentication failed - check your Together AI API key"
                    }
                    return
                
                # For retriable errors, retry with backoff
                if attempt < retries and any(keyword in error_msg.lower() for keyword in ["timeout", "connection", "network", "503"]):
                    time.sleep(backoff)
                    backoff *= 2.0
                    continue
                
                # Final attempt or non-retriable error
                yield {
                    "ok": False, 
                    "status": "exception", 
                    "text": error_msg, 
                    "elapsed": None
                }
                return
        
        yield {
            "ok": False, 
            "status": "unknown", 
            "text": "Exceeded retries", 
            "elapsed": None
        }