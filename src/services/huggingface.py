import requests
import json
import time
from typing import Dict, Generator, Optional, Any

class HuggingFaceClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api-inference.huggingface.co/models"
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _create_request_body(
        self, 
        prompt_text: str, 
        max_tokens: int = 256, 
        temperature: float = 0.0, 
        top_p: float = 1.0
    ) -> Dict[str, Any]:
        return {
            "inputs": prompt_text,
            "parameters": {
                "max_new_tokens": int(max_tokens),
                "temperature": float(temperature),
                "top_p": float(top_p),
                "return_full_text": False,
                "do_sample": temperature > 0.0
            }
        }

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
        url = f"{self.base_url}/{model_id}"
        body = self._create_request_body(prompt_text, max_tokens, temperature, top_p)
        backoff = 1.0
        
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                # Sanitize headers for logging
                log_headers = self.headers.copy()
                if "Authorization" in log_headers:
                    log_headers["Authorization"] = "Bearer [REDACTED]"
                
                resp = requests.post(url, headers=self.headers, json=body, timeout=timeout)
                elapsed = time.time() - start
                
                if resp.status_code == 404:
                    error_msg = f"Model '{model_id}' not found or not accessible.\nPlease verify:\n1. Model ID is correct\n2. Model is publicly accessible\n3. Your token has required access"
                    return {"ok": False, "status": 404, "text": error_msg, "elapsed": elapsed}
                
                if resp.status_code == 401:
                    return {"ok": False, "status": 401, "text": "Authentication failed - check your HuggingFace token", "elapsed": elapsed}
                
                if resp.status_code == 503:
                    # Model loading, wait and retry
                    if attempt < retries:
                        # Model still loading, will retry
                        time.sleep(backoff)
                        backoff *= 2.0
                        continue
                    return {"ok": False, "status": 503, "text": "Model loading timeout", "elapsed": elapsed}
                
                if resp.status_code == 429:
                    # Rate limit
                    return {"ok": False, "status": 429, "text": "Rate limited", "elapsed": elapsed}
                
                if resp.status_code != 200:
                    try:
                        error_data = resp.json()
                        error_text = json.dumps(error_data)
                    except:
                        error_text = resp.text
                    return {"ok": False, "status": resp.status_code, "text": error_text, "elapsed": elapsed}
                
                data = resp.json()
                # Handle different response formats
                if isinstance(data, list) and len(data) > 0:
                    text_out = data[0].get("generated_text", "").strip()
                elif isinstance(data, dict):
                    text_out = data.get("generated_text", "").strip()
                else:
                    # Unexpected response format from API
                    return {"ok": False, "status": "parse_error", "text": "Unexpected response format", "elapsed": elapsed}
                
                return {"ok": True, "text": text_out, "elapsed": elapsed, "raw": data}
            
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    return {"ok": False, "status": "request_exception", "text": str(e), "elapsed": None}
                time.sleep(backoff)
                backoff *= 2.0
        
        return {"ok": False, "status": "unknown", "text": "Exceeded retries", "elapsed": None}

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
        # Get complete response first
        result = self.call(model_id, prompt_text, max_tokens, temperature, top_p, timeout, retries)
        
        if not result["ok"]:
            yield {"ok": False, "status": result["status"], "text": f"HuggingFace API error: {result['text']}"}
            return
        
        text = result["text"]
        if not text:
            yield {"ok": False, "status": "empty_response", "text": "No text generated"}
            return
        
        # Simulate natural streaming
        words = text.split()
        total_words = len(words)
        chunk_sizes = [1, 2, 3]  # Vary chunk sizes for more natural feel
        current_pos = 0
        start_time = time.time()
        
        try:
            while current_pos < total_words:
                # Dynamically adjust chunk size based on position
                progress = current_pos / total_words
                if progress < 0.2:
                    chunk_size = 1  # Start slow
                elif progress > 0.8:
                    chunk_size = 1  # End slow
                else:
                    chunk_size = chunk_sizes[current_pos % len(chunk_sizes)]  # Vary in middle
                
                end_pos = min(current_pos + chunk_size, total_words)
                chunk = " ".join(words[current_pos:end_pos]) + " "
                yield {"ok": True, "text": chunk}
                
                # Dynamic delay based on chunk size and progress
                delay = 0.1 if chunk_size == 1 else (0.05 if chunk_size == 2 else 0.03)
                time.sleep(delay)
                
                current_pos = end_pos
        
        except Exception as e:
            yield {"ok": False, "status": "stream_error", "text": str(e)}
            return
        
        # Final chunk with metadata
        elapsed = time.time() - start_time
        yield {
            "ok": True,
            "final": True,
            "usage": {
                "prompt_tokens": len(prompt_text.split()),
                "completion_tokens": len(text.split()),
            },
            "elapsed": elapsed
        }