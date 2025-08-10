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
                print(f"HF API call to: {url}")
                print(f"Headers: {self.headers}")
                print(f"Body: {json.dumps(body, indent=2)}")
                
                resp = requests.post(url, headers=self.headers, json=body, timeout=timeout)
                elapsed = time.time() - start
                
                print(f"HF Response status: {resp.status_code}")
                print(f"HF Response text: {resp.text[:500]}")
                
                if resp.status_code == 503:
                    # Model loading, wait and retry
                    if attempt < retries:
                        print(f"Model loading, retrying in {backoff}s...")
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
                print(f"HF Success response: {json.dumps(data, indent=2)[:500]}")
                
                # Handle different response formats
                if isinstance(data, list) and len(data) > 0:
                    text_out = data[0].get("generated_text", "").strip()
                elif isinstance(data, dict):
                    text_out = data.get("generated_text", "").strip()
                else:
                    print(f"Unexpected HF response format: {type(data)}")
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
        # HuggingFace Inference API doesn't support streaming for most models
        # Fall back to regular call and simulate streaming
        result = self.call(model_id, prompt_text, max_tokens, temperature, top_p, timeout, retries)
        
        if not result["ok"]:
            yield result
            return
        
        text = result["text"]
        # Simulate streaming by yielding chunks
        if text:
            words = text.split()
            current_text = ""
            for i, word in enumerate(words):
                current_text += word + " "
                yield {"ok": True, "text": word + " "}
                # Small delay to simulate streaming
                time.sleep(0.05)
        
        # Yield final chunk
        yield {"ok": True, "final": True, "usage": {}, "elapsed": result.get("elapsed", 0)}