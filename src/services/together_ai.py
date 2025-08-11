# src/services/together_ai.py
import requests
import json
import time
from typing import Dict, Generator, Optional, Any

class TogetherAIClient:
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.together.xyz/v1/chat/completions"
        self.headers = {
            "Content-Type": "application/json",
        }
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def _create_request_body(
        self, 
        model_id: str,
        prompt_text: str, 
        max_tokens: int = 256, 
        temperature: float = 0.0, 
        top_p: float = 1.0,
        stream: bool = False
    ) -> Dict[str, Any]:
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
            "top_p": float(top_p),
            "stream": stream
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
        body = self._create_request_body(model_id, prompt_text, max_tokens, temperature, top_p)
        backoff = 1.0
        
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                resp = requests.post(self.base_url, headers=self.headers, json=body, timeout=timeout)
                elapsed = time.time() - start
                
                if resp.status_code == 404:
                    error_msg = f"Model '{model_id}' not found or not accessible.\nPlease verify:\n1. Model ID is correct\n2. Model is publicly accessible\n3. Your token has required access"
                    return {"ok": False, "status": 404, "text": error_msg, "elapsed": elapsed}
                
                if resp.status_code == 401:
                    return {"ok": False, "status": 401, "text": "Authentication failed - check your Together AI token", "elapsed": elapsed}
                
                if resp.status_code == 503:
                    # Model loading, wait and retry
                    if attempt < retries:
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
                
                # Handle Together AI response format
                if "choices" in data and len(data["choices"]) > 0:
                    text_out = data["choices"][0]["message"]["content"].strip()
                    usage = data.get("usage", {})
                    return {"ok": True, "text": text_out, "usage": usage, "elapsed": elapsed, "raw": data}
                else:
                    return {"ok": False, "status": "parse_error", "text": "Unexpected response format", "elapsed": elapsed}
            
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
        body = self._create_request_body(model_id, prompt_text, max_tokens, temperature, top_p, stream=True)
        backoff = 1.0
        
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                resp = requests.post(
                    self.base_url, 
                    headers=self.headers, 
                    json=body, 
                    timeout=timeout, 
                    stream=True
                )
                elapsed = time.time() - start
                
                if resp.status_code != 200:
                    yield {"ok": False, "status": resp.status_code, "text": f"Together AI API error: {resp.text}"}
                    return
                
                usage = None
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode()
                        if decoded.startswith("data:"):
                            data = decoded[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and len(chunk["choices"]) > 0:
                                    delta = chunk["choices"][0].get("delta", {})
                                    content = delta.get("content", "")
                                    if content:
                                        yield {"ok": True, "text": content}
                            except Exception:
                                continue
                
                # Extract usage from final data
                try:
                    final_data = json.loads(resp.text)
                    usage = final_data.get("usage", {})
                except Exception:
                    pass
                
                yield {"ok": True, "final": True, "usage": usage, "elapsed": elapsed}
                return
            
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    yield {"ok": False, "status": "request_exception", "text": str(e), "elapsed": None}
                    return
                time.sleep(backoff)
                backoff *= 2.0
        
        yield {"ok": False, "status": "unknown", "text": "Exceeded retries", "elapsed": None}