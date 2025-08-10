import requests
import json
import time
from typing import Dict, Generator, Optional, Any

class OpenRouterClient:
    def __init__(self, api_key: str, api_url: str = "https://openrouter.ai/api/v1/chat/completions"):
        self.api_key = api_key
        self.api_url = api_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

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
            "temperature": float(temperature),
            "top_p": float(top_p),
            "max_tokens": int(max_tokens),
            "stream": stream,
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
                resp = requests.post(self.api_url, headers=self.headers, json=body, timeout=timeout)
                elapsed = time.time() - start
                
                if resp.status_code != 200:
                    return {"ok": False, "status": resp.status_code, "text": resp.text, "elapsed": elapsed}
                
                data = resp.json()
                text_out = data["choices"][0]["message"]["content"]
                usage = data.get("usage", {})
                
                return {"ok": True, "text": text_out, "usage": usage, "elapsed": elapsed, "raw": data}
            
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
        body = self._create_request_body(
            model_id, prompt_text, max_tokens, temperature, top_p, stream=True
        )
        backoff = 1.0
        
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                resp = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=body, 
                    timeout=timeout, 
                    stream=True
                )
                elapsed = time.time() - start
                
                if resp.status_code != 200:
                    yield {"ok": False, "status": resp.status_code, "text": resp.text, "elapsed": elapsed}
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
                                text = chunk["choices"][0]["delta"].get("content", "")
                                if text:
                                    yield {"ok": True, "text": text}
                            except Exception:
                                continue
                
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
