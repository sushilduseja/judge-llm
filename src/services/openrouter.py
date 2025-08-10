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
            "HTTP-Referer": "https://your-app.com",  # Required for some free models
            "X-Title": "Judge LLM App"  # Required for some free models
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
        # Optimize parameters for free models
        return {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": max(0.0, min(1.0, float(temperature))),  # Clamp values
            "top_p": max(0.1, min(1.0, float(top_p))),
            "max_tokens": max(1, min(2048, int(max_tokens))),  # Reasonable limits
            "stream": stream,
        }

    def call(
        self, 
        model_id: str, 
        prompt_text: str, 
        max_tokens: int = 256, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        timeout: int = 60,  # Reduced default timeout
        retries: int = 2     # Reduced retries for speed
    ) -> Dict[str, Any]:
        body = self._create_request_body(model_id, prompt_text, max_tokens, temperature, top_p)
        backoff = 0.5  # Shorter initial backoff
        
        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                resp = requests.post(
                    self.api_url, 
                    headers=self.headers, 
                    json=body, 
                    timeout=timeout
                )
                elapsed = time.time() - start
                
                # Handle different HTTP status codes appropriately
                if resp.status_code == 429:  # Rate limited
                    if attempt == retries:
                        return {
                            "ok": False, 
                            "status": 429, 
                            "text": "Rate limited - please wait and retry", 
                            "elapsed": elapsed
                        }
                    time.sleep(backoff * 2)  # Longer wait for rate limits
                    backoff *= 2.0
                    continue
                
                elif resp.status_code == 502 or resp.status_code == 503:  # Server errors
                    if attempt == retries:
                        return {
                            "ok": False, 
                            "status": resp.status_code, 
                            "text": "Server temporarily unavailable", 
                            "elapsed": elapsed
                        }
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
                
                elif resp.status_code != 200:
                    return {
                        "ok": False, 
                        "status": resp.status_code, 
                        "text": f"API error: {resp.text[:200]}", 
                        "elapsed": elapsed
                    }
                
                # Parse successful response
                try:
                    data = resp.json()
                except json.JSONDecodeError:
                    return {
                        "ok": False, 
                        "status": "json_error", 
                        "text": "Invalid JSON response", 
                        "elapsed": elapsed
                    }
                
                # Extract response safely
                if "choices" not in data or not data["choices"]:
                    return {
                        "ok": False, 
                        "status": "no_choices", 
                        "text": "No response choices returned", 
                        "elapsed": elapsed,
                        "raw": data
                    }
                
                choice = data["choices"][0]
                message = choice.get("message", {})
                text_out = message.get("content", "")
                
                # Handle reasoning models (like DeepSeek R1)
                if not text_out and "reasoning" in message:
                    text_out = message["reasoning"]
                
                usage = data.get("usage", {})
                
                return {
                    "ok": True, 
                    "text": text_out, 
                    "usage": usage, 
                    "elapsed": elapsed, 
                    "raw": data
                }
            
            except requests.exceptions.Timeout:
                if attempt == retries:
                    return {
                        "ok": False, 
                        "status": "timeout", 
                        "text": f"Request timed out after {timeout}s", 
                        "elapsed": timeout
                    }
                time.sleep(backoff)
                backoff *= 1.5
                
            except requests.exceptions.RequestException as e:
                if attempt == retries:
                    return {
                        "ok": False, 
                        "status": "request_exception", 
                        "text": f"Network error: {str(e)[:100]}", 
                        "elapsed": None
                    }
                time.sleep(backoff)
                backoff *= 1.5
        
        return {"ok": False, "status": "unknown", "text": "Exceeded retries", "elapsed": None}

    def stream(
        self, 
        model_id: str, 
        prompt_text: str, 
        max_tokens: int = 256, 
        temperature: float = 0.0, 
        top_p: float = 1.0, 
        timeout: int = 60, 
        retries: int = 2
    ) -> Generator[Dict[str, Any], None, None]:
        body = self._create_request_body(
            model_id, prompt_text, max_tokens, temperature, top_p, stream=True
        )
        backoff = 0.5
        
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
                
                if resp.status_code == 429:  # Rate limited
                    yield {
                        "ok": False, 
                        "status": 429, 
                        "text": "Rate limited - please wait", 
                        "elapsed": elapsed
                    }
                    if attempt < retries:
                        time.sleep(backoff * 2)
                        backoff *= 2.0
                        continue
                    return
                
                if resp.status_code != 200:
                    yield {
                        "ok": False, 
                        "status": resp.status_code, 
                        "text": f"API error: {resp.status_code}", 
                        "elapsed": elapsed
                    }
                    return
                
                # Process streaming response
                accumulated_text = ""
                usage = None
                
                for line in resp.iter_lines():
                    if line:
                        decoded = line.decode('utf-8')
                        if decoded.startswith("data:"):
                            data = decoded[5:].strip()
                            if data == "[DONE]":
                                break
                            try:
                                chunk = json.loads(data)
                                if "choices" in chunk and chunk["choices"]:
                                    delta = chunk["choices"][0].get("delta", {})
                                    text = delta.get("content", "")
                                    if text:
                                        accumulated_text += text
                                        yield {"ok": True, "text": text}
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                
                # Send final response with metadata
                final_elapsed = time.time() - start
                yield {
                    "ok": True, 
                    "final": True, 
                    "text": accumulated_text,
                    "usage": usage, 
                    "elapsed": final_elapsed
                }
                return
            
            except requests.exceptions.Timeout:
                yield {
                    "ok": False, 
                    "status": "timeout", 
                    "text": f"Stream timed out after {timeout}s", 
                    "elapsed": timeout
                }
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
                return
                
            except requests.exceptions.RequestException as e:
                yield {
                    "ok": False, 
                    "status": "request_exception", 
                    "text": f"Stream error: {str(e)[:100]}", 
                    "elapsed": None
                }
                if attempt < retries:
                    time.sleep(backoff)
                    backoff *= 1.5
                    continue
                return
        
        yield {"ok": False, "status": "unknown", "text": "Stream exceeded retries", "elapsed": None}