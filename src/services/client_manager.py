from typing import Dict, Generator, Optional, Any
from .openrouter import OpenRouterClient
from .huggingface import HuggingFaceClient
from ..config.models import ModelCapability

class ClientManager:
    def __init__(self, openrouter_api_key: str, huggingface_api_key: Optional[str] = None):
        self.openrouter_client = OpenRouterClient(openrouter_api_key)
        self.huggingface_client = HuggingFaceClient(huggingface_api_key)

    def call_with_fallback(
        self,
        model_config: ModelCapability,
        prompt_text: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 120,
        retries: int = 3
    ) -> Dict[str, Any]:
        """Call OpenRouter first, fallback to HuggingFace if needed"""
        
        # Try OpenRouter first
        result = self.openrouter_client.call(
            model_config.id, 
            prompt_text, 
            max_tokens, 
            temperature, 
            top_p, 
            timeout, 
            retries
        )
        
        # Check if we need fallback - trigger on any failure if HF fallback exists
        if not result["ok"] and model_config.hf_fallback:
            print(f"OpenRouter failed, trying HuggingFace fallback: {model_config.hf_fallback}")
            
            # Try HuggingFace fallback
            hf_result = self.huggingface_client.call(
                model_config.hf_fallback,
                prompt_text,
                max_tokens,
                temperature,
                top_p,
                timeout,
                retries
            )
            
            print(f"HuggingFace result OK: {hf_result.get('ok')}")
            
            if hf_result["ok"]:
                # Mark as fallback response
                hf_result["fallback_used"] = True
                hf_result["fallback_model"] = model_config.hf_fallback
                hf_result["original_error"] = result.get("text", "OpenRouter failed")
                return hf_result
            else:
                # Both failed, return original error with fallback info
                result["fallback_attempted"] = True
                result["fallback_error"] = hf_result.get("text", "HuggingFace also failed")
                print(f"HuggingFace also failed: {hf_result.get('text')}")
        
        return result

    def stream_with_fallback(
        self,
        model_config: ModelCapability,
        prompt_text: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 120,
        retries: int = 3
    ) -> Generator[Dict[str, Any], None, None]:
        """Stream from OpenRouter first, fallback to HuggingFace if needed"""
        
        # Try OpenRouter streaming first
        openrouter_failed = False
        needs_fallback = False
        original_error = None
        
        try:
            for chunk in self.openrouter_client.stream(
                model_config.id, 
                prompt_text, 
                max_tokens, 
                temperature, 
                top_p, 
                timeout, 
                retries
            ):
                if not chunk["ok"]:
                    needs_fallback = True
                    openrouter_failed = True
                    original_error = chunk.get("text", "OpenRouter failed")
                    break
                else:
                    yield chunk
                    if chunk.get("final"):
                        return
        except Exception as e:
            needs_fallback = True
            openrouter_failed = True
            original_error = str(e)
        
        # If OpenRouter failed and we have a fallback, try HuggingFace
        if needs_fallback and model_config.hf_fallback:
            yield {"ok": True, "text": f"\n\n[Primary model failed: {original_error}. Switching to fallback: {model_config.hf_fallback}]\n\n"}
            
            try:
                for chunk in self.huggingface_client.stream(
                    model_config.hf_fallback,
                    prompt_text,
                    max_tokens,
                    temperature,
                    top_p,
                    timeout,
                    retries
                ):
                    if chunk.get("final"):
                        chunk["fallback_used"] = True
                        chunk["fallback_model"] = model_config.hf_fallback
                        chunk["original_error"] = original_error
                    yield chunk
            except Exception as e:
                yield {"ok": False, "status": "fallback_exception", "text": str(e), "elapsed": None}
        elif needs_fallback:
            # No fallback available
            yield {"ok": False, "status": "no_fallback", "text": f"Primary model failed and no fallback available: {original_error}", "elapsed": None}