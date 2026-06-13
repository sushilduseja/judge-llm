from typing import Iterator, Optional
from ..config.models import ModelCapability
from .llm_client import LLMClient
from .groq import GroqClient
from .together_ai import TogetherAIClient
from .responses import LLMResponse, LLMStreamChunk


class ClientManager:
    def __init__(
        self,
        groq_api_key: str,
        together_api_key: Optional[str] = None,
        primary_client: Optional[LLMClient] = None,
        fallback_client: Optional[LLMClient] = None,
    ):
        self._primary = primary_client or GroqClient(groq_api_key)
        if fallback_client:
            self._fallback = fallback_client
        elif together_api_key:
            self._fallback = TogetherAIClient(together_api_key)
        else:
            self._fallback = None

    def call_with_fallback(
        self,
        model_config: ModelCapability,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 120,
        retries: int = 3,
    ) -> LLMResponse:
        result = self._primary.call(
            model_config.id, prompt, max_tokens, temperature, top_p, timeout, retries
        )

        if not result.ok and self._fallback and model_config.together_fallback:
            original_error = result.error
            result = self._fallback.call(
                model_config.together_fallback,
                prompt,
                max_tokens,
                temperature,
                top_p,
                timeout,
                retries,
            )
            if result.ok:
                result.fallback_used = True
                result.fallback_model = model_config.together_fallback
                result.error = original_error

        return result

    def stream_with_fallback(
        self,
        model_config: ModelCapability,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 120,
        retries: int = 3,
    ) -> Iterator[LLMStreamChunk]:
        primary_failed = False
        original_error = None

        try:
            for chunk in self._primary.stream(
                model_config.id,
                prompt,
                max_tokens,
                temperature,
                top_p,
                timeout,
                retries,
            ):
                if not chunk.ok:
                    primary_failed = True
                    original_error = chunk.error
                    break
                yield chunk
                if chunk.final:
                    return
        except Exception as e:
            primary_failed = True
            original_error = str(e)

        if primary_failed and self._fallback and model_config.together_fallback:
            try:
                for chunk in self._fallback.stream(
                    model_config.together_fallback,
                    prompt,
                    max_tokens,
                    temperature,
                    top_p,
                    timeout,
                    retries,
                ):
                    chunk.fallback_used = True
                    chunk.fallback_model = model_config.together_fallback
                    yield chunk
            except Exception as e:
                yield LLMStreamChunk(ok=False, error=str(e))
        elif primary_failed:
            yield LLMStreamChunk(ok=False, error=original_error or "Primary failed")
