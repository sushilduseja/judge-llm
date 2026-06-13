import time
from typing import Iterator, Optional, Any
from groq import Groq as GroqSDK
from .responses import LLMResponse, LLMStreamChunk


class GroqClient:
    def __init__(self, api_key: str, _client: Optional[Any] = None):
        self._client = _client if _client is not None else GroqSDK(api_key=api_key)

    def call(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 120,
        retries: int = 3,
        **kwargs,
    ) -> LLMResponse:
        body = dict(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
        )
        body.update(kwargs)
        backoff = 1.0

        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                response = self._client.chat.completions.create(timeout=timeout, **body)
                elapsed = time.time() - start

                if response.choices and len(response.choices) > 0:
                    text_out = response.choices[0].message.content or ""
                    return LLMResponse(
                        ok=True,
                        text=text_out.strip(),
                        elapsed=elapsed,
                        raw=response,
                    )
                return LLMResponse(
                    ok=False, error="No choices in response", elapsed=elapsed
                )

            except Exception as e:
                if attempt == retries:
                    return LLMResponse(
                        ok=False, error=str(e), elapsed=time.time() - start
                    )
                time.sleep(backoff)
                backoff *= 2.0

        return LLMResponse(ok=False, error="Exceeded retries")

    def stream(
        self,
        model_id: str,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.0,
        top_p: float = 1.0,
        timeout: int = 120,
        retries: int = 3,
        **kwargs,
    ) -> Iterator[LLMStreamChunk]:
        body = dict(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=float(top_p),
            stream=True,
        )
        body.update(kwargs)
        backoff = 1.0

        for attempt in range(1, retries + 1):
            try:
                start = time.time()
                stream = self._client.chat.completions.create(timeout=timeout, **body)

                for chunk in stream:
                    if (
                        chunk.choices
                        and len(chunk.choices) > 0
                        and chunk.choices[0].delta.content
                    ):
                        yield LLMStreamChunk(
                            ok=True, text=chunk.choices[0].delta.content
                        )

                elapsed = time.time() - start
                yield LLMStreamChunk(ok=True, final=True, elapsed=elapsed)
                return

            except Exception as e:
                if attempt == retries:
                    yield LLMStreamChunk(ok=False, error=str(e))
                    return
                time.sleep(backoff)
                backoff *= 2.0

        yield LLMStreamChunk(ok=False, error="Exceeded retries")
