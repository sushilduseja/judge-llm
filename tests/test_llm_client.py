from typing import Iterator
from src.services.llm_client import LLMClient
from src.services.responses import LLMResponse, LLMStreamChunk


class FakeAdapter:
    """Minimal adapter that satisfies the LLMClient protocol."""

    def call(self, model_id: str, prompt: str, **kwargs) -> LLMResponse:
        return LLMResponse(ok=True, text=f"called {model_id}")

    def stream(self, model_id: str, prompt: str, **kwargs) -> Iterator[LLMStreamChunk]:
        yield LLMStreamChunk(ok=True, text="chunk", final=True)


class TestLLMClientProtocol:
    def test_protocol_accepts_valid_adapter(self):
        adapter: LLMClient = FakeAdapter()
        r = adapter.call("test-model", "hello")
        assert r.ok is True
        assert "test-model" in r.text

    def test_protocol_accepts_streaming_adapter(self):
        adapter: LLMClient = FakeAdapter()
        chunks = list(adapter.stream("test-model", "hello"))
        assert len(chunks) == 1
        assert chunks[0].ok is True
        assert chunks[0].text == "chunk"
        assert chunks[0].final is True

    def test_call_passes_kwargs(self):
        class KwargCapture:
            def call(self, model_id: str, prompt: str, **kwargs) -> LLMResponse:
                self.captured = kwargs
                return LLMResponse(ok=True)
            def stream(self, model_id: str, prompt: str, **kwargs) -> Iterator[LLMStreamChunk]:
                yield LLMStreamChunk(ok=True, final=True)
        adapter: LLMClient = KwargCapture()
        adapter.call("m", "p", max_tokens=100, temperature=0.5)
        assert adapter.captured["max_tokens"] == 100
        assert adapter.captured["temperature"] == 0.5
