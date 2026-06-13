from src.services.responses import LLMResponse, LLMStreamChunk


class TestLLMResponse:
    def test_constructs_with_defaults(self):
        r = LLMResponse(ok=True)
        assert r.ok is True
        assert r.text == ""
        assert r.elapsed == 0.0
        assert r.usage == {}
        assert r.raw is None
        assert r.fallback_used is False
        assert r.fallback_model is None
        assert r.error == ""

    def test_constructs_error_response(self):
        r = LLMResponse(ok=False, error="rate limited")
        assert r.ok is False
        assert r.error == "rate limited"

    def test_success_with_text(self):
        r = LLMResponse(ok=True, text="Hello world", elapsed=1.5, usage={"total_tokens": 50})
        assert r.text == "Hello world"
        assert r.elapsed == 1.5
        assert r.usage["total_tokens"] == 50

    def test_fallback_fields(self):
        r = LLMResponse(ok=True, text="fallback answer", fallback_used=True, fallback_model="llama-3.3-70b")
        assert r.fallback_used is True
        assert r.fallback_model == "llama-3.3-70b"


class TestLLMStreamChunk:
    def test_constructs_with_defaults(self):
        c = LLMStreamChunk(ok=True)
        assert c.ok is True
        assert c.text == ""
        assert c.final is False
        assert c.fallback_used is False

    def test_final_chunk(self):
        c = LLMStreamChunk(ok=True, text="done", final=True, elapsed=2.0)
        assert c.final is True
        assert c.elapsed == 2.0

    def test_error_chunk(self):
        c = LLMStreamChunk(ok=False, error="timeout")
        assert c.ok is False
        assert c.error == "timeout"
