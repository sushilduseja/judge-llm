from unittest.mock import MagicMock
from src.services.client_manager import ClientManager
from src.services.responses import LLMResponse, LLMStreamChunk
from src.config.models import ModelCapability


mock_model = ModelCapability(
    id="llama3-70b-8192",
    name="Llama 3 70B",
    description="test",
    capabilities=[],
    limitations="none",
    best_for="testing",
    together_fallback="meta-llama/Llama-3.3-70B-Instruct-Turbo-Free",
)


class TestClientManagerCall:
    def test_uses_primary_first(self):
        primary = MagicMock()
        primary.call.return_value = LLMResponse(ok=True, text="Groq response")
        fallback = MagicMock()

        cm = ClientManager("dummy_groq_key", primary_client=primary, fallback_client=fallback)
        result = cm.call_with_fallback(mock_model, "hello")

        assert result.ok is True
        assert result.text == "Groq response"
        assert result.fallback_used is False
        primary.call.assert_called_once()
        fallback.call.assert_not_called()

    def test_falls_back_on_primary_failure(self):
        primary = MagicMock()
        primary.call.return_value = LLMResponse(
            ok=False, error="rate limited", fallback_used=False
        )
        fallback = MagicMock()
        fallback.call.return_value = LLMResponse(ok=True, text="Together fallback")

        cm = ClientManager("dummy_groq_key", primary_client=primary, fallback_client=fallback)
        result = cm.call_with_fallback(mock_model, "hello")

        assert result.ok is True
        assert result.text == "Together fallback"
        assert result.fallback_used is True
        assert result.fallback_model == mock_model.together_fallback
        primary.call.assert_called_once()
        fallback.call.assert_called_once()

    def test_returns_error_when_both_fail(self):
        primary = MagicMock()
        primary.call.return_value = LLMResponse(ok=False, error="Groq down")
        fallback = MagicMock()
        fallback.call.return_value = LLMResponse(ok=False, error="Together down")

        cm = ClientManager("dummy_groq_key", primary_client=primary, fallback_client=fallback)
        result = cm.call_with_fallback(mock_model, "hello")

        assert result.ok is False
        primary.call.assert_called_once()
        fallback.call.assert_called_once()


class TestClientManagerStream:
    def test_streams_from_primary_first(self):
        primary = MagicMock()
        primary.stream.return_value = iter(
            [
                LLMStreamChunk(ok=True, text="hello "),
                LLMStreamChunk(ok=True, text="world"),
                LLMStreamChunk(ok=True, final=True),
            ]
        )
        fallback = MagicMock()

        cm = ClientManager("dummy_groq_key", primary_client=primary, fallback_client=fallback)
        chunks = list(cm.stream_with_fallback(mock_model, "hello"))

        assert len(chunks) == 3
        assert chunks[0].text == "hello "
        assert chunks[2].final is True
        primary.stream.assert_called_once()
        fallback.stream.assert_not_called()

    def test_skips_fallback_when_no_together_fallback(self):
        model_no_fallback = ModelCapability(
            id="llama3-70b-8192",
            name="Llama 3 70B",
            description="test",
            capabilities=[],
            limitations="none",
            best_for="testing",
        )
        primary = MagicMock()
        primary.stream.return_value = iter(
            [LLMStreamChunk(ok=False, error="Groq down")]
        )
        fallback = MagicMock()

        cm = ClientManager("dummy_groq_key", primary_client=primary, fallback_client=fallback)
        chunks = list(cm.stream_with_fallback(model_no_fallback, "hello"))

        assert len(chunks) == 1
        assert chunks[0].ok is False
        fallback.stream.assert_not_called()
