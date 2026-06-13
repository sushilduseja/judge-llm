from unittest.mock import MagicMock
from src.services.llm_client import LLMClient
from src.services.together_ai import TogetherAIClient
from src.services.responses import LLMResponse, LLMStreamChunk


class TestTogetherClientProtocol:
    def test_satisfies_llmclient_protocol(self):
        client: LLMClient = TogetherAIClient(api_key="test-key")
        assert isinstance(client, TogetherAIClient)


class TestTogetherClientCall:
    def test_returns_llmresponse_on_success(self):
        mock_sdk = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Together"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_sdk.chat.completions.create.return_value = mock_response

        client = TogetherAIClient(api_key="test-key", _client=mock_sdk)
        result = client.call("llama-3.3-70b", "Say hello")

        assert isinstance(result, LLMResponse)
        assert result.ok is True
        assert result.text == "Hello from Together"

    def test_returns_error_on_exception(self):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = Exception("Together error")

        client = TogetherAIClient(api_key="test-key", _client=mock_sdk)
        result = client.call("llama-3.3-70b", "hello")

        assert isinstance(result, LLMResponse)
        assert result.ok is False
        assert "Together error" in result.error

    def test_model_not_found_404(self):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = Exception("404 not found")

        client = TogetherAIClient(api_key="test-key", _client=mock_sdk)
        result = client.call("bad-model", "hello")

        assert result.ok is False
        assert "not found" in result.error


class TestTogetherClientStream:
    def test_yields_llmstreamchunks(self):
        mock_sdk = MagicMock()
        mock_chunk_1 = MagicMock()
        mock_chunk_1.choices = [MagicMock()]
        mock_chunk_1.choices[0].delta.content = "Hello "
        mock_chunk_2 = MagicMock()
        mock_chunk_2.choices = [MagicMock()]
        mock_chunk_2.choices[0].delta.content = "world"
        mock_sdk.chat.completions.create.return_value = [mock_chunk_1, mock_chunk_2]

        client = TogetherAIClient(api_key="test-key", _client=mock_sdk)
        chunks = list(client.stream("llama-3.3-70b", "Say hello"))

        assert len(chunks) == 3
        for chunk in chunks:
            assert isinstance(chunk, LLMStreamChunk)
        assert chunks[0].text == "Hello "
        assert chunks[2].final is True
