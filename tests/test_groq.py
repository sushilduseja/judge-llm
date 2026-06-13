from unittest.mock import MagicMock
from src.services.llm_client import LLMClient
from src.services.groq import GroqClient
from src.services.responses import LLMResponse, LLMStreamChunk


class TestGroqClientProtocol:
    def test_satisfies_llmclient_protocol(self):
        client: LLMClient = GroqClient(api_key="test-key")
        assert isinstance(client, GroqClient)


class TestGroqClientCall:
    def test_returns_llmresponse_on_success(self):
        mock_sdk = MagicMock()
        mock_choice = MagicMock()
        mock_choice.message.content = "Hello from Groq"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_sdk.chat.completions.create.return_value = mock_response

        client = GroqClient(api_key="test-key", _client=mock_sdk)
        result = client.call("llama3-70b-8192", "Say hello")

        assert isinstance(result, LLMResponse)
        assert result.ok is True
        assert result.text == "Hello from Groq"

    def test_passes_model_and_prompt(self):
        mock_sdk = MagicMock()
        client = GroqClient(api_key="test-key", _client=mock_sdk)
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_sdk.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client.call("llama3-70b-8192", "test prompt")
        args, kwargs = mock_sdk.chat.completions.create.call_args
        assert kwargs["model"] == "llama3-70b-8192"
        assert kwargs["messages"] == [{"role": "user", "content": "test prompt"}]

    def test_passes_kwargs(self):
        mock_sdk = MagicMock()
        client = GroqClient(api_key="test-key", _client=mock_sdk)
        mock_choice = MagicMock()
        mock_choice.message.content = "ok"
        mock_sdk.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        client.call("m", "p", max_tokens=200, temperature=0.5, top_p=0.9)
        _, kwargs = mock_sdk.chat.completions.create.call_args
        assert kwargs["max_tokens"] == 200
        assert kwargs["temperature"] == 0.5
        assert kwargs["top_p"] == 0.9

    def test_returns_error_on_exception(self):
        mock_sdk = MagicMock()
        mock_sdk.chat.completions.create.side_effect = Exception("API error")

        client = GroqClient(api_key="test-key", _client=mock_sdk)
        result = client.call("llama3-70b-8192", "hello")

        assert isinstance(result, LLMResponse)
        assert result.ok is False
        assert "API error" in result.error


class TestGroqClientStream:
    def test_yields_llmstreamchunks(self):
        mock_sdk = MagicMock()
        mock_chunk_1 = MagicMock()
        mock_chunk_1.choices = [MagicMock()]
        mock_chunk_1.choices[0].delta.content = "Hello "
        mock_chunk_2 = MagicMock()
        mock_chunk_2.choices = [MagicMock()]
        mock_chunk_2.choices[0].delta.content = "world"
        mock_sdk.chat.completions.create.return_value = [mock_chunk_1, mock_chunk_2]

        client = GroqClient(api_key="test-key", _client=mock_sdk)
        chunks = list(client.stream("llama3-70b-8192", "Say hello"))

        assert len(chunks) == 3  # two text chunks + final
        for chunk in chunks:
            assert isinstance(chunk, LLMStreamChunk)
        assert chunks[0].text == "Hello "
        assert chunks[1].text == "world"
        assert chunks[2].final is True
