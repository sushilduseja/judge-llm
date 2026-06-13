# Judge LLM — AI Model Arena

Compare AI models side-by-side, get impartial judgments, and discover which model works best for your tasks. Uses Groq for primary inference with automatic fallback to Together AI.

## Quick Start

### Prerequisites
- Python 3.10+
- Groq API key (required — [console.groq.com](https://console.groq.com))
- Together AI API key (optional, for fallback)

### Setup

```bash
git clone https://github.com/sushilduseja/judge-llm.git
cd judge-llm
uv sync

cp .env.example .env
# Edit .env with your API keys

uv run streamlit run app.py
# Open http://localhost:8501
```

## Screenshot

![Screenshot](screenshot.png)

## How to Use

1. Pick two AI models to compare (defaults to smaller models; judge is always the strongest)
2. Enter a prompt — or pick from 12 pre-built code-gen templates
3. Both models respond in real-time
4. An AI judge evaluates and declares a winner
5. Use **Clear** to reset the UI to initial state

## Supported Models

| Model | Provider | Fallback |
|-------|----------|----------|
| Llama 3.3 70B | Groq | Together AI |
| Llama 3.1 8B | Groq | Together AI |
| Qwen 3 32B | Groq | Together AI |
| Llama 4 Scout 17B | Groq | Together AI |

All models automatically fall back to Together AI on failure.

## Architecture

### Client Management

```python
class ClientManager:
    def call_with_fallback(self, model_config, prompt):
        result = self._primary.call(model_config.id, prompt)
        if not result.ok and model_config.together_fallback:
            result = self._fallback.call(model_config.together_fallback, prompt)
            result.fallback_used = True
        return result
```

Groq is the primary provider. If it fails and the model config specifies a `together_fallback`, the manager silently retries with Together AI. Clients are injected as `LLMClient` protocol implementations — `GroqClient` and `TogetherAIClient` are the two adapters.

### Response Types

`LLMResponse` and `LLMStreamChunk` dataclasses replace raw dicts. Every client adapter returns typed objects, so callers access `.ok`, `.text`, `.error` instead of `.get("ok")`.

### Judgment System

Multi-round voting for consensus decisions. Each judge evaluates responses and votes on a winner. Ties and parsing failures are handled gracefully.

## Configuration

### Adding a Model

```json
// models.json
{
  "id": "groq-model-id",
  "name": "Display Name",
  "description": "What this model does best",
  "capabilities": ["coding", "reasoning"],
  "together_fallback": "fallback-model-id"
}
```

### Environment Variables

```
MAX_TOKENS=1024
TEMPERATURE=0.0
HTTP_TIMEOUT=120
JUDGE_REPEATS=3
```

All env vars are loaded automatically via `pydantic-settings` at startup.

## Testing

```bash
uv run python -m pytest tests/ -v
```

## Project Structure

```
judge-llm/
├── app.py                 # Entry point
├── models.json            # Model configurations
├── pyproject.toml         # Project config + dependencies
├── uv.lock                # Locked dependency tree
├── .env.example           # Environment template
├── src/
│   ├── config/
│   │   └── models.py          # Pydantic config models
│   ├── services/
│   │   ├── llm_client.py      # LLMClient protocol
│   │   ├── responses.py       # Typed response dataclasses
│   │   ├── groq.py            # GroqClient adapter
│   │   ├── together_ai.py     # Together AI client
│   │   ├── client_manager.py  # Fallback orchestration
│   │   └── judge.py           # Judgment system
│   └── ui/
│       └── main.py            # Streamlit interface
└── tests/                 # 30 tests across 6 files
```

## Contributing

Areas where help is useful:
- New model integrations
- UI improvements
- Analytics / comparison tracking
- Test coverage

## License

MIT
