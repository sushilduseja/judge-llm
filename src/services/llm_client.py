from typing import Iterator, Protocol
from .responses import LLMResponse, LLMStreamChunk


class LLMClient(Protocol):
    def call(self, model_id: str, prompt: str, **kwargs) -> LLMResponse:
        ...

    def stream(
        self, model_id: str, prompt: str, **kwargs
    ) -> Iterator[LLMStreamChunk]:
        ...
