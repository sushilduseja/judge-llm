from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class LLMResponse:
    ok: bool
    text: str = ""
    elapsed: float = 0.0
    usage: dict = field(default_factory=dict)
    raw: Any = None
    fallback_used: bool = False
    fallback_model: Optional[str] = None
    error: str = ""


@dataclass
class LLMStreamChunk:
    ok: bool
    text: str = ""
    final: bool = False
    elapsed: float = 0.0
    usage: dict = field(default_factory=dict)
    fallback_used: bool = False
    fallback_model: Optional[str] = None
    error: str = ""
