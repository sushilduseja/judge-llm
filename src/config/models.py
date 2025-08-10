from pydantic import BaseModel, Field
from typing import List, Optional

class ModelCapability(BaseModel):
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    capabilities: List[str] = Field(default_factory=list, description="List of model capabilities")
    limitations: str = Field(..., description="Known limitations")
    best_for: str = Field(..., description="Best use cases")

class AppConfig(BaseModel):
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    model_a: str = Field(default="mistralai/devstral-small-2505:free", env="MODEL_A")
    model_b: str = Field(default="qwen/qwen3-8b:free", env="MODEL_B")
    judge_model: str = Field(default="deepseek/deepseek-r1:free", env="JUDGE_MODEL")
    judge_repeats: int = Field(default=1, env="JUDGE_REPEATS")
    max_tokens: int = Field(default=512, env="MAX_TOKENS")
    temperature: float = Field(default=0.0, env="TEMPERATURE")
    top_p: float = Field(default=1.0, env="TOP_P")
    http_timeout: int = Field(default=120, env="HTTP_TIMEOUT")

    class Config:
        env_file = ".env"
