from pydantic import BaseModel, Field
from typing import List, Optional

class ModelCapability(BaseModel):
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    capabilities: List[str] = Field(default_factory=list, description="List of model capabilities")
    limitations: str = Field(..., description="Known limitations")
    best_for: str = Field(..., description="Best use cases")
    performance_tier: str = Field(default="fast", description="Performance tier: ultra_fast, fast, slow, very_slow")
    avg_response_time: str = Field(default="3-5s", description="Average response time estimate")

class AppConfig(BaseModel):
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    model_a: str = Field(default="deepseek/deepseek-chat-v3-0324:free", env="MODEL_A")
    model_b: str = Field(default="qwen/qwen3-coder:free", env="MODEL_B")
    judge_model: str = Field(default="google/gemini-2.0-flash-exp:free", env="JUDGE_MODEL")  # Changed to fast model
    judge_repeats: int = Field(default=1, env="JUDGE_REPEATS")
    max_tokens: int = Field(default=512, env="MAX_TOKENS")
    temperature: float = Field(default=0.1, env="TEMPERATURE")  # Slightly increased for better variety
    top_p: float = Field(default=0.95, env="TOP_P")  # Slightly reduced for more focused responses
    http_timeout: int = Field(default=60, env="HTTP_TIMEOUT")  # Reduced from 120

    class Config:
        env_file = ".env"