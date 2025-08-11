# src/config/models.py
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import os

class ModelCapability(BaseModel):
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    capabilities: List[str] = Field(default_factory=list, description="List of model capabilities")
    limitations: str = Field(..., description="Known limitations")
    best_for: str = Field(..., description="Best use cases")
    together_fallback: Optional[str] = Field(default=None, description="Together AI fallback model ID")
    
    # Add performance hints
    estimated_speed: str = Field(default="medium", description="Speed: fast, medium, slow")
    context_window: Optional[int] = Field(default=None, description="Context window size")
    
    @validator('estimated_speed')
    def validate_speed(cls, v):
        if v not in ['fast', 'medium', 'slow']:
            return 'medium'
        return v

class AppConfig(BaseModel):
    openrouter_api_key: str = Field(..., env="OPENROUTER_API_KEY")
    together_api_key: Optional[str] = Field(default=None, env="TOGETHER_API_KEY")
    
    # Smart defaults based on model capabilities
    max_tokens: int = Field(default=512, env="MAX_TOKENS")
    temperature: float = Field(default=0.0, env="TEMPERATURE")
    top_p: float = Field(default=1.0, env="TOP_P")
    http_timeout: int = Field(default=120, env="HTTP_TIMEOUT")
    
    # Judge settings
    judge_repeats: int = Field(default=1, env="JUDGE_REPEATS")
    
    # Performance settings
    enable_streaming: bool = Field(default=True, env="ENABLE_STREAMING")
    max_concurrent_requests: int = Field(default=2, env="MAX_CONCURRENT_REQUESTS")
    
    # Fallback settings
    enable_fallback: bool = Field(default=True, env="ENABLE_FALLBACK")
    fallback_timeout: int = Field(default=60, env="FALLBACK_TIMEOUT")
    
    @validator('judge_repeats')
    def validate_judge_repeats(cls, v):
        return max(1, min(v, 5))  # Clamp between 1-5
    
    @validator('max_tokens')
    def validate_max_tokens(cls, v):
        return max(64, min(v, 4096))  # Reasonable bounds
    
    @validator('temperature')
    def validate_temperature(cls, v):
        return max(0.0, min(v, 2.0))
    
    @validator('http_timeout')
    def validate_timeout(cls, v):
        return max(30, min(v, 600))  # 30s to 10min
    
    class Config:
        env_file = ".env"
        
    def get_model_settings(self, model_config: ModelCapability) -> Dict[str, Any]:
        """Get optimized settings for specific model"""
        settings = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "timeout": self.http_timeout
        }
        
        # Adjust based on model characteristics
        if "fast" in model_config.estimated_speed:
            settings["timeout"] = min(settings["timeout"], 60)
        elif "slow" in model_config.estimated_speed:
            settings["timeout"] = max(settings["timeout"], 180)
            
        # Adjust tokens for reasoning models
        if "reasoning" in model_config.capabilities or "deepseek" in model_config.id.lower():
            settings["max_tokens"] = min(settings["max_tokens"] * 2, 1024)
            
        return settings