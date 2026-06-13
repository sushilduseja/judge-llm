from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any

class ModelCapability(BaseModel):
    id: str = Field(..., description="Unique identifier for the model")
    name: str = Field(..., description="Human-readable name")
    description: str = Field(..., description="Detailed description")
    capabilities: List[str] = Field(default_factory=list, description="List of model capabilities")
    limitations: str = Field(..., description="Known limitations")
    best_for: str = Field(..., description="Best use cases")
    together_fallback: Optional[str] = Field(default=None, description="Together AI fallback model ID")

    estimated_speed: str = Field(default="medium", description="Speed: fast, medium, slow")
    context_window: Optional[int] = Field(default=None, description="Context window size")

    @field_validator('estimated_speed')
    @classmethod
    def validate_speed(cls, v):
        if v not in ['fast', 'medium', 'slow']:
            return 'medium'
        return v

class AppConfig(BaseSettings):
    model_config = {"env_file": ".env"}
    
    groq_api_key: str
    together_api_key: Optional[str] = None
    
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 1.0
    http_timeout: int = 120
    
    judge_repeats: int = 1
    
    
    @field_validator('judge_repeats')
    @classmethod
    def validate_judge_repeats(cls, v):
        return max(1, min(v, 5))
    
    @field_validator('max_tokens')
    @classmethod
    def validate_max_tokens(cls, v):
        return max(64, min(v, 4096))
    
    @field_validator('temperature')
    @classmethod
    def validate_temperature(cls, v):
        return max(0.0, min(v, 2.0))
    
    @field_validator('http_timeout')
    @classmethod
    def validate_timeout(cls, v):
        return max(30, min(v, 600))
        
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
        if "reasoning" in model_config.capabilities:
            settings["max_tokens"] = min(settings["max_tokens"] * 2, 1024)
            
        return settings