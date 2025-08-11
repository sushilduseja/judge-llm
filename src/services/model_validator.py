# src/services/model_validator.py
import json
from typing import Dict, List, Tuple
from ..config.models import ModelCapability
from .client_manager import ClientManager

class ModelValidator:
    def __init__(self, client_manager: ClientManager):
        self.client_manager = client_manager
        self.test_prompt = "Say hello"

    def validate_model(self, model_config: ModelCapability) -> Tuple[bool, str]:
        """Test if a model is accessible and working"""
        try:
            result = self.client_manager.call_with_fallback(
                model_config,
                self.test_prompt,
                max_tokens=10,
                temperature=0.0,
                timeout=30  # Short timeout for validation
            )
            
            if result.get("ok"):
                return True, "✓ Working"
            else:
                error = result.get("text", "Unknown error")
                return False, f"✗ Error: {error[:100]}"
                
        except Exception as e:
            return False, f"✗ Exception: {str(e)[:100]}"

    def validate_all_models(self, models_config: Dict[str, ModelCapability]) -> Dict[str, Tuple[bool, str]]:
        """Validate all models and return status"""
        results = {}
        for model_id, model_config in models_config.items():
            print(f"Validating {model_config.name}...")
            results[model_id] = self.validate_model(model_config)
        return results

    def get_working_models(self, models_config: Dict[str, ModelCapability]) -> Dict[str, ModelCapability]:
        """Return only models that pass validation"""
        validation_results = self.validate_all_models(models_config)
        working_models = {}
        
        for model_id, model_config in models_config.items():
            is_working, _ = validation_results[model_id]
            if is_working:
                working_models[model_id] = model_config
        
        return working_models