# app.py
#!/usr/bin/env python3
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.config.models import AppConfig, ModelCapability
from src.services.client_manager import ClientManager
from src.services.model_validator import ModelValidator
from src.ui.main import UI

def load_and_validate_models(client_manager: ClientManager) -> dict:
    """Load models and validate they're accessible"""
    try:
        with open('models.json', 'r') as f:
            models_data = json.load(f)
    except FileNotFoundError:
        raise SystemExit("❌ models.json not found")
    except json.JSONDecodeError as e:
        raise SystemExit(f"❌ Invalid JSON in models.json: {e}")
    
    # Parse model configurations
    models_config = {}
    for model in models_data:
        try:
            models_config[model["id"]] = ModelCapability(**model)
        except Exception as e:
            print(f"⚠️ Skipping invalid model config {model.get('id', 'unknown')}: {e}")
    
    if not models_config:
        raise SystemExit("❌ No valid models found in models.json")
    
    print(f"📊 Loaded {len(models_config)} model configurations")
    
    # Optional: Validate models (comment out for faster startup)
    # print("🔍 Validating model accessibility...")
    # validator = ModelValidator(client_manager)
    # working_models = validator.get_working_models(models_config)
    # 
    # if len(working_models) < len(models_config):
    #     print(f"⚠️ {len(models_config) - len(working_models)} models failed validation")
    # 
    # if len(working_models) < 2:
    #     raise SystemExit("❌ Need at least 2 working models for comparison")
    # 
    # return working_models
    
    return models_config

def main():
    print("🚀 Starting Judge LLM...")
    
    # Load environment variables
    load_dotenv()
    
    # Validate API keys
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    together_key = os.getenv("TOGETHER_API_KEY")
    
    if not openrouter_key:
        raise SystemExit("❌ OPENROUTER_API_KEY not found. Please check your .env file.")
    
    if not together_key:
        print("⚠️ TOGETHER_API_KEY not found. Fallback functionality will be disabled.")
    else:
        print("✅ Both API keys found")
    
    # Create configuration
    try:
        config = AppConfig(
            openrouter_api_key=openrouter_key,
            together_api_key=together_key
        )
        print("✅ Configuration loaded")
    except Exception as e:
        raise SystemExit(f"❌ Configuration error: {e}")
    
    # Initialize services
    client_manager = ClientManager(openrouter_key, together_key)
    
    # Load and validate models
    models_config = load_and_validate_models(client_manager)
    
    # Create output directory
    OUT_DIR = Path("arena_results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Launch UI
    print("🎨 Launching UI...")
    ui = UI(config, models_config)
    ui.render()

if __name__ == "__main__":
    main()