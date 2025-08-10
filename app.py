#!/usr/bin/env python3
import os
import json
from pathlib import Path
from dotenv import load_dotenv

from src.config.models import AppConfig, ModelCapability
from src.ui.main import UI

def main():
    # Load environment variables FIRST
    load_dotenv()
    
    # Validate environment variables are loaded
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if not openrouter_key:
        raise SystemExit("OPENROUTER_API_KEY not found in .env file. Please check your .env file and restart.")
    
    # Load configuration - pass the environment variable explicitly
    try:
        config = AppConfig(openrouter_api_key=openrouter_key)
    except Exception as e:
        raise SystemExit(f"Error creating AppConfig: {str(e)}")
    
    # Load model configurations
    try:
        with open('models.json', 'r') as f:
            models_data = json.load(f)
            
        models_config = {
            model["id"]: ModelCapability(**model)
            for model in models_data
        }
    except FileNotFoundError:
        raise SystemExit("models.json not found")
    except json.JSONDecodeError:
        raise SystemExit("Invalid JSON in models.json")
    except Exception as e:
        raise SystemExit(f"Error loading model configuration: {str(e)}")
    
    # Initialize output directory
    OUT_DIR = Path("arena_results")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run UI
    ui = UI(config, models_config)
    ui.render()

if __name__ == "__main__":
    main()