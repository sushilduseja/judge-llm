# src/ui/main.py
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container
import json
import queue
import threading
import time
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

from ..config.models import AppConfig, ModelCapability
from ..services.client_manager import ClientManager
from ..services.judge import JudgeService

class UI:
    def __init__(self, config: AppConfig, models_config: Dict[str, ModelCapability]):
        self.config = config
        self.models_config = models_config
        self.client_manager = ClientManager(
            config.openrouter_api_key, 
            config.together_api_key
        )
        self.judge_service = JudgeService(self.client_manager)
        self.setup_page()

    def setup_page(self):
        st.set_page_config(
            page_title="Model Compare and Judge",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
            <style>
                .stApp {
                    max-width: 1200px;
                    margin: 0 auto;
                }
                .stButton > button {
                    width: 100%;
                    border-radius: 5px;
                    height: 3em;
                    background-color: #0066cc;
                    color: white;
                }
                .model-card {
                    border: 1px solid #ddd;
                    border-radius: 5px;
                    padding: 1rem;
                    margin-bottom: 1rem;
                }
                .main-content {
                    padding: 2rem;
                }
                .fallback-indicator {
                    background-color: #fff3cd;
                    border: 1px solid #ffeaa7;
                    border-radius: 4px;
                    padding: 0.5rem;
                    margin: 0.5rem 0;
                    font-size: 0.85em;
                }
            </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        st.title("🤖 Model Compare and Judge")
        st.caption(
            "Compare different AI models and get an impartial judgment on their performance. "
            "Automatic fallback to Together AI when OpenRouter is unavailable."
        )

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Model Selection with Cards
            st.subheader("Model Selection")
            
            with stylable_container(
                key="model_a_container",
                css_styles="""
                    {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                    }
                """
            ):
                model_a = st.selectbox(
                    "Model A",
                    options=list(self.models_config.keys()),
                    format_func=lambda x: self.models_config[x].name,
                    key="model_a"
                )
                if model_a:
                    st.caption(self.models_config[model_a].description)
                    st.markdown("**Best for:** " + self.models_config[model_a].best_for)
                    if self.models_config[model_a].together_fallback:
                        st.markdown(f"**Fallback:** {self.models_config[model_a].together_fallback}")
            
            with stylable_container(
                key="model_b_container",
                css_styles="""
                    {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                    }
                """
            ):
                model_b = st.selectbox(
                    "Model B",
                    options=list(self.models_config.keys()),
                    format_func=lambda x: self.models_config[x].name,
                    key="model_b"
                )
                if model_b:
                    st.caption(self.models_config[model_b].description)
                    st.markdown("**Best for:** " + self.models_config[model_b].best_for)
                    if self.models_config[model_b].together_fallback:
                        st.markdown(f"**Fallback:** {self.models_config[model_b].together_fallback}")
            
            # Judge Settings
            st.subheader("Judge Settings")
            judge_model = st.selectbox(
                "Judge Model",
                options=list(self.models_config.keys()),
                format_func=lambda x: self.models_config[x].name,
                index=2  # Default to DeepSeek R1
            )
            
            # Advanced Settings
            with st.expander("Advanced Settings"):
                judge_repeats = st.number_input(
                    "Judge repeats",
                    min_value=1,
                    max_value=5,
                    value=self.config.judge_repeats
                )
                max_tokens = st.number_input(
                    "Max tokens",
                    min_value=64,
                    max_value=2048,
                    value=self.config.max_tokens,
                    step=64
                )
                temperature = st.number_input(
                    "Temperature",
                    min_value=0.0,
                    max_value=1.0,
                    value=self.config.temperature,
                    step=0.05,
                    format="%.2f"
                )
                top_p = st.number_input(
                    "Top p",
                    min_value=0.1,
                    max_value=1.0,
                    value=self.config.top_p,
                    step=0.05,
                    format="%.2f"
                )
                http_timeout = st.number_input(
                    "HTTP timeout seconds",
                    min_value=30,
                    max_value=600,
                    value=self.config.http_timeout,
                    step=10
                )
            
            return {
                "model_a": model_a,
                "model_b": model_b,
                "judge_model": judge_model,
                "judge_repeats": judge_repeats,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "http_timeout": http_timeout
            }

    def stream_response_safe(self, model_config: ModelCapability, prompt: str, settings: Dict[str, Any], result_queue: queue.Queue, result_key: str):
        """Thread-safe streaming with fallback support"""
        try:
            full_text = ""
            usage_info = {}
            elapsed_time = 0
            fallback_used = False
            fallback_model = None
            
            for chunk in self.client_manager.stream_with_fallback(
                model_config,
                prompt,
                max_tokens=settings["max_tokens"],
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                timeout=settings["http_timeout"]
            ):
                if chunk["ok"]:
                    if "text" in chunk:
                        full_text += chunk["text"]
                        # Send partial update via queue
                        result_queue.put({
                            result_key: {
                                "text": full_text,
                                "partial": True,
                                "error": None,
                                "fallback_used": chunk.get("fallback_used", False),
                                "fallback_model": chunk.get("fallback_model")
                            }
                        })
                    
                    if chunk.get("final"):
                        usage_info = chunk.get("usage", {})
                        elapsed_time = chunk.get("elapsed", 0)
                        fallback_used = chunk.get("fallback_used", False)
                        fallback_model = chunk.get("fallback_model")
                        
                        # Send final update
                        result_queue.put({
                            result_key: {
                                "text": full_text,
                                "partial": False,
                                "error": None,
                                "usage": usage_info,
                                "elapsed": elapsed_time,
                                "fallback_used": fallback_used,
                                "fallback_model": fallback_model
                            }
                        })
                        return
                else:
                    # Send error via queue
                    error_msg = chunk.get('text', 'Unknown error')
                    result_queue.put({
                        result_key: {
                            "text": None,
                            "partial": False,
                            "error": error_msg
                        }
                    })
                    return
                    
        except Exception as e:
            result_queue.put({
                result_key: {
                    "text": None,
                    "partial": False,
                    "error": str(e)
                }
            })

    def run_comparison_safe(self, prompt: str, settings: Dict[str, Any], placeholder_a, placeholder_b):
        """Safe comparison with fallback support using queue-based communication"""
        
        # Get model configs
        model_config_a = self.models_config[settings["model_a"]]
        model_config_b = self.models_config[settings["model_b"]]
        
        # Create queue for thread communication
        result_queue = queue.Queue()
        
        # Start both model requests in threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_a = executor.submit(
                self.stream_response_safe, 
                model_config_a, 
                prompt, 
                settings, 
                result_queue, 
                "a"
            )
            future_b = executor.submit(
                self.stream_response_safe, 
                model_config_b, 
                prompt, 
                settings, 
                result_queue, 
                "b"
            )
            
            # Track completion and results
            results = {"a": None, "b": None}
            completed = {"a": False, "b": False}
            fallback_info = {"a": None, "b": None}
            
            # Progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
            
            # Process queue updates in main thread (UI-safe)
            while not all(completed.values()):
                try:
                    # Get update from queue with timeout
                    update = result_queue.get(timeout=0.1)
                    
                    for key, data in update.items():
                        if data["error"]:
                            # Handle error
                            if key == "a":
                                placeholder_a.error(f"Error: {data['error']}")
                            else:
                                placeholder_b.error(f"Error: {data['error']}")
                            completed[key] = True
                            results[key] = None
                        
                        elif data["text"] is not None:
                            # Check for fallback usage
                            if data.get("fallback_used"):
                                fallback_info[key] = {
                                    "used": True,
                                    "model": data.get("fallback_model"),
                                    "original_model": settings[f"model_{key}"]
                                }
                            
                            # Update UI with response text
                            display_text = data["text"]
                            
                            # Add fallback indicator if needed
                            if fallback_info[key] and fallback_info[key]["used"]:
                                fallback_msg = f"🔄 **Using fallback:** {fallback_info[key]['model']}\n\n"
                                display_text = fallback_msg + display_text
                            
                            if key == "a":
                                placeholder_a.markdown(display_text)
                            else:
                                placeholder_b.markdown(display_text)
                            
                            # Check if this is the final update
                            if not data["partial"]:
                                completed[key] = True
                                results[key] = data["text"]
                                
                                # Store usage info in session state
                                if "usage" in data:
                                    st.session_state[f"{settings[f'model_{key}']}_usage"] = data["usage"]
                                if "elapsed" in data:
                                    st.session_state[f"{settings[f'model_{key}']}_time"] = data["elapsed"]
                        
                        # Update progress
                        progress = sum(completed.values()) / len(completed)
                        progress_bar.progress(progress)
                        
                        # Update status text
                        if all(completed.values()):
                            status_text.text("✅ Both models completed!")
                        else:
                            active_models = [k for k, v in completed.items() if not v]
                            status_text.text(f"⏳ Running: Model {', '.join(active_models).upper()}")
                
                except queue.Empty:
                    # No updates available, continue waiting
                    continue
            
            # Clean up progress indicators
            progress_container.empty()
            
            return results, fallback_info

    def render_main(self, settings: Dict[str, Any]):
        st.markdown("## 💭 Prompt")
        prompt = st.text_area(
            "Enter your prompt",
            height=240,
            placeholder="Write a function, debug this code, or describe a small project"
        )
        
        # Validate models are different
        if settings["model_a"] == settings["model_b"]:
            st.error("Please select different models for comparison")
            return
        
        if st.button("🏃‍♂️ Compare and Judge", use_container_width=True):
            if not prompt.strip():
                st.warning("Please enter a prompt")
                return
            
            # Create layout
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown(f"### Model A: {self.models_config[settings['model_a']].name}")
                placeholder_a = st.empty()
                
            with col_right:
                st.markdown(f"### Model B: {self.models_config[settings['model_b']].name}")
                placeholder_b = st.empty()
            
            # Run safe comparison with fallback support
            results, fallback_info = self.run_comparison_safe(prompt, settings, placeholder_a, placeholder_b)
            
            # Show fallback information if used
            if any(info and info.get("used") for info in fallback_info.values()):
                st.info("🔄 One or more models used fallback providers due to OpenRouter limitations.")
            
            # Proceed with judgment if both responses succeeded
            if results["a"] is not None and results["b"] is not None:
                st.markdown("### 👨‍⚖️ Judgment")
                with st.spinner("The judge is evaluating responses..."):
                    judge_model_config = self.models_config[settings["judge_model"]]
                    judgment = self.judge_service.judge_majority(
                        prompt,
                        results["a"],
                        results["b"],
                        judge_model_config,
                        settings["judge_repeats"],
                        settings["http_timeout"]
                    )
                
                if judgment["ok"]:
                    winner = judgment["decision"]
                    if winner == "TIE":
                        st.info("🤝 The responses are equally good!")
                    else:
                        winner_name = self.models_config[
                            settings["model_a"] if winner == "A" else settings["model_b"]
                        ].name
                        st.success(f"🏆 The winner is: {winner_name}")
                    
                    # Show voting details
                    with st.expander("Voting Details"):
                        st.json(judgment["counts"])
                        
                        # Show if judge used fallback
                        if "errors" in judgment:
                            st.warning("Some judgment attempts had errors:")
                            for error in judgment["errors"]:
                                st.text(error)
                else:
                    st.error("Failed to get judgment: " + str(judgment.get("reason")))
            else:
                st.error("Cannot proceed with judgment due to model response errors.")

    def render(self):
        self.render_header()
        settings = self.render_sidebar()
        self.render_main(settings)