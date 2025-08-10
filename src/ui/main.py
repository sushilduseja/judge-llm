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
from ..services.openrouter import OpenRouterClient
from ..services.judge import JudgeService

class UI:
    def __init__(self, config: AppConfig, models_config: Dict[str, ModelCapability]):
        self.config = config
        self.models_config = models_config
        self.client = OpenRouterClient(config.openrouter_api_key)
        self.judge_service = JudgeService(self.client)
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
            </style>
        """, unsafe_allow_html=True)

    def render_header(self):
        st.title("🤖 Model Compare and Judge")
        st.caption(
            "Compare different AI models and get an impartial judgment on their performance"
        )

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Performance mode selector
            performance_mode = st.radio(
                "⚡ Performance Mode",
                ["Ultra Fast", "Balanced", "Quality"],
                index=1,  # Default to Balanced
                help="Ultra Fast: 1-3s response, Balanced: 3-6s, Quality: 6-15s"
            )
            
            # Filter models by performance mode
            if performance_mode == "Ultra Fast":
                available_models = {k: v for k, v in self.models_config.items() 
                                if v.performance_tier in ["ultra_fast", "fast"]}
                default_judge = "google/gemini-2.0-flash-exp:free"
            elif performance_mode == "Balanced":
                available_models = {k: v for k, v in self.models_config.items() 
                                if v.performance_tier in ["ultra_fast", "fast"]}
                default_judge = "deepseek/deepseek-chat-v3-0324:free"
            else:  # Quality
                available_models = self.models_config
                default_judge = "deepseek/deepseek-chat-v3-0324:free"  # Avoid R1 for judge
            
            st.subheader("Model Selection")
            
            # Smart defaults based on performance mode
            model_keys = list(available_models.keys())
            default_a_idx = 0 if model_keys else 0
            default_b_idx = min(1, len(model_keys) - 1) if len(model_keys) > 1 else 0
            
            with stylable_container(
                key="model_a_container",
                css_styles="""
                    {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        background-color: #f8f9ff;
                    }
                """
            ):
                model_a = st.selectbox(
                    "Model A",
                    options=model_keys,
                    format_func=lambda x: f"{available_models[x].name} (~{available_models[x].avg_response_time})",
                    index=default_a_idx,
                    key="model_a"
                )
                if model_a:
                    st.caption(f"🎯 {available_models[model_a].best_for}")
            
            with stylable_container(
                key="model_b_container",
                css_styles="""
                    {
                        border: 1px solid #ddd;
                        border-radius: 5px;
                        padding: 1rem;
                        margin-bottom: 1rem;
                        background-color: #fff8f0;
                    }
                """
            ):
                model_b = st.selectbox(
                    "Model B",
                    options=model_keys,
                    format_func=lambda x: f"{available_models[x].name} (~{available_models[x].avg_response_time})",
                    index=default_b_idx,
                    key="model_b"
                )
                if model_b:
                    st.caption(f"🎯 {available_models[model_b].best_for}")
            
            # Judge Settings with performance consideration
            st.subheader("Judge Settings")
            
            # Filter judge models to exclude slow reasoning models
            judge_options = {k: v for k, v in available_models.items() 
                            if v.performance_tier in ["ultra_fast", "fast"]}
            
            judge_model = st.selectbox(
                "Judge Model",
                options=list(judge_options.keys()),
                format_func=lambda x: f"{judge_options[x].name} (~{judge_options[x].avg_response_time})",
                index=list(judge_options.keys()).index(default_judge) if default_judge in judge_options else 0,
                help="Using fast models for judging to maintain responsiveness"
            )
            
            # Simplified settings based on performance mode
            if performance_mode == "Ultra Fast":
                judge_repeats = 1
                max_tokens = 256
                temperature = 0.0
                top_p = 1.0
                http_timeout = 30
                st.info("⚡ Ultra Fast mode: Optimized for speed")
            elif performance_mode == "Balanced":
                judge_repeats = 1
                max_tokens = 512
                temperature = 0.1
                top_p = 0.95
                http_timeout = 60
                st.info("⚖️ Balanced mode: Good speed/quality balance")
            else:  # Quality
                with st.expander("Advanced Settings"):
                    judge_repeats = st.number_input("Judge repeats", 1, 3, 1)
                    max_tokens = st.number_input("Max tokens", 256, 1024, 512, step=64)
                    temperature = st.number_input("Temperature", 0.0, 0.3, 0.1, step=0.05)
                    top_p = st.number_input("Top p", 0.8, 1.0, 0.95, step=0.05)
                    http_timeout = st.number_input("Timeout (s)", 30, 120, 90, step=10)
                st.info("🎯 Quality mode: Best results, slower responses")
            
            return {
                "model_a": model_a,
                "model_b": model_b,
                "judge_model": judge_model,
                "judge_repeats": judge_repeats,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "http_timeout": http_timeout,
                "performance_mode": performance_mode
            }

    def stream_response_safe(self, model_id: str, prompt: str, settings: Dict[str, Any], result_queue: queue.Queue, result_key: str):
        """Thread-safe streaming that communicates via queue"""
        try:
            full_text = ""
            usage_info = {}
            elapsed_time = 0
            
            for chunk in self.client.stream(
                model_id,
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
                                "error": None
                            }
                        })
                    
                    if chunk.get("final"):
                        usage_info = chunk.get("usage", {})
                        elapsed_time = chunk.get("elapsed", 0)
                        # Send final update
                        result_queue.put({
                            result_key: {
                                "text": full_text,
                                "partial": False,
                                "error": None,
                                "usage": usage_info,
                                "elapsed": elapsed_time
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
        """Enhanced comparison with timeout, fallback, and better UX"""
        
        # Create queue for thread communication
        result_queue = queue.Queue()
        
        # Estimate response time based on model performance tier
        estimated_time_a = self._estimate_response_time(settings["model_a"])
        estimated_time_b = self._estimate_response_time(settings["model_b"])
        max_estimated_time = max(estimated_time_a, estimated_time_b)
        
        # Dynamic timeout based on performance mode
        if settings.get("performance_mode") == "Ultra Fast":
            timeout_per_model = 15
        elif settings.get("performance_mode") == "Balanced":
            timeout_per_model = 30
        else:
            timeout_per_model = 60
        
        # Start both model requests in threads
        with ThreadPoolExecutor(max_workers=2) as executor:
            # Submit both tasks
            future_a = executor.submit(
                self.stream_response_safe, 
                settings["model_a"], 
                prompt, 
                settings, 
                result_queue, 
                "a"
            )
            future_b = executor.submit(
                self.stream_response_safe, 
                settings["model_b"], 
                prompt, 
                settings, 
                result_queue, 
                "b"
            )
            
            # Track completion and results
            results = {"a": None, "b": None}
            completed = {"a": False, "b": False}
            start_time = time.time()
            
            # Enhanced progress tracking
            progress_container = st.container()
            with progress_container:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                time_text = st.empty()
            
            # Initial status
            placeholder_a.info("🚀 Starting Model A...")
            placeholder_b.info("🚀 Starting Model B...")
            
            # Process queue updates in main thread (UI-safe)
            timeout_occurred = False
            while not all(completed.values()) and not timeout_occurred:
                try:
                    # Check for timeout
                    elapsed = time.time() - start_time
                    if elapsed > timeout_per_model:
                        timeout_occurred = True
                        break
                    
                    # Get update from queue with timeout
                    update = result_queue.get(timeout=0.1)
                    
                    for key, data in update.items():
                        if data["error"]:
                            # Handle error with user-friendly message
                            error_msg = self._format_error_message(data["error"])
                            if key == "a":
                                placeholder_a.error(f"❌ Model A failed: {error_msg}")
                            else:
                                placeholder_b.error(f"❌ Model B failed: {error_msg}")
                            completed[key] = True
                            results[key] = None
                        
                        elif data["text"] is not None:
                            # Update UI with response text and streaming indicator
                            display_text = data["text"]
                            if data["partial"]:
                                display_text += " ▋"  # Cursor for streaming
                            
                            if key == "a":
                                placeholder_a.markdown(display_text)
                            else:
                                placeholder_b.markdown(display_text)
                            
                            # Check if this is the final update
                            if not data["partial"]:
                                completed[key] = True
                                results[key] = data["text"]
                                
                                # Show completion with stats
                                model_name = settings[f"model_{key}"]
                                completion_time = data.get("elapsed", elapsed)
                                if key == "a":
                                    placeholder_a.success(f"✅ Model A completed in {completion_time:.1f}s")
                                    placeholder_a.markdown(data["text"])
                                else:
                                    placeholder_b.success(f"✅ Model B completed in {completion_time:.1f}s")
                                    placeholder_b.markdown(data["text"])
                        
                        # Update progress and time
                        progress = sum(completed.values()) / len(completed)
                        progress_bar.progress(progress)
                        
                        elapsed = time.time() - start_time
                        if all(completed.values()):
                            status_text.success("🎉 Both models completed successfully!")
                            time_text.success(f"⏱️ Total time: {elapsed:.1f}s")
                        else:
                            active_models = [k.upper() for k, v in completed.items() if not v]
                            status_text.info(f"⏳ Running: Model {', '.join(active_models)}")
                            time_text.info(f"⏱️ Elapsed: {elapsed:.1f}s / ~{max_estimated_time:.0f}s estimated")
                
                except queue.Empty:
                    # No updates available, continue waiting
                    elapsed = time.time() - start_time
                    if elapsed > timeout_per_model:
                        timeout_occurred = True
                    continue
            
            # Handle timeout case
            if timeout_occurred:
                for key, completed_status in completed.items():
                    if not completed_status:
                        if key == "a":
                            placeholder_a.warning("⏰ Model A timed out - try a faster model")
                        else:
                            placeholder_b.warning("⏰ Model B timed out - try a faster model")
                        results[key] = None
            
            # Clean up progress indicators
            progress_container.empty()
            
            return results

    def _estimate_response_time(self, model_id: str) -> float:
        """Estimate response time based on model performance tier"""
        if model_id not in self.models_config:
            return 10.0  # Default estimate
        
        tier = getattr(self.models_config[model_id], 'performance_tier', 'fast')
        estimates = {
            'ultra_fast': 2.0,
            'fast': 4.0,
            'slow': 12.0,
            'very_slow': 25.0
        }
        return estimates.get(tier, 8.0)

    def _format_error_message(self, error: str) -> str:
        """Convert technical errors to user-friendly messages"""
        error_lower = error.lower()
        
        if "timeout" in error_lower:
            return "Response took too long"
        elif "rate limit" in error_lower:
            return "Too many requests, please wait"
        elif "connection" in error_lower:
            return "Connection issue, please retry"
        elif "empty response" in error_lower:
            return "Model returned empty response"
        else:
            return "Unexpected error occurred"

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
            
            # Run safe comparison
            results = self.run_comparison_safe(prompt, settings, placeholder_a, placeholder_b)
            
            # Proceed with judgment if both responses succeeded
            if results["a"] is not None and results["b"] is not None:
                st.markdown("### 👨‍⚖️ Judgment")
                with st.spinner("The judge is evaluating responses..."):
                    judgment = self.judge_service.judge_majority(
                        prompt,
                        results["a"],
                        results["b"],
                        settings["judge_model"],
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
                else:
                    st.error("Failed to get judgment: " + str(judgment.get("reason")))
            else:
                st.error("Cannot proceed with judgment due to model response errors.")

    def render(self):
        self.render_header()
        settings = self.render_sidebar()
        self.render_main(settings)