# src/ui/main.py
from pathlib import Path
import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_extras.stylable_container import stylable_container
import json
import queue
import threading
import time
from typing import Dict, Any, List
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
        
        # Enhanced CSS
        st.markdown("""
            <style>
                .stApp { max-width: 1200px; margin: 0 auto; }
                .stButton > button { width: 100%; border-radius: 5px; height: 3em; }
                .model-card { 
                    border: 1px solid #ddd; border-radius: 8px; padding: 1rem; 
                    margin-bottom: 1rem; background: #f8f9fa;
                }
                .fallback-indicator {
                    background: linear-gradient(90deg, #fff3cd, #ffeaa7);
                    border-radius: 6px; padding: 0.75rem; margin: 0.5rem 0;
                    border-left: 4px solid #f39c12;
                }
                .winner-card { 
                    background: linear-gradient(135deg, #28a745, #20c997);
                    color: white; border-radius: 8px; padding: 1rem;
                }
            </style>
        """, unsafe_allow_html=True)

    def get_model_groups(self) -> Dict[str, List[str]]:
        """Group models by capability for better UX"""
        groups = {
            "Coding Specialists": [],
            "General Purpose": [],
            "Fast & Efficient": []
        }
        
        for model_id, model_config in self.models_config.items():
            if any(cap in ["code generation", "debugging", "code completion"] for cap in model_config.capabilities):
                groups["Coding Specialists"].append(model_id)
            elif "fast" in model_config.name.lower() or "flash" in model_config.name.lower():
                groups["Fast & Efficient"].append(model_id)
            else:
                groups["General Purpose"].append(model_id)
        
        return groups

    def render_model_selector(self, label: str, key: str, default_group: str = "Coding Specialists") -> str:
        """Enhanced model selector with grouping"""
        groups = self.get_model_groups()
        
        # Find a good default
        default_models = groups.get(default_group, list(self.models_config.keys()))
        default_model = default_models[0] if default_models else list(self.models_config.keys())[0]
        
        selected_model = st.selectbox(
            label,
            options=list(self.models_config.keys()),
            format_func=lambda x: self.models_config[x].name,
            key=key,
            index=list(self.models_config.keys()).index(default_model) if default_model in self.models_config else 0
        )
        
        # Show model details
        if selected_model:
            model = self.models_config[selected_model]
            with st.container():
                st.caption(model.description)
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**Best for:** {model.best_for}")
                    capabilities_str = ", ".join(model.capabilities[:3])  # Show first 3 capabilities
                    if len(model.capabilities) > 3:
                        capabilities_str += "..."
                    st.markdown(f"**Capabilities:** {capabilities_str}")
                with col2:
                    st.markdown(f"**Limitations:** {model.limitations}")
                    if model.together_fallback:
                        st.markdown(f"**Fallback:** Available")
        
        return selected_model

    def render_sidebar(self):
        with st.sidebar:
            st.header("⚙️ Settings")
            
            # Model Selection with improved UX
            st.subheader("Model Selection")
            
            # Ensure different models are selected
            if 'model_a' not in st.session_state:
                st.session_state.model_a = list(self.models_config.keys())[0]
            if 'model_b' not in st.session_state:
                models_list = list(self.models_config.keys())
                st.session_state.model_b = models_list[1] if len(models_list) > 1 else models_list[0]
            
            with stylable_container(key="model_a_container", css_styles="{ border: 2px solid #007bff; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }"):
                model_a = self.render_model_selector("🥊 Model A", "model_a", "Coding Specialists")
            
            with stylable_container(key="model_b_container", css_styles="{ border: 2px solid #28a745; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }"):
                model_b = self.render_model_selector("🥊 Model B", "model_b", "General Purpose")
            
            # Auto-fix if same model selected
            if model_a == model_b:
                available_models = [m for m in self.models_config.keys() if m != model_a]
                if available_models:
                    st.session_state.model_b = available_models[0]
                    st.rerun()
            
            # Judge Settings
            st.subheader("🧑‍⚖️ Judge Settings")
            judge_model = st.selectbox(
                "Judge Model",
                options=list(self.models_config.keys()),
                format_func=lambda x: self.models_config[x].name,
                help="Model used to evaluate and compare responses"
            )
            
            # Advanced Settings in expander
            with st.expander("🔧 Advanced Settings"):
                col1, col2 = st.columns(2)
                with col1:
                    judge_repeats = st.slider("Judge votes", 1, 5, self.config.judge_repeats)
                    max_tokens = st.slider("Max tokens", 64, 2048, self.config.max_tokens, step=64)
                with col2:
                    temperature = st.slider("Temperature", 0.0, 1.0, self.config.temperature, step=0.05)
                    http_timeout = st.slider("Timeout (s)", 30, 300, min(self.config.http_timeout, 300), step=15)
            
            return {
                "model_a": model_a,
                "model_b": model_b,
                "judge_model": judge_model,
                "judge_repeats": judge_repeats,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": self.config.top_p,
                "http_timeout": http_timeout
            }

    def render_results_with_metrics(self, results: Dict, fallback_info: Dict, settings: Dict):
        """Enhanced results display with performance metrics"""
        if results["a"] is None or results["b"] is None:
            return
            
        st.markdown("### 📊 Performance Metrics")
        
        # Create metrics columns
        col1, col2, col3, col4 = st.columns(4)
        
        # Model A metrics
        with col1:
            model_a_name = self.models_config[settings["model_a"]].name
            model_a_time_key = f'{settings["model_a"]}_time'
            model_a_time = st.session_state.get(model_a_time_key, 0)
            st.metric(
                "Model A Response Time", 
                f"{model_a_time:.1f}s",
                help=f"Time taken by {model_a_name}"
            )
        
        # Model B metrics  
        with col2:
            model_b_name = self.models_config[settings["model_b"]].name
            model_b_time_key = f'{settings["model_b"]}_time'
            model_b_time = st.session_state.get(model_b_time_key, 0)
            st.metric(
                "Model B Response Time",
                f"{model_b_time:.1f}s", 
                help=f"Time taken by {model_b_name}"
            )
        
        # Token usage - Fixed null-safe access
        with col3:
            model_a_usage_key = f'{settings["model_a"]}_usage'
            usage_a = st.session_state.get(model_a_usage_key, {}) or {}
            st.metric(
                "Model A Tokens",
                usage_a.get('total_tokens', 'N/A')
            )
            
        with col4:
            model_b_usage_key = f'{settings["model_b"]}_usage'
            usage_b = st.session_state.get(model_b_usage_key, {}) or {}
            st.metric(
                "Model B Tokens", 
                usage_b.get('total_tokens', 'N/A')
            )

    def render_main(self, settings: Dict[str, Any]):
        st.markdown("## 💭 Enter Your Prompt")
        
        # Prompt templates for quick start
        with st.expander("📝 Quick Start Templates"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("🐍 Python Function"):
                    st.session_state.prompt_text = "Write a Python function to calculate the factorial of a number using recursion."
            with col2:
                if st.button("🐛 Debug Code"):
                    st.session_state.prompt_text = "Debug this Python code:\n\ndef fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n\nprint(fibonacci(50))  # This is very slow"
            with col3:
                if st.button("📚 Explain Concept"):
                    st.session_state.prompt_text = "Explain how binary search works and provide a Python implementation."
        
        prompt = st.text_area(
            "Enter your prompt",
            height=200,
            placeholder="Ask both models to solve a problem, write code, or explain a concept...",
            value=st.session_state.get('prompt_text', ''),
            key='prompt_input'
        )
        
        # Update session state
        if prompt != st.session_state.get('prompt_text', ''):
            st.session_state.prompt_text = prompt
        
        # Show model comparison
        if settings["model_a"] != settings["model_b"]:
            col1, col2 = st.columns(2)
            with col1:
                st.info(f"🥊 **Model A:** {self.models_config[settings['model_a']].name}")
            with col2:
                st.info(f"🥊 **Model B:** {self.models_config[settings['model_b']].name}")
        
        if st.button("🚀 Compare Models", use_container_width=True, type="primary"):
            if not prompt.strip():
                st.warning("⚠️ Please enter a prompt first")
                return
            
            # Create response layout
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown(f"### 🤖 {self.models_config[settings['model_a']].name}")
                placeholder_a = st.empty()
                
            with col_right:
                st.markdown(f"### 🤖 {self.models_config[settings['model_b']].name}")
                placeholder_b = st.empty()
            
            # Run comparison
            results, fallback_info = self.run_comparison_safe(prompt, settings, placeholder_a, placeholder_b)
            
            # Show fallback info
            if any(info and info.get("used") for info in fallback_info.values()):
                st.warning("🔄 **Fallback Used:** Some models used Together AI due to OpenRouter issues.")
            
            # Show metrics
            self.render_results_with_metrics(results, fallback_info, settings)
            
            # Run judgment
            if results["a"] is not None and results["b"] is not None:
                st.markdown("---")
                st.markdown("### 👨‍⚖️ Judge's Decision")
                
                with st.spinner("🤔 Judge is analyzing responses..."):
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
                        st.info("🤝 **Result:** Both responses are equally good!")
                    else:
                        winner_model = settings["model_a"] if winner == "A" else settings["model_b"]
                        winner_name = self.models_config[winner_model].name
                        
                        st.markdown(f"""
                        <div class="winner-card">
                            <h3>🏆 Winner: {winner_name}</h3>
                            <p>The judge found this response superior in correctness, clarity, and usefulness.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Voting breakdown
                    if settings["judge_repeats"] > 1:
                        with st.expander("📊 Detailed Voting"):
                            votes = judgment["counts"]
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("Model A Votes", votes.get("A", 0))
                            with col2:
                                st.metric("Model B Votes", votes.get("B", 0))
                            with col3:
                                st.metric("Tie Votes", votes.get("TIE", 0))
                            
                            if votes.get("ERR", 0) > 0:
                                st.warning(f"⚠️ {votes['ERR']} judgment(s) failed")
                else:
                    st.error(f"❌ Judgment failed: {judgment.get('reason', 'Unknown error')}")

    def stream_response_safe(self, model_config: ModelCapability, prompt: str, settings: Dict[str, Any], result_queue: queue.Queue, result_key: str):
        """Thread-safe streaming with enhanced error handling"""
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
        """Enhanced comparison with better progress tracking"""
        
        model_config_a = self.models_config[settings["model_a"]]
        model_config_b = self.models_config[settings["model_b"]]
        
        result_queue = queue.Queue()
        
        with ThreadPoolExecutor(max_workers=2) as executor:
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
            
            results = {"a": None, "b": None}
            completed = {"a": False, "b": False}
            fallback_info = {"a": None, "b": None}
            
            # Enhanced progress tracking
            progress_container = st.container()
            with progress_container:
                progress_cols = st.columns(3)
                with progress_cols[0]:
                    progress_a = st.progress(0.0, text="Model A: Waiting...")
                with progress_cols[1]:
                    progress_b = st.progress(0.0, text="Model B: Waiting...")
                with progress_cols[2]:
                    overall_progress = st.progress(0.0, text="Overall: Starting...")
            
            start_time = time.time()
            
            while not all(completed.values()):
                try:
                    update = result_queue.get(timeout=0.1)
                    
                    for key, data in update.items():
                        model_name = self.models_config[settings[f"model_{key}"]].name
                        
                        if data["error"]:
                            if key == "a":
                                placeholder_a.error(f"❌ **Error:** {data['error']}")
                                progress_a.progress(1.0, text="Model A: Failed")
                            else:
                                placeholder_b.error(f"❌ **Error:** {data['error']}")
                                progress_b.progress(1.0, text="Model B: Failed")
                            completed[key] = True
                            results[key] = None
                        
                        elif data["text"] is not None:
                            # Track fallback usage
                            if data.get("fallback_used"):
                                fallback_info[key] = {
                                    "used": True,
                                    "model": data.get("fallback_model"),
                                    "original_model": settings[f"model_{key}"]
                                }
                            
                            # Prepare display text
                            display_text = data["text"]
                            if fallback_info[key] and fallback_info[key]["used"]:
                                fallback_msg = f"🔄 **Fallback:** {fallback_info[key]['model']}\n\n---\n\n"
                                display_text = fallback_msg + display_text
                            
                            # Update UI
                            if key == "a":
                                placeholder_a.markdown(display_text)
                                if not data["partial"]:
                                    progress_a.progress(1.0, text="Model A: Complete ✅")
                                else:
                                    progress_a.progress(0.8, text="Model A: Generating...")
                            else:
                                placeholder_b.markdown(display_text)
                                if not data["partial"]:
                                    progress_b.progress(1.0, text="Model B: Complete ✅")
                                else:
                                    progress_b.progress(0.8, text="Model B: Generating...")
                            
                            if not data["partial"]:
                                completed[key] = True
                                results[key] = data["text"]
                                
                                # Store metrics
                                model_key = settings[f'model_{key}']
                                if "usage" in data:
                                    st.session_state[f"{model_key}_usage"] = data["usage"]
                                if "elapsed" in data:
                                    st.session_state[f"{model_key}_time"] = data["elapsed"]
                        
                        # Update overall progress
                        progress = sum(completed.values()) / len(completed)
                        elapsed = time.time() - start_time
                        
                        if progress == 1.0:
                            overall_progress.progress(1.0, text=f"Complete! ({elapsed:.1f}s total)")
                        else:
                            remaining = [k.upper() for k, v in completed.items() if not v]
                            overall_progress.progress(progress, text=f"Running: {', '.join(remaining)} ({elapsed:.1f}s)")
                
                except queue.Empty:
                    continue
            
            # Clean up progress after delay
            time.sleep(1)
            progress_container.empty()
            
            return results, fallback_info

    def render_header(self):
        st.title("🤖 AI Model Arena")
        st.markdown("""
        **Compare AI models side-by-side** with automatic fallback support. 
        Get impartial judgments on response quality, performance, and usefulness.
        """)
        
        # Show API status
        col1, col2 = st.columns(2)
        with col1:
            if self.config.openrouter_api_key:
                st.success("✅ OpenRouter Connected")
            else:
                st.error("❌ OpenRouter Missing")
        with col2:
            if self.config.together_api_key:
                st.success("✅ Together AI Connected")
            else:
                st.warning("⚠️ Together AI Missing (Fallback Disabled)")

    def render(self):
        self.render_header()
        settings = self.render_sidebar()
        self.render_main(settings)