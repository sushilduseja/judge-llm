# Judge LLM

A modern, production-grade application for comparing and evaluating Large Language Model outputs. Features an intuitive UI for real-time comparison of responses from different models, with automated judging capabilities.

## 🚀 Features

- **Modern UI/UX**:
  - Responsive Streamlit interface
  - Real-time streaming responses
  - Interactive model selection cards
  - Clear visualization of results
  - Progress indicators and tooltips

- **Supported Models**:
  - DeepSeek V3 0324
  - Qwen3 Coder
  - DeepSeek R1 0528
  - Google Gemini 2.0 Flash
  - DeepSeek R1 Distill Qwen 14B
  - NVIDIA Llama 3.1 Nemotron Ultra 253B
  - Reka Flash 3
  - Kimi K2
  - Gemma 3 4B
  - Qrwkv 72B

- **Key Capabilities**:
  - Real-time model comparison
  - Automated response judging
  - Performance metrics visualization
  - Result history tracking
  - Error handling and validation

## 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/judge-llm.git
   cd judge-llm
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows:
   .venv\Scripts\activate
   # On Unix or MacOS:
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenRouter API key
   ```

## 🚀 Usage

1. Start the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser at `http://localhost:8501`

3. Select models to compare:
   - Choose Model A and Model B from the sidebar
   - Configure judging parameters if needed
   - Enter your prompt
   - Click "Compare and Judge"

## 🏗️ Project Structure

```
judge-llm/
├── src/
│   ├── config/
│   │   └── models.py         # Configuration and model definitions
│   ├── services/
│   │   ├── openrouter.py    # API client
│   │   └── judge.py         # Judging logic
│   └── ui/
│       └── main.py          # Streamlit UI components
├── app.py                   # Main entry point
├── models.json             # Model definitions
├── .env.example           # Example environment variables
├── pyproject.toml        # Project metadata
└── requirements.txt      # Development dependencies
```

## ⚙️ Configuration

### Environment Variables

- `OPENROUTER_API_KEY`: Your OpenRouter API key (required)
- `MODEL_A`: Default Model A ID
- `MODEL_B`: Default Model B ID
- `JUDGE_MODEL`: Default judge model ID
- `JUDGE_REPEATS`: Number of judgment rounds
- `MAX_TOKENS`: Maximum output tokens
- `TEMPERATURE`: Model temperature
- `TOP_P`: Top-p sampling parameter
- `HTTP_TIMEOUT`: API timeout in seconds

### Model Configuration

Models are configured in `models.json` with the following properties:
- `id`: Model identifier
- `name`: Display name
- `description`: Detailed description
- `capabilities`: List of capabilities
- `limitations`: Known limitations
- `best_for`: Recommended use cases

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📧 Contact

For questions and support, please open an issue in the GitHub repository.
