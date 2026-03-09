<div align="center">
  <h1>🧠 MultiMind AI</h1>
  <p>
    <b>A local-first web UI that adds a reasoning pipeline on top of small local models.</b>
  </p>
  
  [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
  [![PyPI Version](https://badge.fury.io/py/multimind.svg)](https://pypi.org/project/multimind/)
  [![Local First](https://img.shields.io/badge/Local-First-success.svg)](https://github.com/features)
  [![Ollama Supported](https://img.shields.io/badge/Ollama-Supported-purple.svg)](https://ollama.com/)
  [![LM Studio Supported](https://img.shields.io/badge/LM_Studio-Supported-ff69b4.svg)](https://lmstudio.ai/)
</div>

![Image](https://github.com/user-attachments/assets/7b745a2f-6227-46fc-b4f0-47b2e58a0964)

<br/>

MultiMind AI acts as an intelligent reasoning pipeline for your local AI models. It effortlessly auto-discovers endpoints like Ollama and LM Studio (OpenAI-compatible) and lets you orchestrate dedicated models for different logical phases: **Planning**, **Execution**, and **Critique**.

---

## ✨ Features

- **🧠 Adaptive Reasoning Modes**: Toggle between _Off_, _Medium_, and _Hard_ modes to dictate the depth of the model's reflection.
- **🔌 Zero-Config Auto-Discovery**:
  - Automatically hooks into local **Ollama** endpoints (`http://127.0.0.1:11434`).
  - Supports optional discovery for **LM Studio** (`http://127.0.0.1:1234`).
- **🎯 Precision Model Mapping**: Assign distinct models to handle the different stages of thought (`plan`, `execute`, and `critique`).
- **💬 Immersive UI**: Enjoy a streaming timeline interface with collapsible "thought blocks" to keep your UI clean while the AI thinks.
- **📝 Native Markdown & Math Support**:
  - Final outputs are beautifully rendered as HTML in the chat view.
  - Inline and block math equations are flawlessly typeset using a bundled local **KaTeX** build.
- **⚡ Frictionless Setup**: Purely in-memory settings. Zero `.env` setup required for your first run.

## 🚀 Quick Start

Get up and running in your local environment in seconds:

```bash
# 1. Install the package via pip
pip install multimind

# 2. Launch the application
multimind
```

<details>
<summary><b>🛠 Setting up for Development / Source Install</b></summary>

```bash
# 1. Clone the repository
git clone https://github.com/JitseLambrichts/MultiMind-AI.git
cd MultiMind-AI

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the virtual environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 4. Install the package in editable mode
pip install -e .

# 5. Launch the application
multimind
```

</details>

> **Next:** Open your browser and navigate to [http://127.0.0.1:8000](http://127.0.0.1:8000) 🎉

## 🔌 Supported Backends

MultiMind AI works seamlessly with standard local APIs:

- **Ollama**: Connects via `/api/chat` and `/api/tags`
- **OpenAI-Compatible Servers (e.g., LM Studio)**: Connects via `/v1/chat/completions` and `/v1/models`

_If no provider is automatically detected, you can easily point the backend to your local OpenAI-compatible endpoint using the application's settings panel._

## 💡 How It Works

MultiMind AI splits inference into modular steps, elevating the capabilities of standard models:

1. **Plan**: Formulates a structured approach to the prompt.
2. **Execute**: Generates the primary response.
3. **Critique (Hard Mode)**: Evaluates the execution pass as a rough draft and streams refined, critiqued output as the final answer.

> 📝 **Note:** Chat history is intentionally in-memory only for the current MVP.

## 📊 Benchmarks

We evaluated the performance of MultiMind AI's reasoning pipeline using a subset of 20 questions from the GSM8K dataset. The results demonstrate a clear improvement in model accuracy when utilizing the different reasoning modes.
<br>
<div align="center">
    <img width=65% alt="Image" src="https://github.com/user-attachments/assets/3df2b12f-97c1-4cb2-8b7d-b49a0c5c379c" />
</div>
