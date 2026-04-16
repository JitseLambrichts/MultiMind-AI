<div align="center">
  <h1>🧠 MultiMind AI</h1>
  <p>
    <b>A local-first web UI that adds sequential reasoning pipelines and parallel expert councils on top of small local models.</b>
  </p>
  
  [![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
  [![PyPI Version](https://badge.fury.io/py/multimind.svg)](https://pypi.org/project/multimind/)
  [![Local First](https://img.shields.io/badge/Local-First-success.svg)](https://github.com/features)
  [![Ollama Supported](https://img.shields.io/badge/Ollama-Supported-purple.svg)](https://ollama.com/)
  [![LM Studio Supported](https://img.shields.io/badge/LM_Studio-Supported-ff69b4.svg)](https://lmstudio.ai/)
</div>

![Image](https://github.com/user-attachments/assets/c8578c27-80f7-474a-8bfa-1eca4437900a)

<br/>

MultiMind AI bridges the gap between small local models and complex reasoning. It effortlessly auto-discovers endpoints like Ollama and LM Studio (OpenAI-compatible) and lets you deploy three reasoning architectures: a sequential **Thinking Pipeline** (Planning, Execution, Critique), a parallel **Agent Council** (Expert Advisors & Lead Judge), or a hierarchical **Organisation Mode** (CEO → Departments → Employees → Synthesis).

---

## ✨ Features

- **🧠 Thinking Pipeline**: Elevate smaller models with dedicated **Planning**, **Execution**, and **Critique** phases.
- **🏛 Agent Council**: Deploy a committee of expert models in parallel. Several 'Advisors' provide independent perspectives, synthesized by a **Lead Judge** into a single superior response.
- **🏢 Organisation Mode**: Run a hierarchical multi-agent workflow where a **CEO** decomposes the request, **department heads** delegate work to specialist roles, and the **CEO** synthesizes all outputs into one final answer.
- **🔌 Zero-Config Auto-Discovery**:
  - Automatically hooks into local **Ollama** endpoints (`http://127.0.0.1:11434`).
  - Supports optional discovery for **LM Studio** (`http://127.0.0.1:1234`).
- **🎯 Precision Model Mapping**: Assign distinct models to handle the different stages of thought or council roles.
- **💬 Immersive UI**: Enjoy a streaming timeline interface with collapsible "thought blocks" to keep your UI clean while the AI thinks.
- **📝 Native Markdown & Math Support**: Final outputs are rendered as HTML, with math equations typeset using a bundled local **KaTeX** build.
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

## 🧠 Thinking Pipeline (Sequential Reasoning)

The sequential reasoning pipeline elevates the capabilities of standard models by splitting inference into modular steps:

1.  **Plan**: Formulates a detailed technical roadmap to solve the user's request.
2.  **Execute**: Implements the primary solution based on the established plan.
3.  **Critique (Hard Mode)**: Rigorously audits the implementation for errors or omissions, delivering a refined, superior final answer.

---

## 🏛 Agent Council (Parallel Expert Consensus)

For complex tasks requiring multiple perspectives, MultiMind AI offers the **Agent Council**. This architecture emphasizes parallel expertise over sequential steps:

1.  **Parallel Advisors**: Multiple models process the user's request independently, providing diverse expert viewpoints.
2.  **Diverse Perspectives**: Each advisor follows expert-level system prompts to ensure accurate, independent technical analysis.
3.  **The Judge**: A final 'Lead Synthesizer' model reviews all advisor outputs, cross-examines their findings, resolves conflicts, and merges the best elements into a single definitive response.

---

## 🏢 Organisation Mode (Hierarchical Multi-Agent Workflow)

For tasks that benefit from structured delegation, MultiMind AI includes **Organisation Mode**. Instead of parallel peers only, this mode simulates an org chart with explicit delegation layers:

1. **CEO Planning**: A CEO agent analyzes the user request and splits it into department-level sub-tasks.
2. **Department Delegation**: Each department head converts its sub-task into role-specific assignments.
3. **Employee Execution**: Specialist employee agents execute their assigned tasks and stream their outputs.
4. **CEO Synthesis**: The CEO consolidates all department/employee results into a single cohesive final response.

In the UI, this appears as an interactive organisation chart with expandable nodes and streaming outputs per role, while still ending with one final answer block.

> ⚙️ **Configuration:** Organisation mode uses one selected model for all agents in the hierarchy (configurable via the **Organisation** settings panel).

> 📝 **Note:** Chat history is intentionally in-memory only for the current MVP.

## 📊 Benchmarks

We evaluated the performance of MultiMind AI's reasoning pipeline using a subset of 20 questions from the GSM8K dataset. The results demonstrate a clear improvement in model accuracy when utilizing the different reasoning modes.
<br>

<div align="center">
    <img width=65% alt="Image" src="https://github.com/user-attachments/assets/23024d8a-d246-4020-a4e5-f02906ecfc0d" />
</div>
