<a id="top"></a>

<div align="center">

# [🔍 Deep RAG: Teaching AI to Truly "Understand" Your Knowledge Base](%F0%9F%94%8D%20Deep%20RAG%3A%20Teaching%20AI%20to%20Truly%20%22Understand%22%20Your%20Knowledge%20Base.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 16+](https://img.shields.io/badge/node-16+-green.svg)](https://nodejs.org/)

**[ English | [中文](./README.zh-CN.md) ]**

[Features](#-features) • [Quick Start](#-quick-start) • [How It Works](#-how-it-works) • [Architecture](#-architecture) • [Configuration](#-configuration)

---

![Negation Exclusion.png](Knowledge-Base-File-Summary/Negation%20Exclusion.png)

<details>
<summary>📸 Click to see more example images</summary>

<br>

![Finding Extremes.png](Knowledge-Base-File-Summary/Finding%20Extremes.png)
![Comparative Synthesis.png](Knowledge-Base-File-Summary/Comparative%20Synthesis.png)
![Multi-hop Reasoning.png](Knowledge-Base-File-Summary/Multi-hop%20Reasoning.png)
![Global Understanding.png](Knowledge-Base-File-Summary/Global%20Understanding.png)

</details>

</div>

---

## 🌟 Why Deep RAG?

### The Problem with Traditional RAG

Traditional RAG systems:
- ❌ Split documents into fragments, losing structure
- ❌ Can only "find similar" content, not "find opposite" 
- ❌ Struggle with numerical comparisons and negation logic
- ❌ Cannot perform multi-hop reasoning across documents
- ❌ Lack global understanding for aggregation queries

### The Deep RAG Solution

```
Traditional RAG: Fragment documents → Retrieve chunks → Feed to model
Deep RAG:        Preserve structure → Give model a "map" → Let model navigate
```

Deep RAG provides:
- ✅ **File Summary as Knowledge Map**: LLM sees the entire structure
- ✅ **Active Navigation**: Model retrieves what it needs, when it needs it
- ✅ **Multi-round Retrieval**: Supports complex multi-hop reasoning
- ✅ **Complete Context**: Retrieves full files/directories, not fragments

> **Want to learn more about `Deep RAG`? Welcome to read the intuitive and easy-to-understand article: [🔍 Deep RAG: Teaching AI to Truly "Understand" Your Knowledge Base](%F0%9F%94%8D%20Deep%20RAG%3A%20Teaching%20AI%20to%20Truly%20%22Understand%22%20Your%20Knowledge%20Base.md)**

---

## 🎯 Features

### Core Capabilities

| Capability | Traditional RAG | Deep RAG |
|-----------|----------------|----------|
| **Negation Queries** ("except", "besides") | ❌ | ✅ |
| **Numerical Comparison** ("greater than", "less than") | ❌ | ✅ |
| **Finding Extremes** ("maximum", "minimum") | ❌ | ✅ |
| **Cross-document Comparison** | ❌ | ✅ |
| **Temporal Reasoning** ("last year", "previous") | ❌ | ✅ |
| **Multi-turn Memory** | ❌ | ✅ |
| **Multi-hop Reasoning** | ❌ | ✅ |
| **Global Aggregation** | ❌ | ✅ |

### Technical Features

- 🔌 **Universal LLM Support**: OpenAI, Anthropic, Google Gemini, or any OpenAI-compatible API
- 🛠️ **Dual Tool Calling Modes**: Function Calling + ReAct
- 🎨 **Modern Web UI**: Built with React + TypeScript + Vite
- ⚡ **Streaming Responses**: Real-time response streaming
- 🔧 **Easy Configuration**: Web-based .env editor
- 📊 **Tool Call Visualization**: See what the AI is doing

### Internship-Ready Enhancements (while keeping original Deep RAG strengths)

This repo now includes a practical engineering loop suitable for resume/interview demos:
- ✅ Two-stage retrieval: `search_paths -> retrieve_sections`
- ✅ Evidence gate + critic: final answer must include `### 证据`
- ✅ Auto-retrieval retry + budget guard (token/latency constraints)
- ✅ Multimodal ingestion: voice/image notes become searchable immediately
- ✅ Evaluation loop: RAGAS + token + latency

### Current Architecture

```text
Frontend (React + TS)
  Chat / Evaluation / Voice Ingestion / Image Ingestion
            |
          HTTP + SSE
            v
Backend (FastAPI)
  Chat Orchestrator + ReAct + Critic + Budget Guard
  Tools: search_paths / retrieve_sections
            |
   +--------+-------------------+
   |                            |
Knowledge Base + summary index   LLM Provider + RAGAS evaluator
```

### Core Flow

```text
User question
  -> query expansion + route planning
  -> search_paths (candidate files)
  -> retrieve_sections (evidence snippets)
  -> merge evidence_pool
  -> critic decision (accept/revise/refuse)
  -> [if weak] auto retrieval retry
  -> [if pass] final answer + ### 证据
```

### Key Metrics (resume-friendly)

| Dimension | Baseline/Before | Current | Notes |
|---|---:|---:|---|
| Single-question retrieval context volume | 20,799 chars | 8,343 chars | `-59.9%` |
| Single-question tool chain | `retrieve_files ×3` | `search_paths -> retrieve_sections` | Two-stage retrieval |
| 40Q completion (latest full run) | - | `40 / 40`, failed `0` | `评测/结果/optimized_ragas_40q_v2_汇总.json` |
| 40Q avg token (latest full run) | 6,177.15 | 16,293.77 | higher than baseline, still needs cost optimization |
| 40Q avg latency (latest full run) | 13,721.10 ms | 34,562.76 ms | latest run on `2026-04-07` |
| RAGAS usable questions (latest provider config) | - | `0 / 40` | parser incompatibility under current judge/provider |
| Historical measurable faithfulness (older compatible run) | - | 39.86% | `评测/结果/optimized_ragas_40q_汇总.json` |

> Note: the latest end-to-end run is fully traceable for completion, token, and latency, but the current provider/judge combination still causes RAGAS parser incompatibility. Treat `39.86% faithfulness` as a historical measurable run, not the latest all-green benchmark.

Interview-ready assets:
- `优化/40题_RAGAS_优化后_vs_baseline.md`
- `优化/1分钟Demo脚本.md`
- `优化/亲自解决问题_简历表述.md`

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **Node.js 16+**
- **An LLM API Key** (OpenAI, Google Gemini, Anthropic, or compatible)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/boluo2077/deep-rag.git
cd deep-rag
```

2. **Configure environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys
```

3. **Start the application**
```bash
./start.sh
```

**Quick start (skip dependency checks):**
```bash
./start.sh --fast
```

The script will:
- ✅ Create Python virtual environment
- ✅ Install backend dependencies
- ✅ Install frontend dependencies  
- ✅ Start backend server (http://localhost:8000)
- ✅ Start frontend dev server (http://localhost:5173)
- ✅ Open browser automatically

### Manual Start (Alternative)

**Backend:**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### Stop/Restart

```bash
./stop.sh
```

```bash
./restart.sh
```

```bash
./restart.sh --full
```

---

## 💡 How It Works

### 1. Knowledge Base Structure

Instead of splitting documents into chunks, Deep RAG preserves your file structure:

```
Knowledge-Base/
├─ Product-Line-A-Smartwatch-Series/
│  ├─ SW-2100-Flagship.md
│  ├─ SW-1800-Business.md
│  └─ SW-1500-Sport.md
├─ 2023-Market-Layout/
│  ├─ East-China-Region.md
│  └─ South-China-Region.md
└─ Supplier-Partnership-Records/
   └─ Display-Supplier-CrystalVision.md
```

### 2. File Summary Generation

Generate a structured summary of your knowledge base:

```bash
cd Knowledge-Base-File-Summary
python generate.py
```

This creates a "knowledge map" that looks like:
```
Product-Line-A-Smartwatch-Series/
├─ SW-2100-Flagship.md: 2.1" AMOLED, 72h battery, IP68, $2999
├─ SW-1800-Business.md: 1.8" LCD, 48h battery, IP67, $1899
└─ SW-1500-Sport.md: 1.5" TFT, 36h battery, IP68, $999
```

### 3. System Prompt Integration

The file summary is injected into the system prompt, giving the LLM:
- 📍 Overview of all available knowledge
- 🗺️ File paths for targeted retrieval
- 🎯 Ability to plan multi-step queries

### 4. Active Retrieval

When answering questions, the LLM can:
```python
retrieve_files([
    "Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md",  # Specific file
    "2023-Market-Layout/",                                     # Entire directory
    "/"                                                        # All files
])
```

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Frontend (React)                       │
│  • Chat Interface  • Config Panel  • System Prompt Viewer   │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/SSE
┌───────────────────────────┴─────────────────────────────────┐
│                    Backend (FastAPI)                         │
│  • LLM Provider Abstraction  • Tool Calling Handler         │
│  • Knowledge Base Manager    • ReAct Mode Support           │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────┴────┐         ┌───┴───┐          ┌───┴───┐
   │Knowledge│         │  LLM  │          │ Tools │
   │  Base   │         │  API  │          │(Func/ │
   │  Files  │         │       │          │ReAct) │
   └─────────┘         └───────┘          └───────┘
```

### Project Structure

```
deep-rag/
├── backend/                   # FastAPI backend
│   ├── main.py               # API endpoints
│   ├── config.py             # Configuration management
│   ├── llm_provider.py       # LLM provider abstraction
│   ├── knowledge_base.py     # Knowledge base operations
│   ├── prompts.py            # System prompts & tools
│   ├── react_handler.py      # ReAct mode handler
│   └── models.py             # Pydantic models
├── frontend/                  # React frontend
│   ├── src/
│   │   ├── App.tsx           # Main app component
│   │   ├── components/       # React components
│   │   └── api.ts            # API client
│   └── package.json
├── Knowledge-Base/            # Your documents
├── Knowledge-Base-Chunks/     # Chunked documents (optional)
├── Knowledge-Base-File-Summary/
│   ├── generate.py           # Summary generator
│   └── summary.txt           # Generated summary
├── .env.example              # Environment config template
├── requirements.txt          # Python dependencies
├── start.sh                  # Start script
├── stop.sh                   # Stop script
└── restart.sh                # Restart script
```

---

## ⚙️ Configuration

### Environment Variables

Edit `.env` to configure:

```bash
# LLM Provider (openai, google, anthropic, custom)
API_PROVIDER=google

# Tool Calling Mode (function, react)
TOOL_CALLING_MODE=function

# Model Parameters
TEMPERATURE=0
MAX_TOKENS=8192

# Knowledge Base Paths
KNOWLEDGE_BASE_PATH=./Knowledge-Base
KNOWLEDGE_BASE_FILE_SUMMARY=./Knowledge-Base-File-Summary/summary.txt

# Google Gemini
GOOGLE_API_KEY=your_google_key
GOOGLE_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
GOOGLE_MODEL=gemini-2.5-flash-lite

# OpenAI
OPENAI_API_KEY=your_openai_key
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4.1-mini

# Anthropic Claude
ANTHROPIC_API_KEY=your_anthropic_key
ANTHROPIC_BASE_URL=https://api.anthropic.com/v1
ANTHROPIC_MODEL=claude-3-5-sonnet-20241022

# Custom Provider (any OpenAI-compatible API)
CUSTOM_API_KEY=your_api_key
CUSTOM_BASE_URL=https://your-api.com/v1/chat/completions
CUSTOM_MODEL=your-model
```

### Adding New LLM Providers

No code changes needed! Just add to `.env`:

```bash
PROVIDER_NAME_API_KEY=your_key
PROVIDER_NAME_BASE_URL=https://api.provider.com/v1
PROVIDER_NAME_MODEL=model-name

API_PROVIDER=provider_name
```

### Tool Calling Modes

**Function Calling Mode** (Recommended)
- For models with native function calling support
- Examples: GPT-4+, Gemini 1.5+, Claude 3.5+
- More reliable and structured

**ReAct Mode**
- For models without function calling
- Uses prompt-based reasoning and action
- Compatible with any text-completion model

---

<div align="center">

**If this project helps you, please click the ⭐ Star in the top right corner!**

*Your Star is our motivation to keep improving 💪*

![Star History Chart](https://api.star-history.com/svg?repos=boluo2077/deep-rag&type=Date)

---

<a href="#top">
  <img src="https://img.shields.io/badge/⬆️-Back_to_Top-blue?style=for-the-badge" alt="Back to Top">
</a>

</div>
