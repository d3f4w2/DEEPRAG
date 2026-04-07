<a id="top"></a>

<div align="center">

# [🔍 Deep RAG：让 AI 真正"读懂"知识库](%F0%9F%94%8D%20Deep%20RAG%EF%BC%9A%E8%AE%A9%20AI%20%E7%9C%9F%E6%AD%A3%22%E8%AF%BB%E6%87%82%22%E7%9F%A5%E8%AF%86%E5%BA%93.md)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node.js 16+](https://img.shields.io/badge/node-16+-green.svg)](https://nodejs.org/)

**[ [English](./README.md) | 中文 ]**

[特性](#-特性) • [快速开始](#-快速开始) • [工作原理](#-工作原理) • [架构](#-架构) • [配置](#-配置)

---

![Negation Exclusion.png](Knowledge-Base-File-Summary/Negation%20Exclusion.png)

<details>
<summary>📸 点击查看更多示例图片</summary>

<br>

![Finding Extremes.png](Knowledge-Base-File-Summary/Finding%20Extremes.png)
![Comparative Synthesis.png](Knowledge-Base-File-Summary/Comparative%20Synthesis.png)
![Multi-hop Reasoning.png](Knowledge-Base-File-Summary/Multi-hop%20Reasoning.png)
![Global Understanding.png](Knowledge-Base-File-Summary/Global%20Understanding.png)

</details>

</div>

---

## 🌟 为什么选择 Deep RAG？

### 传统 RAG 的问题

传统 RAG 系统：
- ❌ 将文档拆分成碎片，丢失结构
- ❌ 只能"查找相似"内容，无法"查找相反"
- ❌ 难以处理数值比较和否定逻辑
- ❌ 无法跨文档执行多跳推理
- ❌ 缺乏全局理解，难以进行聚合查询

### Deep RAG 的解决方案

```
传统 RAG: 分割文档 → 检索片段 → 喂给模型
Deep RAG:  保留结构 → 给模型"地图" → 让模型导航
```

Deep RAG 提供：
- ✅ **文件摘要作为知识地图**：LLM 看到整体结构
- ✅ **主动导航**：模型在需要时检索所需内容
- ✅ **多轮检索**：支持复杂的多跳推理
- ✅ **完整上下文**：检索完整文件/目录，而非碎片

> **想进一步了解 `Deep RAG`？欢迎阅读直观易懂的科普文章：[🔍 Deep RAG：让 AI 真正"读懂"知识库](%F0%9F%94%8D%20Deep%20RAG%EF%BC%9A%E8%AE%A9%20AI%20%E7%9C%9F%E6%AD%A3%22%E8%AF%BB%E6%87%82%22%E7%9F%A5%E8%AF%86%E5%BA%93.md)**

---

## 🎯 特性

### 核心能力

| 能力 | 传统 RAG | Deep RAG |
|-----------|----------------|----------|
| **否定查询**（"除了"、"之外"） | ❌ | ✅ |
| **数值比较**（"大于"、"小于"） | ❌ | ✅ |
| **查找极值**（"最大"、"最小"） | ❌ | ✅ |
| **跨文档比较** | ❌ | ✅ |
| **时序推理**（"去年"、"上一个"） | ❌ | ✅ |
| **多轮记忆** | ❌ | ✅ |
| **多跳推理** | ❌ | ✅ |
| **全局聚合** | ❌ | ✅ |

### 技术特性

- 🔌 **通用 LLM 支持**：OpenAI、Anthropic、Google Gemini 或任何 OpenAI 兼容 API
- 🛠️ **双工具调用模式**：Function Calling + ReAct
- 🎨 **现代化 Web UI**：基于 React + TypeScript + Vite 构建
- ⚡ **流式响应**：实时响应流式传输
- 🔧 **便捷配置**：基于 Web 的 .env 编辑器
- 📊 **工具调用可视化**：查看 AI 的操作过程

### 实习项目增强版（保留原特色）

本仓库在原始 Deep RAG 思路上，补全了可用于实习/面试展示的工程闭环：
- ✅ 两阶段检索：`search_paths -> retrieve_sections`
- ✅ 证据放行与 Critic：答案必须带 `### 证据`
- ✅ 自动补检索与预算保护：控制 token/时延上限
- ✅ 多模态入库：语音、图片入库后可立即进入统一检索链路
- ✅ 评测闭环：RAGAS + token + 延迟

### 架构图（当前实现）

```text
┌────────────────────────────────────────────────────────────────────────────┐
│ Frontend (React + TS)                                                     │
│ Chat / Evaluation / Voice Ingestion / Image Ingestion                     │
└───────────────────────────────┬────────────────────────────────────────────┘
                                │ HTTP + SSE
┌───────────────────────────────▼────────────────────────────────────────────┐
│ Backend (FastAPI)                                                         │
│ Chat Orchestrator + ReAct + Critic + Budget Guard                         │
│ Tools: search_paths / retrieve_sections                                    │
└───────────────┬───────────────────────────────┬────────────────────────────┘
                │                               │
┌───────────────▼───────────────────┐   ┌───────▼───────────────────────────┐
│ Knowledge Base (.md / image notes)│   │ LLM Provider (OpenAI-compatible)   │
│ + summary_demo.json path index    │   │ + RAGAS evaluator                  │
└───────────────────────────────────┘   └────────────────────────────────────┘
```

### 核心流程图

```text
用户提问
  -> 查询扩展 + 路由规划
  -> search_paths(候选路径)
  -> retrieve_sections(证据片段)
  -> evidence_pool 合并
  -> critic 判定(accept/revise/refuse)
  -> [不足] 自动补检索
  -> [通过] 输出答案 + ### 证据
```

```text
语音/图片上传
  -> draft(机器草稿)
  -> 人工确认
  -> ingest(写入 Knowledge-Base/*-Notes/*.md)
  -> update_summary_entry(summary_demo.json)
  -> 下一轮 search_paths 可直接命中
```

### 关键指标表（简历可引用）

| 维度 | 基线/优化前 | 当前优化版 | 说明 |
|---|---:|---:|---|
| 单题检索上下文体量 | 20,799 chars | 8,343 chars | `-59.9%` |
| 单题工具链路 | `retrieve_files ×3` | `search_paths -> retrieve_sections` | 两阶段检索 |
| 40题平均 token（历史 baseline） | 6,177.15 | 14,395.48 | 评测集口径：`ragas评测集_40题` |
| 40题平均延迟（历史 baseline） | 13,721.10 ms | 39,080.38 ms | 需要继续做成本优化 |
| 40题忠诚度（RAGAS） | - | 39.86% | 见 `评测/结果/optimized_ragas_40q_汇总.json` |

> 说明：当前 provider 与 RAGAS 某些指标存在输出解析兼容性问题，`context_recall` 与 `answer_correctness` 仍需进一步稳定化（见评测报告中的口径说明）。

实习展示材料：
- `优化/40题_RAGAS_优化后_vs_baseline.md`
- `优化/1分钟Demo脚本.md`
- `优化/亲自解决问题_简历表述.md`

---

## 🚀 快速开始

### 前置要求

- **Python 3.8+**
- **Node.js 16+**
- **LLM API 密钥**（OpenAI、Google Gemini、Anthropic 或兼容服务）

### 安装

1. **克隆仓库**
```bash
git clone https://github.com/boluo2077/deep-rag.git
cd deep-rag
```

2. **配置环境变量**
```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 API 密钥
```

3. **启动应用**
```bash
./start.sh
```

**快速启动（跳过依赖检查）：**
```bash
./start.sh --fast
```

脚本将自动：
- ✅ 创建 Python 虚拟环境
- ✅ 安装后端依赖
- ✅ 安装前端依赖  
- ✅ 启动后端服务器 (http://localhost:8000)
- ✅ 启动前端开发服务器 (http://localhost:5173)
- ✅ 自动打开浏览器

### 手动启动（备选）

**后端：**
```bash
python3 -m venv venv
source venv/bin/activate  # Windows 系统：venv\Scripts\activate
pip install -r requirements.txt
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

**前端：**
```bash
cd frontend
npm install
npm run dev
```

### 停止/重启

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

## 💡 工作原理

### 1. 知识库结构

Deep RAG 不会将文档拆分成碎片，而是保留你的文件结构：

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

### 2. 文件摘要生成

生成知识库的结构化摘要：

```bash
cd Knowledge-Base-File-Summary
python generate.py
```

这将创建一个"知识地图"，如下所示：
```
Product-Line-A-Smartwatch-Series/
├─ SW-2100-Flagship.md: 2.1" AMOLED, 72h battery, IP68, $2999
├─ SW-1800-Business.md: 1.8" LCD, 48h battery, IP67, $1899
└─ SW-1500-Sport.md: 1.5" TFT, 36h battery, IP68, $999
```

### 3. 系统提示词集成

文件摘要被注入到系统提示词中，为 LLM 提供：
- 📍 所有可用知识的概览
- 🗺️ 用于精准检索的文件路径
- 🎯 规划多步骤查询的能力

### 4. 主动检索

在回答问题时，LLM 可以：
```python
retrieve_files([
    "Product-Line-A-Smartwatch-Series/SW-2100-Flagship.md",  # 特定文件
    "2023-Market-Layout/",                                     # 整个目录
    "/"                                                        # 所有文件
])
```

---

## 🏗️ 架构

```
┌─────────────────────────────────────────────────────────────┐
│                       前端 (React)                           │
│  • 聊天界面  • 配置面板  • 系统提示词查看器                    │
└───────────────────────────┬─────────────────────────────────┘
                            │ HTTP/SSE
┌───────────────────────────┴─────────────────────────────────┐
│                    后端 (FastAPI)                            │
│  • LLM 提供者抽象  • 工具调用处理器                          │
│  • 知识库管理器    • ReAct 模式支持                          │
└───────────────────────────┬─────────────────────────────────┘
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
   ┌────┴────┐         ┌───┴───┐          ┌───┴───┐
   │ 知识库  │         │  LLM  │          │ 工具  │
   │  文件   │         │  API  │          │(Func/ │
   │         │         │       │          │ReAct) │
   └─────────┘         └───────┘          └───────┘
```

### 项目结构

```
deep-rag/
├── backend/                   # FastAPI 后端
│   ├── main.py               # API 端点
│   ├── config.py             # 配置管理
│   ├── llm_provider.py       # LLM 提供者抽象
│   ├── knowledge_base.py     # 知识库操作
│   ├── prompts.py            # 系统提示词和工具
│   ├── react_handler.py      # ReAct 模式处理器
│   └── models.py             # Pydantic 模型
├── frontend/                  # React 前端
│   ├── src/
│   │   ├── App.tsx           # 主应用组件
│   │   ├── components/       # React 组件
│   │   └── api.ts            # API 客户端
│   └── package.json
├── Knowledge-Base/            # 你的文档
├── Knowledge-Base-Chunks/     # 分块文档（可选）
├── Knowledge-Base-File-Summary/
│   ├── generate.py           # 摘要生成器
│   └── summary.txt           # 生成的摘要
├── .env.example              # 环境配置模板
├── requirements.txt          # Python 依赖
├── start.sh                  # 启动脚本
├── stop.sh                   # 停止脚本
└── restart.sh                # 重启脚本
```

---

## ⚙️ 配置

### 环境变量

编辑 `.env` 文件进行配置：

```bash
# LLM 提供者（openai、google、anthropic、custom）
API_PROVIDER=google

# 工具调用模式（function、react）
TOOL_CALLING_MODE=function

# 模型参数
TEMPERATURE=0
MAX_TOKENS=8192

# 知识库路径
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

# 自定义提供者（任何 OpenAI 兼容 API）
CUSTOM_API_KEY=your_api_key
CUSTOM_BASE_URL=https://your-api.com/v1/chat/completions
CUSTOM_MODEL=your-model
```

### 添加新的 LLM 提供者

无需修改代码！只需在 `.env` 中添加：

```bash
PROVIDER_NAME_API_KEY=your_key
PROVIDER_NAME_BASE_URL=https://api.provider.com/v1
PROVIDER_NAME_MODEL=model-name

API_PROVIDER=provider_name
```

### 工具调用模式

**Function Calling 模式**（推荐）
- 适用于原生支持函数调用的模型
- 示例：GPT-4+、Gemini 1.5+、Claude 3.5+
- 更可靠、更结构化

**ReAct 模式**
- 适用于不支持函数调用的模型
- 使用基于提示词的推理和操作
- 兼容任何文本补全模型

---

<div align="center">

**如果这个项目对你有帮助，请点击右上角的 ⭐ Star！**

*你的 Star 是我们持续改进的动力 💪*

![Star History Chart](https://api.star-history.com/svg?repos=boluo2077/deep-rag&type=Date)

---

<a href="#top">
  <img src="https://img.shields.io/badge/⬆️-回到顶部-blue?style=for-the-badge" alt="回到顶部">
</a>

</div>
