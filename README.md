# Math Research Agent

一个基于 LLM 的数学研究智能体，集成了论文检索、符号计算、证明辅助、代码执行和自主研究等能力，帮助数学研究者高效地进行文献调研、公式推导、猜想验证和报告生成。

## 功能概览

| 工具 | 说明 |
|------|------|
| **arxiv_search** | 检索 arXiv 数学论文，支持按领域/时间/相关度排序 |
| **symbolic_compute** | 基于 SymPy 的精确符号计算（求导、积分、方程求解、级数展开等） |
| **proof_assist** | 结构化证明辅助——分解定理、推荐证明策略、识别所需引理 |
| **run_code** | 沙箱化 Python 执行环境，支持 numpy/scipy/matplotlib 等科学计算库 |
| **log_experiment** | 记录研究过程中的关键发现和实验结果 |
| **generate_report** | 从会话记录自动生成 Markdown/LaTeX 研究报告 |

此外还有：
- **自主研究模式** — 给定研究目标后，Agent 自动规划、执行多轮研究循环（文献→计算→验证→报告）
- **Paper Discovery** — 论文发现与 AI 辅助筛选，基于选定论文生成研究灵感
- **Jupyter Notebook 导出** — 将研究会话导出为可复现的 Notebook

## 安装

**环境要求：** Python 3.10+

```bash
# 克隆项目
git clone <repo-url>
cd "Math research Agent"

# 创建虚拟环境（推荐）
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 依赖说明

```
httpx          — HTTP 客户端（API 调用）
fastapi        — Web 服务框架
uvicorn        — ASGI 服务器
arxiv          — arXiv 论文检索
sympy          — 符号数学计算
numpy/scipy    — 数值计算
matplotlib     — 数据可视化
mpmath         — 高精度算术
networkx       — 图论计算
pandas         — 数据处理
nbformat       — Jupyter Notebook 生成
python-dotenv  — 环境变量管理
Pillow         — 图像处理
```

## API 配置

复制 `.env.example` 为 `.env`，填入你的模型配置：

```bash
cp .env.example .env
```

### 配置项

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_PROVIDER` | 后端类型：`openai`（兼容协议）或 `anthropic` | `openai` |
| `API_KEY` | API 密钥 | — |
| `BASE_URL` | API 基础地址 | — |
| `MODEL` | 主模型名称 | — |
| `PROOF_MODEL` | 证明辅助用的轻量模型（可选，默认同主模型） | — |

### 各平台配置示例

**阿里云百炼（千问系列，推荐）**
```env
LLM_PROVIDER=openai
API_KEY=sk-sp-xxx
BASE_URL=https://coding.dashscope.aliyuncs.com/v1
MODEL=qwen3.5-plus
PROOF_MODEL=qwen3-coder-flash
```

**Anthropic**
```env
LLM_PROVIDER=anthropic
API_KEY=sk-ant-xxx
MODEL=claude-sonnet-4-6
PROOF_MODEL=claude-haiku-4-5-20251001
```

**DeepSeek**
```env
LLM_PROVIDER=openai
API_KEY=sk-xxx
BASE_URL=https://api.deepseek.com/v1
MODEL=deepseek-chat
```

**本地模型（Ollama 等）**
```env
LLM_PROVIDER=openai
API_KEY=ollama
BASE_URL=http://localhost:11434/v1
MODEL=llama3
```

## 快速开始

```bash
# 启动服务
python server.py
```

服务默认运行在 `http://localhost:8000`，打开浏览器访问即可使用 Web 界面。

### API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web 界面 |
| `POST` | `/api/chat` | 对话（SSE 流式响应） |
| `POST` | `/api/papers` | arXiv 论文检索 |
| `POST` | `/api/auto-select` | AI 论文筛选 |
| `POST` | `/api/generate-ideas` | 基于论文生成研究灵感 |
| `GET` | `/api/experiments` | 查看实验记录 |
| `GET` | `/api/report?fmt=markdown` | 生成研究报告 |
| `GET` | `/api/notebook` | 导出 Jupyter Notebook |
| `POST` | `/api/stop-research` | 停止自主研究 |

### 对话示例

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "调研黎曼猜想最新进展", "domain": "number theory", "max_papers": 5}'
```

## 原理简介

### 架构

```
用户 ──> FastAPI Server ──> MathResearchAgent
                                  │
                         ┌────────┼────────┐
                         │        │        │
                    LLM Client  Tools  Tracking
                    (OpenAI /  (6 tools) (ExperimentLog)
                    Anthropic)
```

### 核心工作流

1. **对话模式**：用户提出数学问题 → Agent 通过 LLM 规划使用哪些工具 → 调用工具获取结果 → LLM 综合分析并回答。支持多轮工具调用，LLM 会持续调用工具直到回答完成。

2. **自主研究模式**：给定研究目标后，Agent 进入 `PLANNING → EXECUTING → REPORTING` 循环：
   - **规划阶段**：LLM 分解研究目标为具体任务（TodoManager）
   - **执行阶段**：逐个执行任务，每步调用工具并记录实验
   - **报告阶段**：汇总所有发现，生成结构化研究报告
   - 可设置迭代次数和时间预算上限

3. **上下文管理**：对话过长时自动压缩历史消息（`context/compressor.py`），保证不超出模型上下文窗口。

4. **代码沙箱**：`run_code` 通过 AST 白名单校验导入、线程超时控制，确保执行安全。

## 项目结构

```
├── server.py                  # FastAPI 服务入口
├── agent.py                   # 核心 Agent（对话 + 自主研究）
├── llm/                       # LLM 客户端抽象层
│   ├── base.py                #   抽象基类
│   ├── openai_client.py       #   OpenAI 兼容实现
│   └── anthropic_client.py    #   Anthropic 原生实现
├── tools/                     # 6 个研究工具
│   ├── arxiv_tool.py          #   arXiv 检索
│   ├── sympy_tool.py          #   符号计算
│   ├── proof_tool.py          #   证明辅助
│   ├── code_tool.py           #   代码执行沙箱
│   ├── log_tool.py            #   实验记录
│   └── report_tool.py         #   报告生成
├── autonomous/                # 自主研究循环
├── context/                   # 对话管理 & 上下文压缩
├── planner/                   # 任务规划（TodoManager）
├── tracking/                  # 实验日志追踪
├── reporting/                 # 报告 & Notebook 生成
├── static/                    # Web 前端（HTML/JS/CSS）
├── .env.example               # 环境变量模板
└── requirements.txt           # Python 依赖
```

## License

MIT
