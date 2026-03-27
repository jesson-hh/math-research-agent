# Math Research Agent

一个基于 LLM 的数学研究智能体，集成了论文检索、符号计算、证明辅助、代码执行和自主研究等能力，帮助数学研究者高效地进行文献调研、公式推导、猜想验证和报告生成。

## 功能概览

### 核心工具

| 工具 | 说明 |
|------|------|
| **arxiv_search** | 检索 arXiv 数学论文，支持按领域/时间/相关度排序，5 分钟缓存 |
| **symbolic_compute** | 基于 SymPy 的精确符号计算（求导、积分、方程求解、级数展开等），30 秒超时 |
| **proof_assist** | 结构化证明辅助——分解定理、推荐证明策略、识别所需引理 |
| **run_code** | 沙箱化 Python 执行环境，支持 numpy/scipy/matplotlib，AST 白名单 + 线程超时 |
| **log_experiment** | 记录研究过程中的关键发现和实验结果（JSONL 存储） |
| **generate_report** | 从实验记录自动生成 Markdown/LaTeX 研究报告 |

### 五大功能模块

**1. Paper Discovery — 论文发现**
- arXiv 关键词检索，支持排序与筛选
- AI 自动筛选高价值论文
- 单篇论文深度分析与多轮问答
- 基于选定论文生成研究灵感

**2. Author Analysis — 学者分析**
- 按作者搜索论文
- AI 分析作者的证明技巧与方法论演变
- 分析结果 JSON 导出

**3. Network Graph — 学术关系网络**
- 合作者共现网络（基于 arXiv 论文列表构建）
- 引文关系网络（基于 Semantic Scholar API）
- Canvas 可视化，节点可点击查看详情
- 支持增量添加论文扩展网络

**4. LaTeX Editor — 笔记工坊**
- 基于论文和灵感自动生成 LaTeX 研究笔记
- 模板选择器（Research Note / Survey / Problem Set / Proof Sketch）
- BibTeX 自动生成
- Agent 对话式编辑（提议 → 接受/拒绝）
- KaTeX 实时预览，导出 .tex 文件

**5. Reports & Experiments — 报告与实验**
- AI 辅助实验方案设计
- 自动生成可执行 Python 实验代码
- 实验日志查看与统计
- Markdown/LaTeX 研究报告生成
- Jupyter Notebook 导出

**6. Auto Research — 自主研究模式**
- 给定研究目标后，Agent 自动规划、执行多轮研究循环
- 三阶段循环：PLANNING → EXECUTING → REPORTING
- 实时进度流式展示（阶段、任务列表、输出）
- 迭代次数/时间预算/API 调用上限可配置
- 失败任务自动反思重试

### 其他特性

- **深色模式** — 一键切换，localStorage 持久化
- **历史管理** — 灵感、论文池、网络图、笔记自动保存，支持重命名/删除/加载
- **上下文压缩** — 对话过长时智能截断历史（保留近 4 轮完整对话）
- **SSE 心跳** — 15 秒心跳保活，防止长操作断连
- **线程安全** — 对话请求隔离，自主研究模式加锁
- **安全下载** — 路径遍历防护，仅允许 `research_log/` 目录

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
networkx       — 图论计算 & 学术关系网络
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

<details>
<summary>高级配置（均可通过环境变量覆盖）</summary>

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_TIMEOUT` | LLM 请求超时（秒） | `120` |
| `SYMPY_TIMEOUT` | SymPy 计算超时（秒） | `30` |
| `CODE_TIMEOUT_MAX` | 代码沙箱超时（秒） | `30` |
| `MAX_TOKENS` | 主模型最大生成长度 | `8192` |
| `PROOF_MAX_TOKENS` | 证明模型最大生成长度 | `4096` |
| `CONTEXT_THRESHOLD` | 上下文压缩阈值（字符） | `80000` |
| `ARXIV_DELAY` | arXiv API 限速间隔（秒） | `3.0` |
| `AUTO_MAX_ITERATIONS` | 自主研究最大迭代次数 | `20` |
| `AUTO_MAX_TIME` | 自主研究最大时间（秒） | `600` |
| `AUTO_MAX_API_CALLS` | 自主研究最大 API 调用次数 | `50` |
| `SERVER_HOST` | 服务监听地址 | `127.0.0.1` |
| `SERVER_PORT` | 服务端口 | `7861` |

</details>

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

**OpenRouter**
```env
LLM_PROVIDER=openai
API_KEY=sk-or-xxx
BASE_URL=https://openrouter.ai/api/v1
MODEL=anthropic/claude-sonnet-4-6
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

服务默认运行在 `http://localhost:7861`，打开浏览器访问即可使用 Web 界面。

### API 端点

#### 对话与研究

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/` | Web 界面 |
| `POST` | `/api/chat` | 对话（SSE 流式响应） |
| `POST` | `/api/autonomous` | 自主研究模式（SSE 流式进度） |
| `POST` | `/api/stop-research` | 停止自主研究 |

#### 论文发现

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/papers` | arXiv 论文检索 |
| `POST` | `/api/auto-select` | AI 论文筛选 |
| `POST` | `/api/generate-ideas` | 基于论文生成研究灵感（SSE） |
| `POST` | `/api/analyze-paper` | 单篇论文深度分析（SSE） |
| `POST` | `/api/paper-qa` | 论文多轮问答（SSE） |

#### 学者分析 & 学术网络

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/author-papers` | 按作者搜索论文 |
| `POST` | `/api/analyze-techniques` | 分析作者证明技巧（SSE） |
| `POST` | `/api/coauthor-network` | 构建合作者网络 |
| `POST` | `/api/citation-network` | 构建引文网络（Semantic Scholar） |
| `POST` | `/api/add-papers-to-network` | 向已有网络增量添加论文 |

#### LaTeX 笔记

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/generate-note` | 生成 LaTeX 研究笔记（SSE） |
| `POST` | `/api/edit-note` | AI 编辑 LaTeX 笔记（SSE） |

#### 实验与报告

| 方法 | 路径 | 说明 |
|------|------|------|
| `POST` | `/api/design-experiment` | AI 设计实验方案（SSE） |
| `POST` | `/api/generate-code-plan` | 生成实验 Python 代码（SSE） |
| `GET` | `/api/experiments` | 查看实验记录 |
| `GET` | `/api/report?fmt=markdown` | 生成研究报告 |
| `GET` | `/api/notebook` | 导出 Jupyter Notebook |

#### 数据管理

| 方法 | 路径 | 说明 |
|------|------|------|
| `GET` | `/api/history/{type}` | 列出历史记录（ideas/pools/notes/networks） |
| `GET` | `/api/history/{type}/{id}` | 获取单条记录 |
| `POST` | `/api/history/{type}` | 保存记录 |
| `PATCH` | `/api/history/{type}/{id}` | 重命名记录 |
| `DELETE` | `/api/history/{type}/{id}` | 删除记录 |
| `GET` | `/api/download?path=...` | 下载生成的文件（限 research_log/） |

### 对话示例

```bash
curl -X POST http://localhost:7861/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "调研黎曼猜想最新进展", "domain": "number theory", "max_papers": 5}'
```

## 原理简介

### 架构

```
用户 ──> FastAPI Server ──> MathResearchAgent
              │                    │
              │           ┌────────┼────────┐
              │           │        │        │
              │      LLM Client  Tools  Tracking
              │      (OpenAI /  (6 tools) (ExperimentLog)
              │      Anthropic)
              │
         ┌────┴─────────────────────┐
         │  Paper Discovery APIs    │
         │  Author / Network APIs   │
         │  LaTeX Note APIs         │
         │  Experiment / Report     │
         │  History CRUD            │
         └──────────────────────────┘
```

### 核心工作流

1. **对话模式**：用户提出数学问题 → Agent 通过 LLM 规划使用哪些工具 → 调用工具获取结果 → LLM 综合分析并回答。支持多轮工具调用，LLM 会持续调用工具直到回答完成。

2. **自主研究模式**：给定研究目标后，Agent 进入 `PLANNING → EXECUTING → REPORTING` 循环：
   - **规划阶段**：LLM 分解研究目标为具体任务（TodoManager）
   - **执行阶段**：逐个执行任务，每步调用工具并记录实验；失败任务自动反思重试
   - **报告阶段**：汇总所有发现，生成结构化研究报告
   - 可设置迭代次数、时间预算和 API 调用上限

3. **论文发现流水线**：检索论文 → AI 筛选 → 深度分析/问答 → 生成灵感 → 生成 LaTeX 笔记 → 设计实验 → 生成代码

4. **上下文管理**：对话过长时自动压缩历史消息（`context/compressor.py`），保证不超出模型上下文窗口。保留最近 4 轮完整对话，智能截断旧的工具返回结果。

5. **代码沙箱**：`run_code` 通过 AST 白名单校验导入、线程超时控制，确保执行安全。

## 项目结构

```
├── server.py                  # FastAPI 服务入口（27 个 API 端点）
├── agent.py                   # 核心 Agent（对话 + 自主研究）
├── config.py                  # 集中配置（环境变量 + 默认值）
├── history_store.py           # 基于文件的 JSON 历史存储
├── log.py                     # 日志配置
├── llm/                       # LLM 客户端抽象层
│   ├── base.py                #   抽象基类
│   ├── openai_client.py       #   OpenAI 兼容实现
│   └── anthropic_client.py    #   Anthropic 原生实现
├── tools/                     # 6 个研究工具
│   ├── arxiv_tool.py          #   arXiv 检索（含作者搜索、缓存）
│   ├── sympy_tool.py          #   符号计算（30s 超时保护）
│   ├── proof_tool.py          #   证明辅助
│   ├── code_tool.py           #   代码执行沙箱（AST 白名单）
│   ├── log_tool.py            #   实验记录
│   └── report_tool.py         #   报告生成
├── autonomous/                # 自主研究循环
│   └── research_loop.py       #   PLAN → EXECUTE → REPORT 循环
├── context/                   # 对话管理 & 上下文压缩
│   ├── conversation.py        #   ConversationManager
│   └── compressor.py          #   智能消息压缩
├── planner/                   # 任务规划（TodoManager）
│   └── todo.py
├── tracking/                  # 实验日志追踪（JSONL）
│   └── experiment_log.py
├── reporting/                 # 报告 & Notebook 生成
│   ├── report_generator.py
│   └── notebook_generator.py
├── static/                    # Web 前端
│   ├── index.html             #   HTML 结构（侧边栏导航）
│   ├── app.js                 #   前端逻辑 & UI
│   └── style.css              #   样式（含深色模式主题）
├── research_log/              # 运行时输出目录
│   ├── experiments.jsonl      #   实验日志
│   └── history/               #   用户保存的历史数据
│       ├── ideas/
│       ├── pools/
│       ├── notes/
│       └── networks/
├── .env.example               # 环境变量模板
└── requirements.txt           # Python 依赖
```

## 技术栈

**后端**：FastAPI + uvicorn，httpx（HTTP 客户端），SSE 流式响应 + 心跳保活

**前端**：原生 JavaScript，marked.js（Markdown 渲染），KaTeX（LaTeX 数学渲染），Canvas API（网络图可视化），CSS Variables 主题系统

**LLM**：OpenAI 兼容协议（httpx）+ Anthropic 原生 SDK，支持流式输出

**科学计算**：SymPy, NumPy, SciPy, matplotlib, networkx, pandas, mpmath

## License

MIT
