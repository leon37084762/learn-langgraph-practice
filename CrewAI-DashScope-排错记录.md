# CrewAI + DashScope（千问）排错记录

## 问题概述

在运行 `03_python_and_ai/23-crewai_ex.py`（CrewAI 调用阿里云 DashScope 千问 API）时，遇到 `401 Incorrect API key provided` 错误，经过多轮排查后解决。

---

## 错误现象

```
ERROR:root:OpenAI API call failed: Error code: 401 - {
  'error': {
    'message': 'Incorrect API key provided.',
    'type': 'invalid_request_error',
    'code': 'invalid_api_key'
  }
}
```

---

## 根本原因分析

本次错误由**多个叠加问题**导致，而非单一的 API key 错误：

### 1. `.env` 文件路径问题

`python-dotenv` 的 `load_dotenv()` **默认只在当前工作目录查找 `.env` 文件**。由于 `.env` 放在脚本子目录 `03_python_and_ai/` 下，而运行命令是在项目根目录执行的，导致环境变量根本没有被加载。

**修复**：显式指定 `.env` 路径：
```python
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent / ".env")
```

---

### 2. CrewAI 走 OpenAI 兼容接口，环境变量名不匹配

CrewAI 底层通过 LiteLLM 调用千问时，实际走的是 **OpenAI 兼容接口**（`https://dashscope.aliyuncs.com/compatible-mode/v1`），而非 DashScope 原生 API。

这意味着 LiteLLM 需要的是：
- `OPENAI_API_KEY`（而非 `DASHSCOPE_API_KEY`）
- `OPENAI_API_BASE`（需要显式设置为千问兼容接口地址）

**修复**：在代码中做环境变量映射：
```python
import os

os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

---

### 3. 环境变量设置时机太晚

CrewAI / LiteLLM 在 `import crewai` 时会读取并缓存 API 配置。如果在 `import` 之后才设置环境变量，配置不会生效。

**修复**：`load_dotenv` 和环境变量映射必须放在 `from crewai import ...` **之前**。

---

### 4. `llm` 参数前缀错误

`llm="dashscope/qwen-plus"` 会让 LiteLLM 走 **DashScope 原生 provider**，绕过 OpenAI 兼容接口，导致上述环境变量映射失效。

**修复**：改用 `openai/` 前缀，强制走 OpenAI 兼容接口：
```python
llm="openai/qwen-plus"
```

---

### 5. `litellm` 包未安装

CrewAI 对非原生 provider（如 `openai/qwen-plus` 这种自定义组合）的模型路由，依赖 **LiteLLM** 作为 fallback。如果未安装 `litellm`，会直接报错：

```
ImportError: Unable to initialize LLM with model 'openai/qwen-plus'.
The model did not match any supported native provider, and the LiteLLM fallback package is not installed.
```

**修复**：
```bash
pip install litellm
```

---

## 最终可用代码

```python
import os
from pathlib import Path
from dotenv import load_dotenv

# 必须在 import crewai 之前完成环境变量配置
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from crewai import Agent, Task, Crew, Process

researcher = Agent(
    role="行业研究员",
    goal="收集并总结2026年新能源电池技术进展",
    backstory="专注硬科技领域3年，擅长提炼技术核心点。",
    llm="openai/qwen-plus",  # 使用 openai/ 前缀走兼容接口
    verbose=True
)

writer = Agent(
    role="技术撰稿人",
    goal="将研究成果转化为通俗易懂的科普文章",
    backstory="前科技媒体主编，擅长将专业内容大众化。",
    llm="openai/qwen-plus",
    verbose=True
)

# ... Task 和 Crew 定义省略
```

---

## 经验教训

| 踩坑点 | 正确做法 |
|--------|----------|
| `load_dotenv()` 默认找不到子目录的 `.env` | 用 `Path(__file__).parent / ".env"` 显式指定 |
| 以为 `DASHSCOPE_API_KEY` 就够了 | CrewAI 走 OpenAI 兼容接口，需要映射到 `OPENAI_API_KEY` + `OPENAI_API_BASE` |
| `import crewai` 之后再设环境变量 | 必须在 `import` **之前**设置 |
| `llm="dashscope/qwen-plus"` | 改为 `llm="openai/qwen-plus"` 强制走兼容接口 |
| 没装 `litellm` | CrewAI 非原生模型必须 `pip install litellm` |

---

## 相关依赖版本

```
langchain          1.2.15
langchain-core     1.3.2
langgraph          1.1.9
crewai             1.14.3
litellm            1.83.14
```

---

*记录时间：2026-04-30*
