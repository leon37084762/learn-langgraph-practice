# 智能体搭建 & LangGraph 飞速上手 — 练习与实现

本项目是 [LearnGraph.online](https://www.learngraph.online/LearnGraph%201.X/README.html) 课程《智能体搭建 & LangGraph 飞速上手》的个人练习与代码实现仓库。

## 📚 课程简介

本书/课程是一部系统学习 **LangGraph** 与 **Multi-Agent** 系统开发的实战教程。从 Python 基础到生产部署，通过 12 大模块、80+ 篇详细解读，帮助学习者掌握智能体构建的完整技能。

## 🎯 学习目标

- 理解 LangGraph 的核心概念和设计理念
- 掌握状态管理、节点定义、条件边等关键技术
- 学会构建人机协作的智能工作流
- 实现复杂的 Multi-Agent 协作系统
- 掌握记忆系统和持久化技术
- 了解生产环境部署的最佳实践
- 通过完整案例掌握实战技能

## 📖 章节目录

| 模块 | 主题 | 内容 |
|------|------|------|
| 0 | Python 基础 | Python 核心基础、面向对象与工程实践、AI 开发工具链 |
| 1 | 基础概念 | LangGraph 框架对比、上手案例、LangChain 回顾、术语汇总 |
| 2 | 核心组件 | Simple Graph、Chain、Router、Agent、Agent Memory、Deployment |
| 3 | 核心机制 | State Schema、Reducers、Multiple Schemas、Trim & Filter、Summarization、External Memory |
| 4 | 人机协作 | Breakpoints、Dynamic Breakpoints、Human Feedback、Streaming、Time Travel、HITL |
| 5 | 高级模式 | Parallelization、Sub-graph、Map-Reduce、Research Assistant |
| 6 | 记忆系统 | Memory Agent、Memory Store、Memory Schema（Profile / Collection） |
| 7 | 生产部署 | Creating Deployments、Connecting、Double-texting、Assistant |
| 8 | 经典案例 | Agent Simulation Evaluation、Information Gather Prompting、代码助手 RAG+自我纠正 |
| 9 | 高级研究助手 | 需求澄清、研究智能体、MCP 集成、多智能体协同、完整系统集成 |
| 10 | TradingAgent | 架构总览、状态管理、工具系统、研究员辩论、条件逻辑、端到端执行、NVDA 实战 |
| 11 | 精华总结 | LangGraph 循环对比、Mermaid 与 Graph 架构图 |
| 18 | Claude Quickstarts | Customer Support Agent、Financial Data Analyst、Computer Use、Browser Use、Autonomous Coding、Agents |

## ⚠️ LangChain 1.x 与旧版导入差异

本项目使用 **LangChain 1.2.15**（最新官方版本）。LangChain 在 1.x 版本中对包结构进行了重大调整，`langchain` 包本身已成为元包（metapackage），核心功能拆分至独立的子包中。

### 常见导入路径变更

| 旧版写法（< 1.0 / < 0.2） | 新版写法（1.x） |
|---------------------------|----------------|
| `from langchain.prompts import ChatPromptTemplate` | `from langchain_core.prompts import ChatPromptTemplate` |
| `from langchain.schema import HumanMessage, SystemMessage` | `from langchain_core.messages import HumanMessage, SystemMessage` |
| `from langchain.schema.output_parser import StrOutputParser` | `from langchain_core.output_parsers import StrOutputParser` |
| `from langchain.chat_models import ChatOpenAI` | `from langchain_openai import ChatOpenAI` |
| `from langchain.embeddings import OpenAIEmbeddings` | `from langchain_openai import OpenAIEmbeddings` |
| `from langchain.vectorstores import Chroma` | `from langchain_chroma import Chroma` |

### 核心包说明

- **`langchain_core`**：提供基础抽象类、消息类型、提示模板、输出解析器等核心组件
- **`langchain_openai`**：OpenAI 模型与嵌入的集成
- **`langgraph`**：用于构建 Multi-Agent 工作流的编排框架

> 课程中的旧版导入代码需按上表进行替换，否则会出现 `ModuleNotFoundError`。

## 🚀 学习建议

1. **循序渐进**：按章节顺序学习，每个章节都建立在前面的基础上
2. **动手实践**：每个概念都配有可运行的代码，边学边练
3. **理解为王**：重点理解核心思想，而非记忆技术细节
4. **多做实验**：尝试修改示例代码，观察不同参数的影响
5. **案例驱动**：Module 9 和 Module 10 是完整的实战项目，强烈推荐

## 🔗 相关链接

- 课程官网：[https://www.learngraph.online](https://www.learngraph.online)
- 本仓库用于存放课程学习过程中的代码练习、笔记与项目实现

---

*Happy Coding with LangGraph!*
