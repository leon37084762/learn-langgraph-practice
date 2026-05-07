import os

# 使用 HuggingFace 国内镜像加速模型下载
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import chromadb
from chromadb.config import Settings

"""
=== 版本改进说明 ===

【以前版本的错误与局限】
1. 字符串拼接陷阱：
   documents 列表中相邻字符串缺少逗号，Python 自动将其拼接为一个字符串，
   导致 documents 长度=1，与 ids/metadatas 长度=4 不匹配，触发 ValueError。

2. 默认距离度量问题：
   未指定距离度量，ChromaDB 默认使用 squared L2，范围 [0, +∞)。
   代码中用 "1 - distance" 计算相似度，当距离 > 1 时会出现负数相似度，
   这个公式在数学上完全不成立。

3. 知识库内容匮乏：
   仅 4 条通用介绍性文档，与查询"如何构建 AI Agent"语义完全不相关。
   查询结果被迫返回不相关内容，无法体现 RAG 的实际价值。

4. 查询结果展示简陋：
   只打印文档内容和错误的相似度，没有元数据信息、相关性分级和距离指标。

【本次改进内容】
1. 修复字符串拼接：为每个文档字符串添加逗号分隔，确保列表元素独立。
2. 改用余弦距离：metadata={"hnsw:space": "cosine"}，距离范围 [0, 2]，
   相似度公式改为 "1 - dist / 2"，正确归一化到 [0, 1]。
3. 扩充知识库：从 4 条增加到 14 条，新增大量 AI Agent 核心知识
   （ReAct、Tool Use、Multi-Agent、AutoGen、CrewAI、LangGraph、记忆系统等）。
4. 直观结果展示：
   - 同时显示相似度和原始距离
   - 添加相关性分级（✅ 高相关 / ⚠️ 中等相关 / ❌ 低相关）
   - 展示分类、主题等元数据
   - 元数据过滤查询示例更丰富
"""

print("=== chromadb vector db ===")
client = chromadb.Client(Settings(
    persist_directory="./chroma_db",
    anonymized_telemetry=False
))

# 删除旧集合（若存在），确保使用新的距离度量重新创建
if "my_knowledge_base" in [c.name for c in client.list_collections()]:
    client.delete_collection("my_knowledge_base")

collection = client.get_or_create_collection(
    name="my_knowledge_base",
    metadata={"hnsw:space": "cosine"}  # 使用余弦距离，范围 [0, 2]
)

# 扩充知识库：添加大量与 AI Agent 相关的知识
documents = [
    # 原有内容
    "LangGraph 是一个用于构建语言图的库，支持状态管理和多 Agent 协作。",
    "Streamlit 是一个用于构建 Web 应用的 Python 框架，适合快速搭建原型界面。",
    "ChromaDB 是一个开源向量数据库，用于存储和检索文本嵌入向量。",
    "RAG（检索增强生成）是一种将外部知识库与 LLM 结合的技术，通过检索相关文档来增强回答质量。",
    # AI Agent 核心知识
    "AI Agent 是指能够感知环境、做出决策并执行动作的自主智能体，通常由 LLM 作为大脑驱动。",
    "构建 AI Agent 的核心步骤包括：定义角色与目标、选择底层 LLM、设计工具调用（Tool Use）、构建记忆系统、实现规划与反思循环。",
    "ReAct 是一种经典的 Agent 推理框架，将推理（Reasoning）和行动（Acting）交替进行，让 Agent 能够逐步解决问题。",
    "Tool Use（工具调用）是 Agent 扩展能力的关键机制，允许 Agent 调用搜索引擎、代码执行器、数据库查询等外部工具。",
    "Multi-Agent 系统由多个独立 Agent 组成，每个 Agent 负责不同任务，通过协作完成复杂目标。",
    "AutoGen 是微软开源的多 Agent 对话框架，支持自定义 Agent、群聊、代码执行等功能。",
    "CrewAI 是一个用于编排多 Agent 协作的框架，通过定义角色、任务和流程来组织 Agent 团队。",
    "LangGraph 基于 LangChain 构建，使用图结构来定义 Agent 的工作流，支持循环、条件分支和状态持久化。",
    "Agent 的记忆系统通常分为短期记忆（对话上下文）和长期记忆（知识库存储），RAG 是实现长期记忆的常用方案。",
    "Chain-of-Thought（思维链）提示技术通过引导 LLM 逐步推理，显著提升 Agent 解决复杂问题的能力。",
]
ids = [f"doc{i+1}" for i in range(len(documents))]
metadatas = [
    {"category": "framework", "topic": "agent"},
    {"category": "framework", "topic": "ui"},
    {"category": "database", "topic": "vector"},
    {"category": "technique", "topic": "rag"},
    {"category": "concept", "topic": "agent"},
    {"category": "guide", "topic": "agent"},
    {"category": "framework", "topic": "reasoning"},
    {"category": "concept", "topic": "tool"},
    {"category": "concept", "topic": "multi-agent"},
    {"category": "framework", "topic": "autogen"},
    {"category": "framework", "topic": "crewai"},
    {"category": "framework", "topic": "langgraph"},
    {"category": "concept", "topic": "memory"},
    {"category": "technique", "topic": "prompt"},
]

collection.add(
    documents=documents,
    ids=ids,
    metadatas=metadatas,
)

print(f"✅ 已添加 {len(documents)} 个文档到向量数据库\n")

# 统计信息
count = collection.count()
print(f"📊 总文档数: {count}")

# 4. 语义搜索 + 直观结果展示（循环输入模式）
print("\n" + "=" * 60)
print("💡 语义搜索模式：输入查询内容，观察向量检索结果")
print("   输入 'exit' 或 'quit' 退出循环")
print("=" * 60)

while True:
    query = input("\n🔍 请输入查询内容: ").strip()
    if query.lower() in ("exit", "quit"):
        print("👋 退出查询模式")
        break
    if not query:
        continue

    results = collection.query(
        query_texts=[query],
        n_results=5,
        include=["documents", "distances", "metadatas"]
    )

    print(f"\n📌 查询: \"{query}\"")
    print("-" * 60)
    for i, (doc, dist, meta) in enumerate(zip(
        results['documents'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        # 余弦距离 [0, 2] 转相似度 [0, 1]
        similarity = 1 - dist / 2
        relevance = "✅ 高相关" if similarity > 0.75 else ("⚠️ 中等相关" if similarity > 0.5 else "❌ 低相关")
        print(f"\n  排名 {i+1}  {relevance}  (相似度: {similarity:.3f} | 距离: {dist:.3f})")
        print(f"  分类: {meta['category']} | 主题: {meta['topic']}")
        print(f"  内容: {doc}")

# 5. 按元数据过滤查询示例
print("\n" + "=" * 60)
print("📂 元数据过滤查询示例")
print("=" * 60)

filtered_results = collection.query(
    query_texts=["AI Agent 框架对比"],
    where={"category": "framework"},
    n_results=5,
    include=["documents", "distances", "metadatas"]
)

print(f"\n🔍 过滤查询: \"AI Agent 框架对比\"  (category=framework)")
print("-" * 60)
for i, (doc, dist, meta) in enumerate(zip(
    filtered_results['documents'][0],
    filtered_results['distances'][0],
    filtered_results['metadatas'][0]
)):
    similarity = 1 - dist / 2
    print(f"\n  排名 {i+1}  相似度: {similarity:.3f}")
    print(f"  主题: {meta['topic']}")
    print(f"  内容: {doc}")

print("\n\n=== ChromaDB 核心概念 ===")
print("""
1. Collection（集合）:
   - 存储相关文档的容器
   - 每个集合有独立的向量空间

2. Document（文档）:
   - 文本内容
   - 自动生成向量嵌入
   - 可附加元数据

3. Metadata（元数据）:
   - 附加结构化信息
   - 支持过滤查询

4. 查询方式:
   - query_texts: 语义搜索
   - where: 元数据过滤
   - n_results: 返回数量

5. 持久化:
   - persist_directory: 本地存储路径
   - 自动保存
""")