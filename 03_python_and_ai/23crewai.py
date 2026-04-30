print("== CrewAI  核心概念 ==\n")
print("1. Agent Role:")
print("""
from crewai import Agent
researcher = Agent(
    name="Researcher",
    role="研究员",
    goal="收集和分析信息",
    backstory="你是一名经验丰富的研究员，擅长收集和分析各种信息。",
    tools = [search_tools,scrape_tool]
)
writer = Agent(
    name="Writer",
    role="作家",
    goal="撰写高质量的文章",
    backstory="你是一名经验丰富的作家，擅长撰写各种类型的文章。",
    tools = [write_tool]
)
""")


# 2. Task
print("\n2. Task:")
