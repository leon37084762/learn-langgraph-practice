import os
from pathlib import Path
from dotenv import load_dotenv

# 必须在 import crewai 之前设置好 OpenAI 兼容接口环境变量
load_dotenv(dotenv_path=Path(__file__).parent / ".env")
os.environ["OPENAI_API_KEY"] = os.getenv("DASHSCOPE_API_KEY")
os.environ["OPENAI_API_BASE"] = "https://dashscope.aliyuncs.com/compatible-mode/v1"

from crewai import Agent, Task, Crew, Process

# 1. 定义使用千问的 Agent（通过 OpenAI 兼容接口调用）
researcher = Agent(
    role="行业研究员",
    goal="收集并总结2026年新能源电池技术进展",
    backstory="专注硬科技领域3年，擅长提炼技术核心点。",
    llm="openai/qwen-plus",
    verbose=True
)

writer = Agent(
    role="技术撰稿人",
    goal="将研究成果转化为通俗易懂的科普文章",
    backstory="前科技媒体主编，擅长将专业内容大众化。",
    llm="openai/qwen-plus",
    verbose=True
)

# 2. 定义任务
task1 = Task(
    description="调研固态电池、钠离子电池的最新量产进展与成本趋势",
    expected_output="包含技术路线、代表企业、成本对比的要点列表",
    agent=researcher
)

task2 = Task(
    description="基于调研结果写一篇800字科普短文，面向普通投资者",
    expected_output="Markdown格式文章，含小标题与核心结论",
    agent=writer
)

# 3. 启动 Crew
crew = Crew(
    agents=[researcher, writer],
    tasks=[task1, task2],
    process=Process.sequential,
    verbose=True
)

result = crew.kickoff()
print("\n✅ 最终输出:\n", result.raw)