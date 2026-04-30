from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage,SystemMessage

print("== LangChain Core Concept ===\n")
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system","你是一个专业的{role}."),
        ("human","{user_input}")
        ]
    )
formatted = prompt_template.format_messages(
    role="天气助手",
    user_input="中山今天天气怎样"
    )
print("1.Prompt Template:")
for msg in formatted:
    print(f"[{msg.type}] {msg.content}")


print ("\n2. chain function call:")
print("""
user input
  ↓
[prompt Template] -> format intput
  ↓
[LLM] -> Generate Response
  ↓
[Output Parser] -> parse output
  ↓
Final Response
""")

print("3.Tools:")
tools_example = """
- 搜索工具:实时信息查询
- 计算器工具:执行数学运算
- 数据库工具:查询数据库
- api工具:调用外部api
"""
print(tools_example)

print("\n4. langChain arch:")
print("""
[application level - chatbots,agents,QA system]
[chain - LLMChain,SequentialChain]
[component - promps,LLMs,Tools]
[Models - OpenAI,QWen,Anthropic etc.]
""")
