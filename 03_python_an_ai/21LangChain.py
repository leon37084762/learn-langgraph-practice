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