"""
AutoGen v0.4+ 新版 API 示例（含代码执行）
新版核心变化：
- 使用 model_client 替代 llm_config
- 异步调用（async/await）
- 代码执行使用 CodeExecutorAgent + LocalCommandLineCodeExecutor
- 使用 RoundRobinGroupChat 组织多 Agent 协作
"""
import asyncio
import os
from pathlib import Path
from dotenv import load_dotenv

# 加载环境变量
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.code_executors import LocalCommandLineCodeExecutor
from autogen_agentchat.agents import AssistantAgent, CodeExecutorAgent
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo

async def dialog_stream_mode():
    """交互式对话模式（流式输出）：用户输入 -> 助手逐块回复"""
    from autogen_core.models._types import SystemMessage, UserMessage, AssistantMessage

    print("== AutoGen v0.4+ 流式对话 ==\n输入 'exit' 或 'quit' 退出\n")
    model_client = OpenAIChatCompletionClient(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="unknown",
        ),
    )

    # 维护对话历史
    messages = [
        SystemMessage(content="你是一个 helpful 的助手，可以回答各种问题。", source="system")
    ]

    while True:
        try:
            user_input = input("[user] ")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in ("exit", "quit"):
            print("[system] 对话结束")
            break

        messages.append(UserMessage(content=user_input, source="user"))

        # 直接调用底层 create_stream 实现真正的流式输出
        print("\n[assistant] ", end="", flush=True)
        full_content = ""
        async for chunk in model_client.create_stream(
            messages=messages,
            cancellation_token=CancellationToken(),
        ):
            if isinstance(chunk, str):
                print(chunk, end="", flush=True)
                full_content += chunk
        print("\n")

        messages.append(AssistantMessage(content=full_content, source="assistant"))

    await model_client.close()

async def dialog_mode():
    """交互式对话模式：用户输入 -> 助手回复，支持多轮对话"""
    print("== AutoGen v0.4+ 交互式对话 ==\n输入 'exit' 或 'quit' 退出\n")

    model_client = OpenAIChatCompletionClient(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="unknown",
        ),
    )

    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="你是一个 helpful 的助手，可以回答各种问题。",
    )

    while True:
        try:
            user_input = input("[user] ")
        except (EOFError, KeyboardInterrupt):
            break

        if user_input.strip().lower() in ("exit", "quit"):
            print("[system] 对话结束")
            break

        response = await assistant.on_messages(
            [TextMessage(content=user_input, source="user")],
            cancellation_token=CancellationToken(),
        )

        print(f"\n[assistant] {response.chat_message.content}\n")

    await model_client.close()

async def code_execution_mode():
    # 1. 配置模型客户端（通过 OpenAI 兼容接口调用千问）
    model_client = OpenAIChatCompletionClient(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="unknown",
        ),
    )


    # 2. 创建代码执行器 Agent（替代旧版 UserProxyAgent 的 code_execution_config）
    code_executor = LocalCommandLineCodeExecutor(work_dir="coding")
    code_executor_agent = CodeExecutorAgent(
        name="code_executor",
        code_executor=code_executor,
    )

    # 3. 创建助手 Agent
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message=(
            "你是一个 helpful 的编程助手。"
            "当用户要求写代码时，请用 ```python ... ``` 代码块格式输出代码，"
            "code_executor 会自动提取并执行代码。"
        ),
    )

    # 4. 创建团队：助手 和 代码执行器 轮流协作
    team = RoundRobinGroupChat(
        participants=[assistant, code_executor_agent],
        termination_condition=MaxMessageTermination(max_messages=6),
    )

    # 5. 运行对话流
    print("== AutoGen v0.4+ 代码执行对话开始 ==\n")
    async for message in team.run_stream(
        task="帮我写一个计算斐波那契数列的Python函数，并执行验证输出前10项"
    ):
        # run_stream 产出两类对象：普通消息 和 最终的 TaskResult
        if hasattr(message, "source") and hasattr(message, "content"):
            source = message.source
            content = message.content
            if content:
                print(f"[{source}] {content[:300]}{'...' if len(content) > 300 else ''}\n")
        elif hasattr(message, "stop_reason"):
            # TaskResult 结束对象
            print(f"[team] 对话结束，原因: {message.stop_reason}\n")

    # 关闭模型客户端
    await model_client.close()


async def group_chat_mode():
    """多角色群聊模式：工程师、科学家、批评家围绕任务协作讨论"""
    model_client = OpenAIChatCompletionClient(
        model="qwen-plus",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        model_info=ModelInfo(
            vision=False,
            function_calling=True,
            json_output=True,
            structured_output=True,
            family="unknown",
        ),
    )

    engineer = AssistantAgent(
        name="engineer",
        model_client=model_client,
        system_message="你是软件工程师，擅长编写代码、设计系统架构和解决技术问题。",
    )

    scientist = AssistantAgent(
        name="scientist",
        model_client=model_client,
        system_message="你是科学家，擅长数据分析、理论推导和实验设计。",
    )

    critic = AssistantAgent(
        name="critic",
        model_client=model_client,
        system_message="你是批评家，负责审查其他人的观点，指出潜在问题和改进建议。",
    )

    team = RoundRobinGroupChat(
        participants=[engineer, scientist, critic],
        termination_condition=MaxMessageTermination(max_messages=10),
    )

    task = input("请输入群聊任务: ").strip()
    if not task:
        task = "讨论如何设计一个高效的缓存系统"

    print(f"\n== AutoGen v0.4+ 群聊开始 ==\n任务: {task}\n")
    async for message in team.run_stream(task=task):
        if hasattr(message, "source") and hasattr(message, "content"):
            source = message.source
            content = message.content
            if content:
                print(f"[{source}] {content[:300]}{'...' if len(content) > 300 else ''}\n")
        elif hasattr(message, "stop_reason"):
            print(f"[team] 群聊结束，原因: {message.stop_reason}\n")

    await model_client.close()
async def main():
    print("=" * 40)
    print("AutoGen v0.4+ 测试模式选择")
    print("=" * 40)
    print("1. 交互式对话（非流式）- dialog_mode")
    print("2. 交互式对话（流式）   - dialog_stream_mode")
    print("3. 代码执行模式         - code_execution_mode")
    print("4. 多角色群聊模式       - group_chat_mode")
    print("=" * 40)

    choice = input("请选择模式 (1/2/3/4): ").strip()

    if choice == "1":
        await dialog_mode()
    elif choice == "2":
        await dialog_stream_mode()
    elif choice == "3":
        await code_execution_mode()
    elif choice == "4":
        await group_chat_mode()
    else:
        print("无效选择，默认运行交互式对话（非流式）")
        await dialog_mode()
if __name__ == "__main__":
    asyncio.run(main())
