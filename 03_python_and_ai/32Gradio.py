import asyncio
import concurrent.futures
import os
import queue
import threading
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 加载环境变量
load_dotenv(dotenv_path=Path(__file__).parent / ".env")


def get_model_client():
    """创建模型客户端"""
    return OpenAIChatCompletionClient(
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


def _build_messages(message, history):
    """将 Gradio history 转为 AutoGen 消息列表"""
    messages = []
    for user_msg, bot_msg in history:
        messages.append(TextMessage(content=user_msg, source="user"))
        if bot_msg:
            messages.append(TextMessage(content=bot_msg, source="assistant"))
    messages.append(TextMessage(content=message, source="user"))
    return messages


async def _get_response_async(message, history):
    """非流式：获取完整响应"""
    model_client = get_model_client()
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="你是一个 helpful 的助手，可以回答各种问题。",
    )
    messages = _build_messages(message, history)
    response = await assistant.on_messages(
        messages,
        cancellation_token=CancellationToken(),
    )
    result = response.chat_message.content
    await model_client.close()
    return result


async def _stream_response_async(message, history, q):
    """流式：通过队列逐块传递响应"""
    from autogen_core.models._types import SystemMessage, UserMessage, AssistantMessage

    model_client = get_model_client()
    messages = [
        SystemMessage(content="你是一个 helpful 的助手，可以回答各种问题。", source="system")
    ]
    for user_msg, bot_msg in history:
        messages.append(UserMessage(content=user_msg, source="user"))
        messages.append(AssistantMessage(content=bot_msg, source="assistant"))
    messages.append(UserMessage(content=message, source="user"))

    async for chunk in model_client.create_stream(
        messages=messages,
        cancellation_token=CancellationToken(),
    ):
        if isinstance(chunk, str):
            q.put(chunk)
    q.put(None)  # 结束标记
    await model_client.close()


def chat_function(message, history, mode):
    """Gradio 入口（始终返回 generator，支持流式/非流式切换）"""
    if mode == "非流式":
        # 非流式：在线程池中运行异步调用，yield 一次完整响应
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(asyncio.run, _get_response_async(message, history))
            full_text = future.result()
        yield full_text

    else:  # 流式
        q = queue.Queue()

        async def _async_stream():
            await _stream_response_async(message, history, q)

        def _run():
            asyncio.run(_async_stream())

        thread = threading.Thread(target=_run)
        thread.start()

        full_content = ""
        while True:
            chunk = q.get()
            if chunk is None:
                break
            full_content += chunk
            yield full_content  # Gradio 更新当前消息为累积内容

        thread.join()


# Gradio 界面
demo = gr.ChatInterface(
    fn=chat_function,
    title="🤖 AI ChatBot",
    description="基于 AutoGen v0.4+ 的智能对话助手（DashScope Qwen）",
    examples=[
        ["给我分析一下今年全球天气", "非流式"],
        ["你是谁", "非流式"],
        ["用 Python 写一个快速排序", "流式"],
    ],
    additional_inputs=[
        gr.Radio(
            choices=["非流式", "流式"],
            value="非流式",
            label="输出模式",
        )
    ],
)

if __name__ == "__main__":
    demo.launch(share=True)
