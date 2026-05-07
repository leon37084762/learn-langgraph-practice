# 安装：pip install streamlit
# 运行：streamlit run 31_streamLit.py
# 注意：需在 learn_lang_graph 环境下运行

import asyncio
import os
import queue
import sqlite3
import threading
import uuid
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_core.models import ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient

# 加载环境变量
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

# 数据库路径
DB_PATH = Path(__file__).parent / "chat_history.db"


# ========== 数据库操作 ==========

def init_db():
    """初始化 SQLite 数据库"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT UNIQUE NOT NULL,
            title TEXT,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (session_id) REFERENCES sessions(session_id)
        )
    """)
    conn.commit()
    conn.close()


def create_session(title: str = None) -> str:
    """创建新会话，返回 session_id"""
    session_id = str(uuid.uuid4())[:8]
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO sessions (session_id, title) VALUES (?, ?)",
        (session_id, title or f"会话 {datetime.now().strftime('%m-%d %H:%M')}"),
    )
    conn.commit()
    conn.close()
    return session_id


def get_all_sessions():
    """获取所有会话列表"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT session_id, title, created_at FROM sessions ORDER BY created_at DESC"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


def delete_session(session_id: str):
    """删除会话及其消息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


def save_message(session_id: str, role: str, content: str):
    """保存单条消息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
        (session_id, role, content),
    )
    conn.commit()
    conn.close()


def load_messages(session_id: str):
    """加载指定会话的所有消息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
    )
    rows = cursor.fetchall()
    conn.close()
    return [{"role": row[0], "content": row[1]} for row in rows]


def clear_messages(session_id: str):
    """清空指定会话的消息"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()


# ========== 业务逻辑函数 ==========

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


async def get_response_normal(prompt: str) -> str:
    """非流式对话"""
    model_client = get_model_client()
    assistant = AssistantAgent(
        name="assistant",
        model_client=model_client,
        system_message="你是一个 helpful 的助手，可以回答各种问题。",
    )
    response = await assistant.on_messages(
        [TextMessage(content=prompt, source="user")],
        cancellation_token=CancellationToken(),
    )
    result = response.chat_message.content
    await model_client.close()
    return result


def stream_generator(prompt: str):
    """同步生成器：通过线程+队列桥接异步流式输出，实现真正的边获取边显示"""
    from autogen_core.models._types import SystemMessage, UserMessage

    q = queue.Queue()

    async def _async_fetch():
        model_client = get_model_client()
        messages = [
            SystemMessage(content="你是一个 helpful 的助手，可以回答各种问题。", source="system"),
            UserMessage(content=prompt, source="user"),
        ]
        async for chunk in model_client.create_stream(
            messages=messages,
            cancellation_token=CancellationToken(),
        ):
            if isinstance(chunk, str):
                q.put(chunk)
        q.put(None)  # 结束标记
        await model_client.close()

    def _run_async():
        asyncio.run(_async_fetch())

    thread = threading.Thread(target=_run_async)
    thread.start()

    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk

    thread.join()


# ========== Streamlit 界面 ==========

st.set_page_config(
    page_title="AI ChatBot",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 AI ChatBot")
st.caption("基于 AutoGen v0.4+ 的智能对话助手")

# 初始化数据库（必须在任何数据库操作之前调用）
init_db()

# 侧边栏
with st.sidebar:
    st.header("⚙️ 配置")
    mode = st.selectbox(
        "对话模式",
        ["非流式对话", "流式对话"],
        index=0,
    )
    st.divider()
    st.info("""
    💡 **使用说明**
    - 在下方输入框输入消息
    - 按 Enter 发送
    - AI 自动回复
    """)
    if st.button("🗑️ 清除当前会话"):
        clear_messages(st.session_state.current_session_id)
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.subheader("💬 会话管理")

    if st.button("➕ 新建会话"):
        new_id = create_session()
        st.session_state.current_session_id = new_id
        st.session_state.messages = []
        st.rerun()

    sessions = get_all_sessions()
    if sessions:
        session_options = {row[1]: row[0] for row in sessions}
        selected_title = st.selectbox(
            "切换会话",
            list(session_options.keys()),
            index=0,
        )
        selected_id = session_options[selected_title]
        if selected_id != st.session_state.current_session_id:
            st.session_state.current_session_id = selected_id
            st.session_state.messages = load_messages(selected_id)
            st.rerun()

        if st.button("🗑️ 删除当前会话"):
            delete_session(st.session_state.current_session_id)
            # 切换到最新的会话，或新建一个
            remaining = get_all_sessions()
            if remaining:
                st.session_state.current_session_id = remaining[0][0]
                st.session_state.messages = load_messages(remaining[0][0])
            else:
                st.session_state.current_session_id = create_session()
                st.session_state.messages = []
            st.rerun()
    else:
        st.info("暂无会话")

# 初始化会话状态
if "current_session_id" not in st.session_state:
    # 默认创建一个新会话
    st.session_state.current_session_id = create_session()

if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.current_session_id)

# 显示对话历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# 用户输入
if prompt := st.chat_input("输入你的消息..."):
    # 保存用户消息到数据库
    save_message(st.session_state.current_session_id, "user", prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    # AI 响应
    with st.chat_message("assistant"):
        if mode == "非流式对话":
            with st.spinner("思考中..."):
                response_text = asyncio.run(get_response_normal(prompt))
            st.write(response_text)
        else:
            # 流式输出：通过线程+队列边获取边渲染
            response_text = st.write_stream(stream_generator(prompt))

    # 保存 AI 响应到数据库
    save_message(st.session_state.current_session_id, "assistant", response_text)
    st.session_state.messages.append({"role": "assistant", "content": response_text})