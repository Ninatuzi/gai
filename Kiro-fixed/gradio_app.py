"""
ChatWiki Gradio 可视化界面（多会话 + 流式输出 + 会话持久化）

功能：
- 左侧侧边栏：会话列表（新建/切换/保留历史，重启不丢失）
- 中间：对话区（Chatbot，流式输出）
- 右侧：监测指标面板

启动：
    python gradio_app.py
"""

import json
import logging
import os
import time
from typing import Any, Dict, List

import gradio as gr

from chatwiki import ChatWikiAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatwiki.gradio")

# ============================================================
# 全局 Agent 实例 + 会话管理（持久化）
# ============================================================
agent: ChatWikiAgent = None
# 存储所有会话: {workspace_id: {"name": str, "history": list}}
sessions: Dict[str, Dict] = {}

# 会话持久化文件路径
SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gradio_sessions.json")


def _save_sessions():
    """保存会话列表到文件"""
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("保存会话文件失败: %s", e)


def _load_sessions():
    """从文件加载会话列表"""
    global sessions
    if os.path.exists(SESSIONS_FILE):
        try:
            with open(SESSIONS_FILE, "r", encoding="utf-8") as f:
                sessions = json.load(f)
            logger.info("恢复 %d 个历史会话", len(sessions))
        except Exception as e:
            logger.warning("加载会话文件失败: %s", e)
            sessions = {}


def get_agent() -> ChatWikiAgent:
    global agent
    if agent is None:
        agent = ChatWikiAgent()
        agent.init()
        logger.info("ChatWiki Agent 初始化完成")
    return agent


def create_new_session(name=None):
    """创建新会话，返回 workspace_id"""
    ag = get_agent()
    ws_id = ag.create_workspace(f"gradio_{int(time.time())}")
    session_name = name or f"会话 {len(sessions) + 1}"
    sessions[ws_id] = {"name": session_name, "history": []}
    _save_sessions()
    return ws_id


def get_session_list():
    """返回会话列表给 Radio 组件"""
    if not sessions:
        return []
    return [sessions[ws_id]["name"] for ws_id in sessions]


def get_ws_id_by_name(name):
    """根据会话名称获取 workspace_id"""
    if name is None or not sessions:
        return None
    for ws_id, info in sessions.items():
        if info["name"] == name:
            return ws_id
    return None


# ============================================================
# 流式交互函数
# ============================================================

def chat_fn_stream(user_message, chat_history, workspace_id):
    """流式输出版本"""
    if not user_message.strip():
        yield chat_history, "", "", "", "", "", "", "", "", workspace_id
        return

    ag = get_agent()

    # 如果没有 workspace，自动创建一个
    if not workspace_id or workspace_id not in sessions:
        workspace_id = create_new_session()

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": "思考中..."})

    yield chat_history, "处理中...", "", "", "", "", "", "", "", workspace_id

    t0 = time.time()
    full_answer = ""
    result = None

    for token, is_final, final_result in ag.ask_stream(workspace_id=workspace_id, query=user_message):
        if is_final:
            result = final_result
            if token:
                full_answer = token
                chat_history[-1]["content"] = full_answer
        else:
            full_answer += token
            chat_history[-1]["content"] = full_answer

        if not is_final:
            yield chat_history, "", "", "", "", "", "", "", "", workspace_id

    elapsed = time.time() - t0

    if result is None:
        result = {}

    # 保存到 session 并持久化
    sessions[workspace_id]["history"] = chat_history
    _save_sessions()

    # 组装指标
    intent = result.get("intent", "")
    rewritten = result.get("rewritten_query", "")
    if rewritten == user_message:
        rewrite_display = "未改写（问题已完整）"
    else:
        rewrite_display = rewritten or "未改写"

    wiki_hit_str = "未命中"
    if result.get("wiki_found"):
        wiki_hit_str = f"命中 {result.get('wiki_modules_hit', 0)} 模块 / {result.get('wiki_count', 0)} 条wiki"

    if result.get("intent") == "chitchat":
        rag_hit_str = "未调用（闲聊）"
    elif result.get("rag_found"):
        rag_hit_str = f"命中 {result.get('rag_chunks_count', 0)} 条"
    else:
        rag_hit_str = "未命中"

    module_id = result.get("module_id", "")
    module_display = "无（闲聊）"
    if module_id:
        modules = ag.get_modules(workspace_id)
        for m in modules:
            if m["module_id"] == module_id:
                module_display = f"{m['topic']} ({module_id[:8]}...)"
                break
        else:
            module_display = f"{module_id[:8]}..."

    time_display = f"{elapsed:.1f}s"
    steps = result.get("steps", [])
    steps_display = " -> ".join(steps) if steps else ""
    modules_md = _format_modules_markdown(ag, workspace_id)

    yield (
        chat_history, intent, rewrite_display, wiki_hit_str,
        rag_hit_str, module_display, time_display, steps_display,
        modules_md, workspace_id,
    )


def new_session_fn():
    """点击新建会话"""
    ws_id = create_new_session()
    choices = get_session_list()
    selected = sessions[ws_id]["name"]
    return (
        [], ws_id,
        gr.update(choices=choices, value=selected),
        "", "", "", "", "", "", "",
    )


def switch_session_fn(selected_name, workspace_id):
    """点击侧边栏切换会话"""
    ws_id = get_ws_id_by_name(selected_name)
    if ws_id is None:
        return [], workspace_id, "", "", "", "", "", "", ""

    history = sessions[ws_id].get("history", [])
    ag = get_agent()
    modules_md = _format_modules_markdown(ag, ws_id)

    return history, ws_id, "", "", "", "", "", "", modules_md


def _format_modules_markdown(ag, workspace_id):
    modules = ag.get_modules(workspace_id)
    if not modules:
        return ""
    lines = []
    for i, m in enumerate(modules, 1):
        topic = m.get("topic", "")
        wiki_count = m.get("wiki_count", 0)
        summary = m.get("summary", "")
        lines.append(f"**{i}. {topic}** ({wiki_count}条wiki)")
        if summary:
            lines.append(f"   {summary[:80]}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# Gradio 界面构建
# ============================================================

def build_app():
    # 启动时加载历史会话
    _load_sessions()
    initial_choices = get_session_list()

    with gr.Blocks(title="ChatWiki") as app:
        gr.Markdown("# ChatWiki - 对话级记忆检索 Agent")

        workspace_state = gr.State(value="")

        with gr.Row():
            # ========== 左侧：会话列表 ==========
            with gr.Column(scale=2, min_width=180):
                gr.Markdown("### 会话列表")
                new_session_btn = gr.Button("+ 新建会话", variant="primary", size="sm")
                session_radio = gr.Radio(
                    choices=initial_choices,
                    label="",
                    interactive=True,
                )

            # ========== 中间：对话区 ==========
            with gr.Column(scale=5):
                chatbot = gr.Chatbot(label="对话", height=520)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入问题后按 Enter 发送...",
                        scale=8, show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)

            # ========== 右侧：监测面板 ==========
            with gr.Column(scale=3):
                gr.Markdown("### 监测指标")
                intent_display = gr.Textbox(label="意图识别", interactive=False, max_lines=1)
                rewrite_display = gr.Textbox(label="问题改写/增强", interactive=False, max_lines=2)
                with gr.Row():
                    wiki_display = gr.Textbox(label="Wiki 命中", interactive=False, max_lines=1)
                    rag_display = gr.Textbox(label="RAG 命中", interactive=False, max_lines=1)
                with gr.Row():
                    module_display = gr.Textbox(label="归属模块", interactive=False, max_lines=1)
                    time_display = gr.Textbox(label="耗时", interactive=False, max_lines=1)
                steps_display = gr.Textbox(label="执行步骤", interactive=False, max_lines=3)
                gr.Markdown("### 话题模块列表")
                modules_display = gr.Markdown(value="")

        # ========== 事件绑定 ==========
        chat_outputs = [
            chatbot, intent_display, rewrite_display, wiki_display,
            rag_display, module_display, time_display, steps_display,
            modules_display, workspace_state,
        ]

        # 发送消息
        msg_input.submit(
            fn=chat_fn_stream,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=chat_outputs,
        ).then(fn=lambda: "", outputs=msg_input)

        send_btn.click(
            fn=chat_fn_stream,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=chat_outputs,
        ).then(fn=lambda: "", outputs=msg_input)

        # 新建会话
        new_session_btn.click(
            fn=new_session_fn,
            outputs=[
                chatbot, workspace_state, session_radio,
                intent_display, rewrite_display, wiki_display,
                rag_display, module_display, time_display, steps_display,
            ],
        )

        # 切换会话
        session_radio.change(
            fn=switch_session_fn,
            inputs=[session_radio, workspace_state],
            outputs=[
                chatbot, workspace_state,
                intent_display, rewrite_display, wiki_display,
                rag_display, module_display, time_display, modules_display,
            ],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
