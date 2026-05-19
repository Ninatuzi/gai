"""
ChatWiki Gradio 可视化界面（多会话 + 流式输出 + 会话持久化）

功能：
- 左侧侧边栏：会话列表（新建/切换/保留历史，重启不丢失）
- 中间：对话区（Chatbot，流式输出）
- 右侧：监测指标面板（流结束后刷新）

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
sessions: Dict[str, Dict] = {}
# 存最近一次问答的指标，供 .then() 回调读取
last_metrics: Dict[str, str] = {}

SESSIONS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "gradio_sessions.json")


def _save_sessions():
    os.makedirs(os.path.dirname(SESSIONS_FILE), exist_ok=True)
    try:
        with open(SESSIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(sessions, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning("保存会话文件失败: %s", e)


def _load_sessions():
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
    ag = get_agent()
    ws_id = ag.create_workspace(f"gradio_{int(time.time())}")
    session_name = name or f"会话 {len(sessions) + 1}"
    sessions[ws_id] = {"name": session_name, "history": []}
    _save_sessions()
    return ws_id


def get_session_list():
    if not sessions:
        return []
    return [sessions[ws_id]["name"] for ws_id in sessions]


def get_ws_id_by_name(name):
    if name is None or not sessions:
        return None
    for ws_id, info in sessions.items():
        if info["name"] == name:
            return ws_id
    return None


# ============================================================
# 流式交互：只更新 chatbot + workspace_state
# ============================================================

def chat_fn_stream(user_message, chat_history, workspace_id):
    """流式输出：只负责更新对话区，指标在 .then() 里更新"""
    global last_metrics

    if not user_message.strip():
        last_metrics = {}
        yield chat_history, workspace_id
        return

    ag = get_agent()

    if not workspace_id or workspace_id not in sessions:
        workspace_id = create_new_session()

    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": "思考中..."})

    yield chat_history, workspace_id

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
            yield chat_history, workspace_id

    elapsed = time.time() - t0
    if result is None:
        result = {}

    # 保存对话历史
    sessions[workspace_id]["history"] = chat_history
    _save_sessions()

    # 组装指标存到全局变量
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

    last_metrics = {
        "intent": intent,
        "rewrite": rewrite_display,
        "wiki": wiki_hit_str,
        "rag": rag_hit_str,
        "module": module_display,
        "time": time_display,
        "steps": steps_display,
        "modules_md": modules_md,
    }

    # 最终 yield
    yield chat_history, workspace_id


def get_last_metrics():
    """流式结束后被 .then() 调用，更新指标面板 + 清空输入框"""
    m = last_metrics
    return (
        m.get("intent", ""),
        m.get("rewrite", ""),
        m.get("wiki", ""),
        m.get("rag", ""),
        m.get("module", ""),
        m.get("time", ""),
        m.get("steps", ""),
        m.get("modules_md", ""),
        "",  # 清空输入框
    )


# ============================================================
# 其他回调
# ============================================================

def new_session_fn():
    ws_id = create_new_session()
    choices = get_session_list()
    selected = sessions[ws_id]["name"]
    return (
        [], ws_id,
        gr.update(choices=choices, value=selected),
        "", "", "", "", "", "", "",
    )


def delete_session_fn(selected_name, workspace_id):
    """删除选中的会话（workspace + 数据 + 向量）"""
    ws_id = get_ws_id_by_name(selected_name)
    if ws_id is None:
        choices = get_session_list()
        return [], workspace_id, gr.update(choices=choices, value=None), "", "", "", "", "", "", ""

    # 删除 workspace 数据（JSON + Milvus）
    try:
        ag = get_agent()
        ag.delete_workspace(ws_id)
    except Exception as e:
        logger.warning("删除 workspace 失败: %s", e)

    # 从 sessions 移除
    sessions.pop(ws_id, None)
    _save_sessions()

    # 更新 UI
    choices = get_session_list()
    # 如果删的是当前会话，清空显示
    new_ws_id = "" if ws_id == workspace_id else workspace_id
    new_history = []
    if new_ws_id and new_ws_id in sessions:
        new_history = sessions[new_ws_id].get("history", [])

    return (
        new_history, new_ws_id,
        gr.update(choices=choices, value=None),
        "", "", "", "", "", "", "",
    )


def switch_session_fn(selected_name, workspace_id):
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
                del_session_btn = gr.Button("- 删除会话", variant="stop", size="sm")
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
        # 流式只更新 chatbot + workspace
        stream_outputs = [chatbot, workspace_state]

        # 指标 + 清空输入框（流结束后触发）
        metric_outputs = [
            intent_display, rewrite_display, wiki_display,
            rag_display, module_display, time_display, steps_display,
            modules_display, msg_input,
        ]

        # Enter 发送
        msg_input.submit(
            fn=chat_fn_stream,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=stream_outputs,
        ).then(
            fn=get_last_metrics,
            outputs=metric_outputs,
        )

        # 点击发送
        send_btn.click(
            fn=chat_fn_stream,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=stream_outputs,
        ).then(
            fn=get_last_metrics,
            outputs=metric_outputs,
        )

        # 新建会话
        new_session_btn.click(
            fn=new_session_fn,
            outputs=[
                chatbot, workspace_state, session_radio,
                intent_display, rewrite_display, wiki_display,
                rag_display, module_display, time_display, steps_display,
            ],
        )

        # 删除会话
        del_session_btn.click(
            fn=delete_session_fn,
            inputs=[session_radio, workspace_state],
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
