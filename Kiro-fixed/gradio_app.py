"""
ChatWiki Gradio 可视化界面（流式输出版）

功能：
- 左侧：对话区（Chatbot，流式输出）
- 右侧：监测指标面板

启动：
    python gradio_app.py
"""

import logging
import time
from typing import Any, Dict, List, Tuple

import gradio as gr

from chatwiki import ChatWikiAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatwiki.gradio")

# ============================================================
# 全局 Agent 实例
# ============================================================
agent: ChatWikiAgent = None


def get_agent() -> ChatWikiAgent:
    global agent
    if agent is None:
        agent = ChatWikiAgent()
        agent.init()
        logger.info("ChatWiki Agent 初始化完成")
    return agent


# ============================================================
# 流式交互函数
# ============================================================

def chat_fn_stream(user_message, chat_history, workspace_id):
    """
    流式输出版本：前面步骤一次性执行完，答案逐字输出。
    Gradio generator 模式。
    """
    if not user_message.strip():
        yield chat_history, "", "", "", "", "", "", "", "", workspace_id
        return

    ag = get_agent()

    if not workspace_id:
        workspace_id = ag.create_workspace("gradio_user")

    # 添加用户消息
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})

    # 先添加一个空的 assistant 消息占位
    chat_history.append({"role": "assistant", "content": "思考中..."})

    # 先 yield 一次，让用户看到"思考中"
    yield chat_history, "处理中...", "", "", "", "", "", "", "", workspace_id

    t0 = time.time()
    full_answer = ""
    result = None

    for token, is_final, final_result in ag.ask_stream(workspace_id=workspace_id, query=user_message):
        if is_final:
            result = final_result
            if token:  # 闲聊一次性返回
                full_answer = token
                chat_history[-1]["content"] = full_answer
        else:
            full_answer += token
            chat_history[-1]["content"] = full_answer

        # 流式 yield（每个 token 都更新界面）
        if not is_final:
            yield chat_history, "", "", "", "", "", "", "", "", workspace_id

    elapsed = time.time() - t0

    if result is None:
        result = {}

    # ---- 组装指标 ----
    intent = result.get("intent", "")
    rewritten = result.get("rewritten_query", "")
    if rewritten == user_message:
        rewrite_display = "未改写（问题已完整）"
    else:
        rewrite_display = rewritten or "未改写"

    # Wiki 命中
    wiki_hit_str = "未命中"
    if result.get("wiki_found"):
        wiki_hit_str = f"命中 {result.get('wiki_modules_hit', 0)} 模块 / {result.get('wiki_count', 0)} 条wiki"

    # RAG 命中
    if result.get("intent") == "chitchat":
        rag_hit_str = "未调用（闲聊）"
    elif result.get("rag_found"):
        rag_hit_str = f"命中 {result.get('rag_chunks_count', 0)} 条"
    else:
        rag_hit_str = "未命中"

    # 归属模块
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

    # 最终 yield（带完整指标）
    yield (
        chat_history, intent, rewrite_display, wiki_hit_str,
        rag_hit_str, module_display, time_display, steps_display,
        modules_md, workspace_id,
    )


def new_workspace_fn():
    """创建新 workspace，清空对话"""
    ag = get_agent()
    ws_id = ag.create_workspace(f"gradio_{int(time.time())}")
    return [], ws_id, "", "", "", "", "", "", ""


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
    with gr.Blocks(title="ChatWiki") as app:
        gr.Markdown("# ChatWiki - 对话级记忆检索 Agent")
        gr.Markdown("基于 LangGraph | 话题自动分组 | Wiki + RAG 混合检索 | 流式输出")

        workspace_state = gr.State(value="")

        with gr.Row():
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(label="对话", height=520)
                with gr.Row():
                    msg_input = gr.Textbox(
                        placeholder="输入问题后按 Enter 发送...",
                        scale=8, show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                with gr.Row():
                    new_ws_btn = gr.Button("新建对话", variant="secondary", size="sm")

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

        outputs = [
            chatbot, intent_display, rewrite_display, wiki_display,
            rag_display, module_display, time_display, steps_display,
            modules_display, workspace_state,
        ]

        msg_input.submit(
            fn=chat_fn_stream,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=outputs,
        ).then(fn=lambda: "", outputs=msg_input)

        send_btn.click(
            fn=chat_fn_stream,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=outputs,
        ).then(fn=lambda: "", outputs=msg_input)

        new_ws_btn.click(
            fn=new_workspace_fn,
            outputs=[chatbot, workspace_state, intent_display, rewrite_display,
                     wiki_display, rag_display, module_display, time_display,
                     steps_display, modules_display],
        )

    return app


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
