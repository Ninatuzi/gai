"""
ChatWiki Gradio 可视化界面

功能：
- 左侧：对话区（Chatbot）
- 右侧：监测指标面板（意图/改写/Wiki命中/RAG命中/模块归属/耗时/步骤链路/话题模块列表）

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
# 核心交互函数
# ============================================================

def chat_fn(
    user_message: str,
    chat_history: List[Dict[str, str]],
    workspace_id: str,
):
    """
    用户发送消息后的处理函数。
    返回：更新后的 chat_history, 指标面板各字段, workspace_id
    """
    if not user_message.strip():
        return chat_history, "", "", "", "", "", "", "", "", workspace_id

    ag = get_agent()

    # 如果没有 workspace，自动创建
    if not workspace_id:
        workspace_id = ag.create_workspace("gradio_user")

    # 计时
    t0 = time.time()
    result = ag.ask(workspace_id=workspace_id, query=user_message)
    elapsed = time.time() - t0

    answer = result.get("answer", "（无回答）")

    # 更新对话历史
    chat_history = chat_history or []
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})

    # ---- 组装指标 ----
    intent = result.get("intent", "")
    rewritten = result.get("rewritten_query", "")
    # 如果改写后和原问题一样，显示"未改写"
    if rewritten == user_message:
        rewrite_display = "未改写（问题已完整）"
    else:
        rewrite_display = rewritten or "未改写"

    # Wiki 命中
    wiki_hit_str = "未命中"
    if result.get("wiki_found"):
        wiki_hit_str = f"命中 {result.get('wiki_modules_hit', 0)} 模块 / {result.get('wiki_count', 0)} 条wiki"

    # RAG 命中
    rag_hit_str = "未调用（Wiki已命中）"
    if result.get("intent") == "chitchat":
        rag_hit_str = "未调用（闲聊）"
    elif not result.get("wiki_found"):
        if result.get("rag_found"):
            rag_hit_str = f"命中 {result.get('rag_chunks_count', 0)} 条"
        else:
            rag_hit_str = "已调用，未命中"

    # 归属模块
    module_id = result.get("module_id", "")
    module_display = "无（闲聊）"
    if module_id:
        # 查询模块名称
        modules = ag.get_modules(workspace_id)
        for m in modules:
            if m["module_id"] == module_id:
                module_display = f"{m['topic']} ({module_id[:8]}...)"
                break
        else:
            module_display = f"{module_id[:8]}..."

    # 耗时
    time_display = f"{elapsed:.1f}s"

    # 执行步骤
    steps = result.get("steps", [])
    steps_display = " → ".join(steps) if steps else "无"

    # 话题模块列表
    modules_md = _format_modules_markdown(ag, workspace_id)

    return (
        chat_history,
        intent,
        rewrite_display,
        wiki_hit_str,
        rag_hit_str,
        module_display,
        time_display,
        steps_display,
        modules_md,
        workspace_id,
    )


def new_workspace_fn():
    """创建新 workspace，清空对话"""
    ag = get_agent()
    ws_id = ag.create_workspace(f"gradio_{int(time.time())}")
    return [], ws_id, "", "", "", "", "", "", "暂无话题模块"


def _format_modules_markdown(ag: ChatWikiAgent, workspace_id: str) -> str:
    """格式化话题模块列表为 Markdown"""
    modules = ag.get_modules(workspace_id)
    if not modules:
        return "暂无话题模块"

    lines = []
    for i, m in enumerate(modules, 1):
        topic = m.get("topic", "未命名")
        wiki_count = m.get("wiki_count", 0)
        summary = m.get("summary", "")
        lines.append(f"**{i}. {topic}** （{wiki_count}条wiki）")
        if summary:
            lines.append(f"   {summary[:80]}")
        lines.append("")
    return "\n".join(lines)


# ============================================================
# Gradio 界面构建
# ============================================================

def build_app() -> gr.Blocks:
    with gr.Blocks(title="ChatWiki - 对话记忆检索 Agent") as app:
        # 标题
        gr.Markdown("# ChatWiki - 对话级记忆检索 Agent")
        gr.Markdown("基于 LangGraph 的多轮对话知识管理系统 | 话题自动分组 | Wiki + RAG 混合检索")

        # 隐藏的 workspace_id state
        workspace_state = gr.State(value="")

        with gr.Row():
            # ========== 左侧：对话区 ==========
            with gr.Column(scale=7):
                chatbot = gr.Chatbot(
                    label="对话",
                    height=520,
                    show_copy_button=True,
                )
                with gr.Row():
                    msg_input = gr.Textbox(
                        label="输入问题",
                        placeholder="输入问题后按 Enter 发送...",
                        scale=8,
                        show_label=False,
                    )
                    send_btn = gr.Button("发送", variant="primary", scale=1)
                with gr.Row():
                    new_ws_btn = gr.Button("新建对话", variant="secondary", size="sm")
                    gr.Markdown("*提示：新建对话会创建新的 workspace，之前的记忆不会丢失*")

            # ========== 右侧：监测面板 ==========
            with gr.Column(scale=3):
                gr.Markdown("### 当前轮监测指标")

                with gr.Group():
                    intent_display = gr.Textbox(
                        label="意图识别",
                        value="",
                        interactive=False,
                        max_lines=1,
                    )
                    rewrite_display = gr.Textbox(
                        label="问题改写/增强",
                        value="",
                        interactive=False,
                        max_lines=2,
                    )

                with gr.Row():
                    wiki_display = gr.Textbox(
                        label="Wiki 命中",
                        value="",
                        interactive=False,
                        max_lines=1,
                    )
                    rag_display = gr.Textbox(
                        label="RAG 命中",
                        value="",
                        interactive=False,
                        max_lines=1,
                    )

                with gr.Row():
                    module_display = gr.Textbox(
                        label="归属模块",
                        value="",
                        interactive=False,
                        max_lines=1,
                    )
                    time_display = gr.Textbox(
                        label="耗时",
                        value="",
                        interactive=False,
                        max_lines=1,
                    )

                steps_display = gr.Textbox(
                    label="执行步骤链路",
                    value="",
                    interactive=False,
                    max_lines=3,
                )

                gr.Markdown("### 话题模块列表")
                modules_display = gr.Markdown(value="暂无话题模块")

        # ========== 事件绑定 ==========
        # 输出列表（跟 chat_fn 返回值顺序一致）
        outputs = [
            chatbot,
            intent_display,
            rewrite_display,
            wiki_display,
            rag_display,
            module_display,
            time_display,
            steps_display,
            modules_display,
            workspace_state,
        ]

        # Enter 发送
        msg_input.submit(
            fn=chat_fn,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=outputs,
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )

        # 点击发送按钮
        send_btn.click(
            fn=chat_fn,
            inputs=[msg_input, chatbot, workspace_state],
            outputs=outputs,
        ).then(
            fn=lambda: "",
            outputs=msg_input,
        )

        # 新建对话
        new_ws_btn.click(
            fn=new_workspace_fn,
            outputs=[
                chatbot,
                workspace_state,
                intent_display,
                rewrite_display,
                wiki_display,
                rag_display,
                module_display,
                time_display,
                steps_display,
                modules_display,
            ],
        )

    return app


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
