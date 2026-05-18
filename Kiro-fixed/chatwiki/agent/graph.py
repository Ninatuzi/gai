"""
ChatWiki Agent 图（LangGraph 实现）
负责：定义节点、条件边、编译 StateGraph

流程：
    用户提问
        │
        ▼
    [node_intent] 意图识别
        │
        ├── chitchat ──→ [node_fallback] 保底直答 → [node_log_chat] → END
        │
        ▼ knowledge_query / followup_query
    [node_query_rewrite] 问题改写
        │
        ▼
    [node_wiki_search] Wiki 检索
        │
        ├── wiki 命中 ──────────────────────┐
        │                                   ▼
        └── wiki 未命中 → [node_rag] → [node_aggregate] 答案聚合
                                            │
                                            ▼
                                    [node_wiki_write] 写入 Wiki
                                            │
                                            ▼
                                    [node_log_chat] 记录 chat_log
                                            │
                                           END
"""

import logging
from typing import Any, Dict, List, Optional

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END, START

from chatwiki.skills.base import SkillInput

logger = logging.getLogger(__name__)


# ============================================================
# State 定义（LangGraph 要求 TypedDict）
# ============================================================

class ChatWikiState(TypedDict, total=False):
    # 输入
    workspace_id: str
    query: str

    # 各节点产出
    intent: str                        # knowledge_query / followup_query / chitchat
    rewritten_query: str               # 改写后的问题
    wiki_context: str                  # Wiki 检索的格式化文本
    wiki_found: bool                   # Wiki 是否命中
    wiki_modules_hit: int              # 命中的模块数
    wiki_count: int                    # 命中的 wiki 条数
    rag_context: str                   # RAG 检索的格式化文本
    rag_found: bool                    # RAG 是否命中
    rag_chunks: List[Dict[str, Any]]   # RAG 原始 chunks
    rag_chunks_count: int              # RAG 命中条数
    answer: str                        # 最终回答

    # 写入结果
    wiki_id: Optional[str]
    module_id: Optional[str]
    turn_number: int

    # 执行链路记录
    steps: List[str]


# ============================================================
# 节点工厂：把各 Skill 包装成 LangGraph 节点函数
# ============================================================

def make_nodes(skills: Dict[str, Any]) -> Dict[str, Any]:
    """
    根据已注册的 skills 字典，生成所有节点函数。
    节点函数签名：(state: ChatWikiState) -> Dict，返回要更新的字段。
    """

    def _skill(name):
        return skills.get(name)

    # ----------------------------------------------------------
    # node_intent：意图识别
    # ----------------------------------------------------------
    def node_intent(state: ChatWikiState) -> Dict:
        skill = _skill("intent")
        steps = list(state.get("steps") or [])

        if not skill:
            steps.append("intent:knowledge_query(default)")
            return {"intent": "knowledge_query", "steps": steps}

        wiki_skill = _skill("history_memory")
        recent_modules = []
        recent_chat = ""
        if wiki_skill:
            modules = wiki_skill.get_modules(state["workspace_id"])
            recent_modules = [m["topic"] for m in modules[:5]]
            # 提供最近 5 轮对话帮助意图判断（区分闲聊追问 vs 知识追问）
            recent_chat = wiki_skill.get_recent_context(state["workspace_id"], n=5)

        inp = SkillInput(
            query=state["query"],
            workspace_id=state["workspace_id"],
            context={"recent_modules": recent_modules, "recent_chat": recent_chat},
        )
        out = skill.run(inp)
        intent = out.data.get("intent", "knowledge_query")
        steps.append(f"intent:{intent}")
        return {"intent": intent, "steps": steps}

    # ----------------------------------------------------------
    # node_fallback：保底直答（闲聊路径）
    # ----------------------------------------------------------
    def node_fallback(state: ChatWikiState) -> Dict:
        skill = _skill("fallback")
        steps = list(state.get("steps") or [])

        if not skill:
            steps.append("fallback:default")
            return {"answer": "你好！", "steps": steps}

        wiki_skill = _skill("history_memory")
        chat_history = ""
        if wiki_skill:
            chat_history = wiki_skill.get_recent_context(state["workspace_id"], n=15)

        inp = SkillInput(
            query=state["query"],
            workspace_id=state["workspace_id"],
            context={"chat_history": chat_history},
        )
        out = skill.run(inp)
        steps.append("fallback:direct_answer")
        return {"answer": out.data.get("answer", ""), "steps": steps}

    # ----------------------------------------------------------
    # node_query_rewrite：问题改写（指代消解）
    # ----------------------------------------------------------
    def node_query_rewrite(state: ChatWikiState) -> Dict:
        skill = _skill("query_rewrite")
        steps = list(state.get("steps") or [])

        if not skill:
            steps.append("query_rewrite:skip")
            return {"rewritten_query": state["query"], "steps": steps}

        wiki_skill = _skill("history_memory")
        recent_context = ""
        if wiki_skill:
            recent_context = wiki_skill.get_recent_context(state["workspace_id"], n=5)

        inp = SkillInput(
            query=state["query"],
            workspace_id=state["workspace_id"],
            context={
                "recent_context": recent_context,
                "intent": state.get("intent", "knowledge_query"),
            },
        )
        out = skill.run(inp)
        steps.append("query_rewrite")
        return {
            "rewritten_query": out.data.get("rewritten_query", state["query"]),
            "steps": steps,
        }

    # ----------------------------------------------------------
    # node_wiki_search：Wiki 历史记忆检索
    # ----------------------------------------------------------
    def node_wiki_search(state: ChatWikiState) -> Dict:
        skill = _skill("history_memory")
        steps = list(state.get("steps") or [])

        if not skill:
            steps.append("wiki_search:no_skill")
            return {
                "wiki_found": False, "wiki_context": "",
                "wiki_modules_hit": 0, "wiki_count": 0,
                "steps": steps,
            }

        search_query = state.get("rewritten_query") or state["query"]
        inp = SkillInput(query=search_query, workspace_id=state["workspace_id"])
        out = skill.run(inp)

        found = out.data.get("found", False)
        modules_hit = out.data.get("modules_hit", 0)
        wiki_count = len(out.data.get("wikis", []))
        steps.append(f"wiki_search:found={found},modules={modules_hit},wikis={wiki_count}")
        return {
            "wiki_found": found,
            "wiki_context": out.data.get("context_text", ""),
            "wiki_modules_hit": modules_hit,
            "wiki_count": wiki_count,
            "steps": steps,
        }

    # ----------------------------------------------------------
    # node_rag：RAG 检索
    # ----------------------------------------------------------
    def node_rag(state: ChatWikiState) -> Dict:
        skill = _skill("rag")
        steps = list(state.get("steps") or [])

        if not skill:
            steps.append("rag_search:no_skill")
            return {
                "rag_found": False, "rag_context": "",
                "rag_chunks": [], "rag_chunks_count": 0,
                "steps": steps,
            }

        search_query = state.get("rewritten_query") or state["query"]
        inp = SkillInput(query=search_query, workspace_id=state["workspace_id"])
        out = skill.run(inp)

        chunks = out.data.get("chunks", [])
        found = out.data.get("found", False)
        steps.append(f"rag_search:found={found},chunks={len(chunks)}")
        return {
            "rag_found": found,
            "rag_context": out.data.get("context_text", ""),
            "rag_chunks": chunks,
            "rag_chunks_count": len(chunks),
            "steps": steps,
        }

    # ----------------------------------------------------------
    # node_aggregate：答案聚合（Wiki + RAG → 最终回答）
    # ----------------------------------------------------------
    def node_aggregate(state: ChatWikiState) -> Dict:
        skill = _skill("aggregator")
        steps = list(state.get("steps") or [])

        if not skill:
            steps.append("aggregation:no_skill")
            return {"answer": "抱歉，聚合模块未注册。", "steps": steps}

        inp = SkillInput(
            query=state["query"],
            workspace_id=state["workspace_id"],
            context={
                "wiki_context": state.get("wiki_context", ""),
                "rag_context": state.get("rag_context", ""),
            },
        )
        out = skill.run(inp)
        steps.append("aggregation")
        return {"answer": out.data.get("answer", ""), "steps": steps}

    # ----------------------------------------------------------
    # node_wiki_write：写入 Wiki
    # ----------------------------------------------------------
    def node_wiki_write(state: ChatWikiState) -> Dict:
        skill = _skill("history_memory")
        steps = list(state.get("steps") or [])

        if not skill:
            steps.append("wiki_write:no_skill")
            return {"steps": steps}

        rag_chunks = state.get("rag_chunks") or []
        knowledge: List[str] = []
        if state.get("rag_found") and rag_chunks:
            knowledge = [c.get("content", "") for c in rag_chunks if c.get("content")][:5]

        result = skill.write(
            workspace_id=state["workspace_id"],
            query=state["query"],
            answer=state.get("answer", ""),
            knowledge=knowledge,
            intent=state.get("intent", "knowledge_query"),
            match_query=state.get("rewritten_query"),
        )
        module_id = result.get("module_id")
        steps.append(f"wiki_write:module={module_id[:8] if module_id else 'N/A'}")
        return {
            "wiki_id": result.get("wiki_id"),
            "module_id": module_id,
            "turn_number": result.get("turn_number", 0),
            "steps": steps,
        }

    # ----------------------------------------------------------
    # node_log_chat：记录 chat_log（闲聊和知识问答共用）
    # ----------------------------------------------------------
    def node_log_chat(state: ChatWikiState) -> Dict:
        wiki_skill = _skill("history_memory")
        if not wiki_skill:
            return {}
        try:
            wiki_skill.log_chat(
                workspace_id=state["workspace_id"],
                query=state["query"],
                answer=state.get("answer", ""),
                intent=state.get("intent", ""),
            )
        except Exception as e:
            logger.warning("记录chat_log失败: %s", e)
        return {}

    return {
        "node_intent": node_intent,
        "node_fallback": node_fallback,
        "node_query_rewrite": node_query_rewrite,
        "node_wiki_search": node_wiki_search,
        "node_rag": node_rag,
        "node_aggregate": node_aggregate,
        "node_wiki_write": node_wiki_write,
        "node_log_chat": node_log_chat,
    }


# ============================================================
# 条件边路由函数
# ============================================================

def route_by_intent(state: ChatWikiState) -> str:
    """意图识别后路由：chitchat → 保底直答，其余 → 问题改写"""
    if state.get("intent") == "chitchat":
        return "chitchat"
    return "knowledge"


def route_by_wiki(state: ChatWikiState) -> str:
    """Wiki 检索后路由：命中 → 直接聚合，未命中 → RAG"""
    if state.get("wiki_found"):
        return "wiki_hit"
    return "need_rag"


# ============================================================
# 图构建入口
# ============================================================

def build_graph(skills: Dict[str, Any]):
    """
    根据已注册的 skills 构建并编译 LangGraph StateGraph。

    Args:
        skills: {skill_name: skill_instance} 字典，由 ChatWikiAgent 传入

    Returns:
        已编译的 CompiledGraph，调用方式：
            result: ChatWikiState = graph.invoke({
                "workspace_id": "xxx",
                "query": "析锂是什么？",
                "steps": [],
            })
    """
    nodes = make_nodes(skills)

    builder = StateGraph(ChatWikiState)

    # 注册所有节点
    for name, fn in nodes.items():
        builder.add_node(name, fn)

    # 入口节点
    builder.add_edge(START, "node_intent")

    # 意图路由
    builder.add_conditional_edges(
        "node_intent",
        route_by_intent,
        {
            "chitchat": "node_fallback",
            "knowledge": "node_query_rewrite",
        },
    )

    # 闲聊路径：fallback → log_chat → END
    builder.add_edge("node_fallback", "node_log_chat")

    # 知识问答路径：改写 → wiki 检索
    builder.add_edge("node_query_rewrite", "node_wiki_search")

    # Wiki 路由：命中 → 聚合，未命中 → RAG → 聚合
    builder.add_conditional_edges(
        "node_wiki_search",
        route_by_wiki,
        {
            "wiki_hit": "node_aggregate",
            "need_rag": "node_rag",
        },
    )
    builder.add_edge("node_rag", "node_aggregate")

    # 聚合 → 写入 wiki → log_chat → END
    builder.add_edge("node_aggregate", "node_wiki_write")
    builder.add_edge("node_wiki_write", "node_log_chat")

    # log_chat 是两条路径的汇合点，统一到 END
    builder.add_edge("node_log_chat", END)

    return builder.compile()
