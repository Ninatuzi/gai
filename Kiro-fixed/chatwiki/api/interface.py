"""
ChatWiki Agent 统一对外接口
插入即用：from chatwiki import ChatWikiAgent
"""

import logging
from typing import Any, Dict, List, Optional

from chatwiki.agent.graph import ChatWikiState, build_graph
from chatwiki.config import Settings, get_settings, reset_settings
from chatwiki.core.workspace import WorkspaceManager
from chatwiki.models import EmbeddingClient, LLMClient
from chatwiki.skills.aggregator_tool import AggregatorSkill
from chatwiki.skills.fallback_tool import FallbackSkill
from chatwiki.skills.intent_tool import IntentSkill
from chatwiki.skills.query_rewrite_tool import QueryRewriteSkill
from chatwiki.skills.rag_tool import RAGSkill
from chatwiki.skills.wiki_memory_tool import WikiMemorySkill
from chatwiki.storage import MilvusClient, MySQLClient

logger = logging.getLogger(__name__)


class ChatWikiAgent:
    """
    ChatWiki Agent 对外统一接口

    使用示例:
        agent = ChatWikiAgent()
        agent.init()

        ws_id = agent.create_workspace(chat_id="user_001")
        result = agent.ask(workspace_id=ws_id, query="析锂是什么？")
        print(result["answer"])
        print(result["steps"])
    """

    def __init__(self, settings: Optional[Settings] = None):
        if settings is not None:
            reset_settings(settings)
        self.settings = get_settings()

        # 基础客户端
        self.mysql = MySQLClient(self.settings)
        self.milvus = MilvusClient(self.settings)
        self.llm = LLMClient(self.settings)
        self.embedding = EmbeddingClient(self.settings)

        # Workspace 管理
        self.workspace_mgr = WorkspaceManager(self.mysql, self.milvus)

        # 创建 Skills
        self.intent_skill = IntentSkill(self.llm)
        self.rewrite_skill = QueryRewriteSkill(self.llm)
        self.wiki_skill = WikiMemorySkill(self.mysql, self.milvus, self.llm, self.embedding)
        self.rag_skill = RAGSkill()
        self.aggregator_skill = AggregatorSkill(self.llm)
        self.fallback_skill = FallbackSkill(self.llm)

        # 组装 skills 字典，传给 LangGraph 图构建
        self._skills = {
            "intent": self.intent_skill,
            "query_rewrite": self.rewrite_skill,
            "history_memory": self.wiki_skill,
            "rag": self.rag_skill,
            "aggregator": self.aggregator_skill,
            "fallback": self.fallback_skill,
        }

        # 编译 LangGraph 图
        self.graph = build_graph(self._skills)

    # ============================================================
    # 初始化 & 健康检查
    # ============================================================

    def init(self) -> None:
        """首次使用：建库建表 + 创建 Milvus collection"""
        logger.info("初始化 ChatWiki Agent...")
        self.mysql.init_schema()
        self.milvus.init_collections(dimension=self.embedding.dimension)
        logger.info("ChatWiki Agent 初始化完成")

    def health_check(self) -> Dict[str, bool]:
        return {
            "mysql": self.mysql.health_check(),
            "milvus": self.milvus.health_check(),
            "llm": self.llm.health_check(),
            "embedding": self.embedding.health_check(),
        }

    # ============================================================
    # Workspace 管理
    # ============================================================

    def create_workspace(self, chat_id: str) -> str:
        return self.workspace_mgr.create(chat_id)

    def delete_workspace(self, workspace_id: str) -> None:
        self.workspace_mgr.delete(workspace_id)

    def list_workspaces(self) -> List[Dict[str, Any]]:
        return self.workspace_mgr.list_all()

    # ============================================================
    # 核心方法
    # ============================================================

    def ask(self, workspace_id: str, query: str) -> ChatWikiState:
        """
        完整 Agent 问答：意图→改写→Wiki→RAG→聚合→写入

        Returns:
            ChatWikiState dict，包含 answer、steps 等所有中间状态
        """
        return self.graph.invoke({
            "workspace_id": workspace_id,
            "query": query,
            "steps": [],
        })

    def ask_stream(self, workspace_id: str, query: str):
        """
        流式版本：手动执行前面步骤，聚合阶段流式输出 token。
        避免跑完整 graph（会多一次非流式聚合+写入）。
        
        Yields: (token: str, is_final: bool, result: dict|None)
            - 流式输出时: (token, False, None)
            - 最后一次: ("", True, full_result_dict)
        """
        from chatwiki.skills.base import SkillInput
        from chatwiki.agent.graph import make_nodes

        # 手动构建节点函数
        nodes = make_nodes(self._skills)
        
        # 初始状态
        state = {
            "workspace_id": workspace_id,
            "query": query,
            "steps": [],
        }

        # Step 1: 意图识别
        state.update(nodes["node_intent"](state))

        # 如果是闲聊，走 fallback 直接返回
        if state.get("intent") == "chitchat":
            state.update(nodes["node_fallback"](state))
            nodes["node_log_chat"](state)
            yield state.get("answer", ""), True, state
            return

        # Step 2: 问题改写
        state.update(nodes["node_query_rewrite"](state))

        # Step 3: Wiki 检索
        state.update(nodes["node_wiki_search"](state))

        # Step 4: RAG 检索
        state.update(nodes["node_rag"](state))

        # Step 5: 流式聚合（只执行一次）
        inp = SkillInput(
            query=query,
            workspace_id=workspace_id,
            context={
                "wiki_context": state.get("wiki_context", ""),
                "rag_context": state.get("rag_context", ""),
            },
        )

        full_answer = ""
        for token in self.aggregator_skill.run_stream(inp):
            full_answer += token
            yield token, False, None

        state["answer"] = full_answer
        steps = list(state.get("steps") or [])
        steps.append("aggregation")
        state["steps"] = steps

        # Step 6: 写入 Wiki
        rag_chunks = state.get("rag_chunks") or []
        knowledge = [c.get("content", "") for c in rag_chunks if c.get("content")][:5] if state.get("rag_found") else []

        write_result = self.wiki_skill.write(
            workspace_id=workspace_id,
            query=query,
            answer=full_answer,
            knowledge=knowledge,
            intent=state.get("intent", "knowledge_query"),
            match_query=state.get("rewritten_query"),
        )
        state["wiki_id"] = write_result.get("wiki_id")
        state["module_id"] = write_result.get("module_id")
        state["turn_number"] = write_result.get("turn_number", 0)
        steps.append(f"wiki_write:module={write_result.get('module_id', '')[:8]}")
        state["steps"] = steps

        # Step 7: 记录 chat_log
        self.wiki_skill.log_chat(
            workspace_id=workspace_id,
            query=query,
            answer=full_answer,
            intent=state.get("intent", ""),
        )

        yield "", True, state

    # ============================================================
    # 数据查看
    # ============================================================

    def get_modules(self, workspace_id: str) -> List[Dict[str, Any]]:
        return self.wiki_skill.get_modules(workspace_id)

    def get_wikis(self, workspace_id: str) -> List[Dict[str, Any]]:
        return self.wiki_skill.get_all_wikis(workspace_id)

    def get_wikis_by_module(self, module_id: str) -> List[Dict[str, Any]]:
        return self.wiki_skill.get_wikis_by_module(module_id)

    def get_registered_skills(self) -> List[str]:
        """查看已注册的所有 Skill"""
        return list(self._skills.keys())
