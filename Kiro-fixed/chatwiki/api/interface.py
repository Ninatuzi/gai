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
