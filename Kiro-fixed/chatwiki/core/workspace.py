"""
Workspace 管理
"""

import logging
from typing import Any, Dict, List, Optional

from chatwiki.storage import MilvusClient, MySQLClient

logger = logging.getLogger(__name__)


class WorkspaceManager:
    def __init__(self, mysql: MySQLClient, milvus: MilvusClient):
        self.mysql = mysql
        self.milvus = milvus

    def create(self, chat_id: str) -> str:
        """创建 workspace（已存在则返回现有 id）"""
        existing = self.mysql.get_workspace_by_chat(chat_id)
        if existing:
            return existing["workspace_id"]
        return self.mysql.create_workspace(chat_id)

    def get_by_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        return self.mysql.get_workspace_by_chat(chat_id)

    def list_all(self) -> List[Dict[str, Any]]:
        return self.mysql.list_workspaces()

    def delete(self, workspace_id: str) -> None:
        """删除 workspace（级联删 module + wiki + Milvus 向量）"""
        try:
            self.milvus.delete_by_workspace(workspace_id)
        except Exception as e:
            logger.warning("删除 Milvus 向量失败: %s", e)
        self.mysql.delete_workspace(workspace_id)
        logger.info("workspace 已删除: %s", workspace_id)
