"""
JSON 文件存储客户端
每个 workspace 一个 JSON 文件，存储 modules + wikis + chat_log
替代 MySQL，零依赖
"""

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from chatwiki.config import get_settings

logger = logging.getLogger(__name__)

# 默认数据目录
DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data")


class JSONClient:
    """
    JSON 文件存储客户端
    每个 workspace 对应一个文件: data/{workspace_id}.json
    """

    def __init__(self, settings=None, data_dir: Optional[str] = None):
        self.data_dir = data_dir or DEFAULT_DATA_DIR
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info("JSONClient 数据目录: %s", self.data_dir)

    # ============================================================
    # 文件读写
    # ============================================================

    def _get_path(self, workspace_id: str) -> str:
        return os.path.join(self.data_dir, f"{workspace_id}.json")

    def _load(self, workspace_id: str) -> Dict[str, Any]:
        """加载 workspace 的 JSON 数据"""
        path = self._get_path(workspace_id)
        if not os.path.exists(path):
            return {}
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save(self, workspace_id: str, data: Dict[str, Any]) -> None:
        """保存 workspace 的 JSON 数据"""
        path = self._get_path(workspace_id)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _new_workspace_data(self, workspace_id: str, chat_id: str) -> Dict[str, Any]:
        """创建空的 workspace 数据结构"""
        return {
            "workspace_id": workspace_id,
            "chat_id": chat_id,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "modules": [],
            "chat_log": [],
        }

    # ============================================================
    # 初始化 & 健康检查
    # ============================================================

    def init_schema(self) -> None:
        """确保数据目录存在"""
        os.makedirs(self.data_dir, exist_ok=True)
        logger.info("JSON 存储已就绪: %s", self.data_dir)

    def health_check(self) -> bool:
        try:
            test_path = os.path.join(self.data_dir, ".health_check")
            with open(test_path, "w") as f:
                f.write("ok")
            os.remove(test_path)
            return True
        except Exception as e:
            logger.error("JSON 存储健康检查失败: %s", e)
            return False

    # ============================================================
    # Workspace
    # ============================================================

    def create_workspace(self, chat_id: str) -> str:
        wid = str(uuid.uuid4())
        data = self._new_workspace_data(wid, chat_id)
        self._save(wid, data)
        return wid

    def ensure_workspace(self, workspace_id: str) -> str:
        """
        确保 workspace 存在：如果已存在则直接返回，不存在则以 workspace_id 为 ID 创建。
        适用于 Dify 直接传 conversation_id 作为 workspace_id 的场景。
        """
        path = self._get_path(workspace_id)
        if os.path.exists(path):
            return workspace_id
        # 不存在则创建，用 workspace_id 本身作为 chat_id
        data = self._new_workspace_data(workspace_id, chat_id=workspace_id)
        self._save(workspace_id, data)
        logger.info("自动创建 workspace: %s", workspace_id)
        return workspace_id

    def get_workspace_by_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        """遍历所有文件找到匹配的 chat_id"""
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".json"):
                continue
            wid = fname[:-5]
            data = self._load(wid)
            if data.get("chat_id") == chat_id and data.get("status") == "active":
                return {"workspace_id": wid, "chat_id": chat_id, "status": "active",
                        "created_at": data.get("created_at")}
        return None

    def list_workspaces(self) -> List[Dict[str, Any]]:
        result = []
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".json"):
                continue
            wid = fname[:-5]
            data = self._load(wid)
            if data.get("status") == "active":
                result.append({
                    "workspace_id": wid,
                    "chat_id": data.get("chat_id", ""),
                    "status": "active",
                    "created_at": data.get("created_at"),
                })
        return result

    def delete_workspace(self, workspace_id: str) -> None:
        path = self._get_path(workspace_id)
        if os.path.exists(path):
            os.remove(path)
            logger.info("删除 workspace 文件: %s", path)

    # ============================================================
    # Wiki Module
    # ============================================================

    def create_module(self, workspace_id: str, topic: str, summary: str = "") -> str:
        mid = str(uuid.uuid4())
        data = self._load(workspace_id)
        data["modules"].append({
            "module_id": mid,
            "topic": topic,
            "summary": summary,
            "wiki_count": 0,
            "created_at": datetime.now().isoformat(),
            "wikis": [],
        })
        self._save(workspace_id, data)
        return mid

    def update_module(self, module_id: str, topic: Optional[str] = None,
                      summary: Optional[str] = None, wiki_count: Optional[int] = None,
                      workspace_id: Optional[str] = None) -> None:
        """更新模块信息。需要知道 workspace_id 才能找到文件"""
        if not workspace_id:
            workspace_id = self._find_workspace_by_module(module_id)
        if not workspace_id:
            return
        data = self._load(workspace_id)
        for m in data["modules"]:
            if m["module_id"] == module_id:
                if topic is not None:
                    m["topic"] = topic
                if summary is not None:
                    m["summary"] = summary
                if wiki_count is not None:
                    m["wiki_count"] = wiki_count
                break
        self._save(workspace_id, data)

    def get_module(self, module_id: str) -> Optional[Dict[str, Any]]:
        """查找 module（需要遍历文件）"""
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".json"):
                continue
            data = self._load(fname[:-5])
            for m in data.get("modules", []):
                if m["module_id"] == module_id:
                    return {
                        "module_id": m["module_id"],
                        "workspace_id": data["workspace_id"],
                        "topic": m["topic"],
                        "summary": m.get("summary", ""),
                        "wiki_count": m.get("wiki_count", 0),
                    }
        return None

    def get_modules_by_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        data = self._load(workspace_id)
        return [
            {
                "module_id": m["module_id"],
                "workspace_id": workspace_id,
                "topic": m["topic"],
                "summary": m.get("summary", ""),
                "wiki_count": m.get("wiki_count", len(m.get("wikis", []))),
            }
            for m in data.get("modules", [])
        ]

    def get_latest_module(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        data = self._load(workspace_id)
        modules = data.get("modules", [])
        if not modules:
            return None
        m = modules[-1]
        return {
            "module_id": m["module_id"],
            "workspace_id": workspace_id,
            "topic": m["topic"],
            "summary": m.get("summary", ""),
            "wiki_count": m.get("wiki_count", len(m.get("wikis", []))),
        }

    # ============================================================
    # Wiki Record
    # ============================================================

    def create_wiki(self, workspace_id: str, module_id: str, turn_number: int,
                    query: str, answer: str, knowledge: List[str],
                    summary: str = "") -> str:
        wid = str(uuid.uuid4())
        data = self._load(workspace_id)
        for m in data["modules"]:
            if m["module_id"] == module_id:
                m["wikis"].append({
                    "wiki_id": wid,
                    "turn_number": turn_number,
                    "query": query,
                    "answer": answer,
                    "knowledge": knowledge,
                    "summary": summary,
                    "created_at": datetime.now().isoformat(),
                })
                m["wiki_count"] = len(m["wikis"])
                break
        self._save(workspace_id, data)
        return wid

    def get_wikis_by_module(self, module_id: str) -> List[Dict[str, Any]]:
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".json"):
                continue
            data = self._load(fname[:-5])
            for m in data.get("modules", []):
                if m["module_id"] == module_id:
                    return m.get("wikis", [])
        return []

    def get_wikis_by_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        data = self._load(workspace_id)
        all_wikis = []
        for m in data.get("modules", []):
            all_wikis.extend(m.get("wikis", []))
        all_wikis.sort(key=lambda w: w.get("turn_number", 0))
        return all_wikis

    def get_turn_count(self, workspace_id: str) -> int:
        data = self._load(workspace_id)
        count = 0
        for m in data.get("modules", []):
            count += len(m.get("wikis", []))
        return count

    # ============================================================
    # Chat Log
    # ============================================================

    def log_chat(self, workspace_id: str, turn_number: int, query: str,
                 answer: str, intent: str = "knowledge_query") -> str:
        log_id = str(uuid.uuid4())
        data = self._load(workspace_id)
        data["chat_log"].append({
            "log_id": log_id,
            "turn_number": turn_number,
            "query": query,
            "answer": answer,
            "intent": intent,
            "created_at": datetime.now().isoformat(),
        })
        self._save(workspace_id, data)
        return log_id

    def get_recent_chat_log(self, workspace_id: str, n: int = 15) -> List[Dict[str, Any]]:
        data = self._load(workspace_id)
        logs = data.get("chat_log", [])
        return logs[-n:]

    def get_chat_log_count(self, workspace_id: str) -> int:
        data = self._load(workspace_id)
        return len(data.get("chat_log", []))

    # ============================================================
    # 工具方法
    # ============================================================

    def _find_workspace_by_module(self, module_id: str) -> Optional[str]:
        """通过 module_id 找到所属的 workspace_id"""
        for fname in os.listdir(self.data_dir):
            if not fname.endswith(".json"):
                continue
            wid = fname[:-5]
            data = self._load(wid)
            for m in data.get("modules", []):
                if m["module_id"] == module_id:
                    return wid
        return None
