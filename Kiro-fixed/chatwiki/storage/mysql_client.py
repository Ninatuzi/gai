"""
MySQL 客户端封装
新表结构：workspace / wiki_record / wiki_module
"""

import json
import logging
import uuid
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import mysql.connector
from mysql.connector import pooling

from chatwiki.config import get_settings

logger = logging.getLogger(__name__)


SCHEMA_STATEMENTS = [
    # 工作空间表
    """
    CREATE TABLE IF NOT EXISTS workspace (
        workspace_id VARCHAR(36) PRIMARY KEY,
        chat_id VARCHAR(64) NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        status VARCHAR(16) DEFAULT 'active',
        UNIQUE KEY uk_chat_id (chat_id),
        INDEX idx_status (status)
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    # 话题模块表
    """
    CREATE TABLE IF NOT EXISTS wiki_module (
        module_id VARCHAR(36) PRIMARY KEY,
        workspace_id VARCHAR(36) NOT NULL,
        topic VARCHAR(256) NOT NULL,
        summary TEXT,
        wiki_count INT DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
        INDEX idx_workspace (workspace_id),
        CONSTRAINT fk_module_workspace FOREIGN KEY (workspace_id)
            REFERENCES workspace(workspace_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    # Wiki 记录表（每轮对话一条）
    """
    CREATE TABLE IF NOT EXISTS wiki_record (
        wiki_id VARCHAR(36) PRIMARY KEY,
        workspace_id VARCHAR(36) NOT NULL,
        module_id VARCHAR(36) NOT NULL,
        turn_number INT NOT NULL,
        query TEXT NOT NULL,
        answer TEXT NOT NULL,
        knowledge JSON,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_workspace (workspace_id),
        INDEX idx_module (module_id),
        CONSTRAINT fk_wiki_workspace FOREIGN KEY (workspace_id)
            REFERENCES workspace(workspace_id) ON DELETE CASCADE,
        CONSTRAINT fk_wiki_module FOREIGN KEY (module_id)
            REFERENCES wiki_module(module_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
    # 短期对话日志（记录所有轮次含闲聊，用于最近N轮上下文拼接）
    """
    CREATE TABLE IF NOT EXISTS chat_log (
        log_id VARCHAR(36) PRIMARY KEY,
        workspace_id VARCHAR(36) NOT NULL,
        turn_number INT NOT NULL,
        query TEXT NOT NULL,
        answer TEXT NOT NULL,
        intent VARCHAR(32) DEFAULT 'knowledge_query',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        INDEX idx_workspace_turn (workspace_id, turn_number),
        CONSTRAINT fk_log_workspace FOREIGN KEY (workspace_id)
            REFERENCES workspace(workspace_id) ON DELETE CASCADE
    ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci
    """,
]


class MySQLClient:
    def __init__(self, settings=None):
        cfg = (settings or get_settings()).mysql
        self.cfg = cfg
        self._pool: Optional[pooling.MySQLConnectionPool] = None

    def _get_pool(self) -> pooling.MySQLConnectionPool:
        if self._pool is None:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="chatwiki_pool",
                pool_size=5,
                host=self.cfg.host,
                port=self.cfg.port,
                user=self.cfg.user,
                password=self.cfg.password,
                database=self.cfg.database,
                charset=self.cfg.charset,
                use_pure=True,
                autocommit=False,
            )
        return self._pool

    @contextmanager
    def _cursor(self, dictionary: bool = True):
        conn = self._get_pool().get_connection()
        cur = conn.cursor(dictionary=dictionary)
        try:
            yield cur, conn
        finally:
            cur.close()
            conn.close()

    def init_schema(self) -> None:
        """创建数据库 + 建表"""
        conn = mysql.connector.connect(
            host=self.cfg.host, port=self.cfg.port,
            user=self.cfg.user, password=self.cfg.password, charset=self.cfg.charset,
        )
        try:
            cur = conn.cursor()
            cur.execute(
                f"CREATE DATABASE IF NOT EXISTS `{self.cfg.database}` "
                "CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci"
            )
            conn.commit()
            cur.close()
        finally:
            conn.close()

        with self._cursor(dictionary=False) as (cur, conn):
            for stmt in SCHEMA_STATEMENTS:
                cur.execute(stmt)
            conn.commit()
        logger.info("MySQL 表结构已就绪")

    def health_check(self) -> bool:
        try:
            with self._cursor(dictionary=False) as (cur, _):
                cur.execute("SELECT 1")
                return cur.fetchone()[0] == 1
        except Exception as e:
            logger.error("MySQL 健康检查失败: %s", e)
            return False

    # ============================================================
    # Workspace
    # ============================================================

    def create_workspace(self, chat_id: str) -> str:
        wid = str(uuid.uuid4())
        with self._cursor(dictionary=False) as (cur, conn):
            cur.execute(
                "INSERT INTO workspace (workspace_id, chat_id) VALUES (%s, %s)",
                (wid, chat_id),
            )
            conn.commit()
        return wid

    def get_workspace_by_chat(self, chat_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as (cur, _):
            cur.execute("SELECT * FROM workspace WHERE chat_id = %s", (chat_id,))
            return cur.fetchone()

    def list_workspaces(self) -> List[Dict[str, Any]]:
        with self._cursor() as (cur, _):
            cur.execute("SELECT * FROM workspace WHERE status = 'active' ORDER BY created_at DESC")
            return cur.fetchall()

    def delete_workspace(self, workspace_id: str) -> None:
        with self._cursor(dictionary=False) as (cur, conn):
            cur.execute("DELETE FROM workspace WHERE workspace_id = %s", (workspace_id,))
            conn.commit()

    # ============================================================
    # Wiki Module
    # ============================================================

    def create_module(self, workspace_id: str, topic: str, summary: str = "") -> str:
        mid = str(uuid.uuid4())
        with self._cursor(dictionary=False) as (cur, conn):
            cur.execute(
                "INSERT INTO wiki_module (module_id, workspace_id, topic, summary) "
                "VALUES (%s, %s, %s, %s)",
                (mid, workspace_id, topic, summary),
            )
            conn.commit()
        return mid

    def update_module(self, module_id: str, topic: Optional[str] = None,
                      summary: Optional[str] = None, wiki_count: Optional[int] = None) -> None:
        sets, vals = [], []
        if topic is not None:
            sets.append("topic = %s"); vals.append(topic)
        if summary is not None:
            sets.append("summary = %s"); vals.append(summary)
        if wiki_count is not None:
            sets.append("wiki_count = %s"); vals.append(wiki_count)
        if not sets:
            return
        vals.append(module_id)
        with self._cursor(dictionary=False) as (cur, conn):
            cur.execute(f"UPDATE wiki_module SET {', '.join(sets)} WHERE module_id = %s", tuple(vals))
            conn.commit()

    def get_module(self, module_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as (cur, _):
            cur.execute("SELECT * FROM wiki_module WHERE module_id = %s", (module_id,))
            return cur.fetchone()

    def get_modules_by_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as (cur, _):
            cur.execute(
                "SELECT * FROM wiki_module WHERE workspace_id = %s ORDER BY created_at DESC",
                (workspace_id,),
            )
            return cur.fetchall()

    def get_latest_module(self, workspace_id: str) -> Optional[Dict[str, Any]]:
        with self._cursor() as (cur, _):
            cur.execute(
                "SELECT * FROM wiki_module WHERE workspace_id = %s ORDER BY created_at DESC LIMIT 1",
                (workspace_id,),
            )
            return cur.fetchone()

    # ============================================================
    # Wiki Record
    # ============================================================

    def create_wiki(self, workspace_id: str, module_id: str, turn_number: int,
                    query: str, answer: str, knowledge: List[str]) -> str:
        wid = str(uuid.uuid4())
        with self._cursor(dictionary=False) as (cur, conn):
            cur.execute(
                "INSERT INTO wiki_record (wiki_id, workspace_id, module_id, turn_number, "
                "query, answer, knowledge) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                (wid, workspace_id, module_id, turn_number, query, answer,
                 json.dumps(knowledge, ensure_ascii=False)),
            )
            conn.commit()
        return wid

    def get_wikis_by_module(self, module_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as (cur, _):
            cur.execute(
                "SELECT * FROM wiki_record WHERE module_id = %s ORDER BY turn_number",
                (module_id,),
            )
            rows = cur.fetchall()
        for r in rows:
            r["knowledge"] = self._parse_json(r.get("knowledge"), [])
        return rows

    def get_wikis_by_workspace(self, workspace_id: str) -> List[Dict[str, Any]]:
        with self._cursor() as (cur, _):
            cur.execute(
                "SELECT * FROM wiki_record WHERE workspace_id = %s ORDER BY turn_number",
                (workspace_id,),
            )
            rows = cur.fetchall()
        for r in rows:
            r["knowledge"] = self._parse_json(r.get("knowledge"), [])
        return rows

    def get_wiki_count_in_module(self, module_id: str) -> int:
        with self._cursor(dictionary=False) as (cur, _):
            cur.execute("SELECT COUNT(*) FROM wiki_record WHERE module_id = %s", (module_id,))
            return cur.fetchone()[0]

    def get_turn_count(self, workspace_id: str) -> int:
        with self._cursor(dictionary=False) as (cur, _):
            cur.execute("SELECT COUNT(*) FROM wiki_record WHERE workspace_id = %s", (workspace_id,))
            return cur.fetchone()[0]

    # ============================================================
    # Chat Log（短期对话日志，含闲聊）
    # ============================================================

    def log_chat(self, workspace_id: str, turn_number: int, query: str,
                 answer: str, intent: str = "knowledge_query") -> str:
        log_id = str(uuid.uuid4())
        with self._cursor(dictionary=False) as (cur, conn):
            cur.execute(
                "INSERT INTO chat_log (log_id, workspace_id, turn_number, query, answer, intent) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (log_id, workspace_id, turn_number, query, answer, intent),
            )
            conn.commit()
        return log_id

    def get_recent_chat_log(self, workspace_id: str, n: int = 5) -> List[Dict[str, Any]]:
        """获取最近 N 轮对话（含闲聊），用于问题改写上下文"""
        with self._cursor() as (cur, _):
            cur.execute(
                "SELECT * FROM chat_log WHERE workspace_id = %s "
                "ORDER BY turn_number DESC LIMIT %s",
                (workspace_id, n),
            )
            rows = cur.fetchall()
        rows.reverse()  # 按时间正序
        return rows

    def get_chat_log_count(self, workspace_id: str) -> int:
        with self._cursor(dictionary=False) as (cur, _):
            cur.execute("SELECT COUNT(*) FROM chat_log WHERE workspace_id = %s", (workspace_id,))
            return cur.fetchone()[0]

    @staticmethod
    def _parse_json(value: Any, default: Any = None) -> Any:
        if value is None:
            return default
        if isinstance(value, (list, dict)):
            return value
        if isinstance(value, (bytes, bytearray)):
            value = value.decode("utf-8")
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return default
