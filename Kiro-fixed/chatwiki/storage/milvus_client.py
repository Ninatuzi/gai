"""
Milvus 客户端封装
Collection:
  - wiki_modules: 存储 module 的 summary 向量，用于第一级检索（找相关话题模块）
  - wiki_summaries: 存储每条 wiki 的 summary 向量，用于第二级检索（模块内找最相关 wiki）
"""

import logging
from typing import Any, Dict, List, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

from chatwiki.config import get_settings

logger = logging.getLogger(__name__)

COLLECTION_WIKI_MODULES = "wiki_modules"
COLLECTION_WIKI_SUMMARIES = "wiki_summaries"
_CONNECTION_ALIAS = "chatwiki_default"


class MilvusClient:
    def __init__(self, settings=None):
        cfg = (settings or get_settings()).milvus
        self.cfg = cfg
        self._connected = False

    def connect(self) -> None:
        if self._connected:
            return
        connections.connect(alias=_CONNECTION_ALIAS, host=self.cfg.host, port=str(self.cfg.port))
        self._connected = True
        logger.info("Milvus connected: %s:%s", self.cfg.host, self.cfg.port)

    def close(self) -> None:
        if self._connected:
            try:
                connections.disconnect(_CONNECTION_ALIAS)
            except Exception:
                pass
            self._connected = False

    def health_check(self) -> bool:
        try:
            self.connect()
            utility.list_collections(using=_CONNECTION_ALIAS)
            return True
        except Exception as e:
            logger.error("Milvus 健康检查失败: %s", e)
            return False

    def init_collections(self, dimension: int) -> None:
        """初始化所有 collection"""
        self.connect()
        self._init_modules_collection(dimension)
        self._init_summaries_collection(dimension)

    def _init_modules_collection(self, dimension: int) -> None:
        """初始化 wiki_modules collection（第一级：模块检索）"""
        if utility.has_collection(COLLECTION_WIKI_MODULES, using=_CONNECTION_ALIAS):
            col = Collection(COLLECTION_WIKI_MODULES, using=_CONNECTION_ALIAS)
            col.load()
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="workspace_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="topic", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=2048),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields=fields, description="Wiki Module summaries for retrieval")
        col = Collection(name=COLLECTION_WIKI_MODULES, schema=schema, using=_CONNECTION_ALIAS)
        col.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": self.cfg.hnsw_m, "efConstruction": self.cfg.hnsw_ef_construction},
            },
        )
        col.load()
        logger.info("创建 collection: %s (dim=%d)", COLLECTION_WIKI_MODULES, dimension)

    def _init_summaries_collection(self, dimension: int) -> None:
        """初始化 wiki_summaries collection（第二级：wiki 条目检索）"""
        if utility.has_collection(COLLECTION_WIKI_SUMMARIES, using=_CONNECTION_ALIAS):
            col = Collection(COLLECTION_WIKI_SUMMARIES, using=_CONNECTION_ALIAS)
            col.load()
            return
        fields = [
            FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
            FieldSchema(name="workspace_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="module_id", dtype=DataType.VARCHAR, max_length=64),
            FieldSchema(name="summary", dtype=DataType.VARCHAR, max_length=1024),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dimension),
        ]
        schema = CollectionSchema(fields=fields, description="Wiki entry summaries for fine-grained retrieval")
        col = Collection(name=COLLECTION_WIKI_SUMMARIES, schema=schema, using=_CONNECTION_ALIAS)
        col.create_index(
            field_name="embedding",
            index_params={
                "metric_type": "COSINE",
                "index_type": "HNSW",
                "params": {"M": self.cfg.hnsw_m, "efConstruction": self.cfg.hnsw_ef_construction},
            },
        )
        col.load()
        logger.info("创建 collection: %s (dim=%d)", COLLECTION_WIKI_SUMMARIES, dimension)

    # ============================================================
    # Module 操作（第一级）
    # ============================================================

    def _col(self) -> Collection:
        self.connect()
        return Collection(COLLECTION_WIKI_MODULES, using=_CONNECTION_ALIAS)

    def upsert_module(self, module_id: str, workspace_id: str, topic: str,
                      summary: str, embedding: List[float]) -> None:
        """插入或更新 module 向量"""
        col = self._col()
        try:
            col.upsert([
                [module_id],
                [workspace_id],
                [topic[:256]],
                [summary[:2048]],
                [embedding],
            ])
        except Exception:
            col.delete(f'id == "{module_id}"')
            col.insert([
                [module_id],
                [workspace_id],
                [topic[:256]],
                [summary[:2048]],
                [embedding],
            ])
        col.flush()

    def search_modules(self, workspace_id: str, query_embedding: List[float],
                       top_k: int = 5) -> List[Dict[str, Any]]:
        """在指定 workspace 内搜索相关 module"""
        col = self._col()
        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": self.cfg.hnsw_ef}},
            limit=top_k,
            expr=f'workspace_id == "{workspace_id}"',
            output_fields=["workspace_id", "topic", "summary"],
        )
        out: List[Dict[str, Any]] = []
        for hits in results:
            for h in hits:
                out.append({
                    "module_id": h.id,
                    "score": float(h.score),
                    "workspace_id": h.entity.get("workspace_id"),
                    "topic": h.entity.get("topic"),
                    "summary": h.entity.get("summary"),
                })
        return out

    def delete_by_workspace(self, workspace_id: str) -> None:
        """删除某 workspace 的所有 module 向量"""
        col = self._col()
        col.delete(f'workspace_id == "{workspace_id}"')
        col.flush()
        # 同时删除 wiki_summaries
        self._delete_wiki_summaries_by_workspace(workspace_id)

    def delete_module(self, module_id: str) -> None:
        col = self._col()
        col.delete(f'id == "{module_id}"')
        col.flush()

    # ============================================================
    # Wiki Summary 操作（第二级）
    # ============================================================

    def _col_summaries(self) -> Collection:
        self.connect()
        return Collection(COLLECTION_WIKI_SUMMARIES, using=_CONNECTION_ALIAS)

    def upsert_wiki_summary(self, wiki_id: str, workspace_id: str, module_id: str,
                            summary: str, embedding: List[float]) -> None:
        """插入或更新 wiki summary 向量"""
        col = self._col_summaries()
        try:
            col.upsert([
                [wiki_id],
                [workspace_id],
                [module_id],
                [summary[:1024]],
                [embedding],
            ])
        except Exception:
            col.delete(f'id == "{wiki_id}"')
            col.insert([
                [wiki_id],
                [workspace_id],
                [module_id],
                [summary[:1024]],
                [embedding],
            ])
        col.flush()

    def search_wiki_summaries(self, module_ids: List[str], query_embedding: List[float],
                              top_k: int = 5) -> List[Dict[str, Any]]:
        """
        在指定的 module(s) 内搜索最相关的 wiki summaries。
        支持多个 module_id 联合搜索。
        """
        col = self._col_summaries()
        # 构建 module_id in [...] 表达式
        ids_str = ", ".join(f'"{mid}"' for mid in module_ids)
        expr = f"module_id in [{ids_str}]"

        results = col.search(
            data=[query_embedding],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"ef": self.cfg.hnsw_ef}},
            limit=top_k,
            expr=expr,
            output_fields=["workspace_id", "module_id", "summary"],
        )
        out: List[Dict[str, Any]] = []
        for hits in results:
            for h in hits:
                out.append({
                    "wiki_id": h.id,
                    "score": float(h.score),
                    "workspace_id": h.entity.get("workspace_id"),
                    "module_id": h.entity.get("module_id"),
                    "summary": h.entity.get("summary"),
                })
        return out

    def delete_wiki_summary(self, wiki_id: str) -> None:
        col = self._col_summaries()
        col.delete(f'id == "{wiki_id}"')
        col.flush()

    def _delete_wiki_summaries_by_workspace(self, workspace_id: str) -> None:
        """删除某 workspace 下所有 wiki summary 向量"""
        col = self._col_summaries()
        col.delete(f'workspace_id == "{workspace_id}"')
        col.flush()
