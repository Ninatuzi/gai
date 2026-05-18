"""
Milvus 客户端封装
Collection: wiki_modules — 存储 module 的 summary 向量，用于检索相关话题模块
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
        """初始化 wiki_modules collection"""
        self.connect()
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

    def delete_module(self, module_id: str) -> None:
        col = self._col()
        col.delete(f'id == "{module_id}"')
        col.flush()
