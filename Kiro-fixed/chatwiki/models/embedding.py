"""
BGE-M3 Embedding 客户端封装（复用）
"""

import logging
from typing import List

from openai import OpenAI

from chatwiki.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingClient:
    def __init__(self, settings=None):
        cfg = (settings or get_settings()).embedding
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key)

    @property
    def dimension(self) -> int:
        return self.cfg.dimension

    def embed(self, text: str) -> List[float]:
        if not text or not text.strip():
            text = " "
        resp = self._client.embeddings.create(input=text, model=self.cfg.model_name)
        return resp.data[0].embedding

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        cleaned = [t if t and t.strip() else " " for t in texts]
        resp = self._client.embeddings.create(input=cleaned, model=self.cfg.model_name)
        return [d.embedding for d in sorted(resp.data, key=lambda x: x.index)]

    def health_check(self) -> bool:
        try:
            vec = self.embed("测试")
            return len(vec) == self.cfg.dimension
        except Exception as e:
            logger.error("Embedding健康检查失败: %s", e)
            return False
