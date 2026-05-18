"""
全局配置模块
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LLMConfig:
    """DeepSeek LLM 配置"""
    base_url: str = "http://10.0.6.89:8080/v1"
    model_name: str = "DeepSeek_32B_f16"
    api_key: str = "EMPTY"
    temperature: float = 0.1
    max_tokens: int = 4096


@dataclass
class EmbeddingConfig:
    """BGE-M3 Embedding 配置"""
    base_url: str = "http://10.7.10.102:8001/v1"
    api_key: str = "qwertyuiop1A."
    model_name: str = "/root/embedding_model/bge-m3"
    dimension: int = 1024


@dataclass
class MilvusConfig:
    """Milvus 配置"""
    host: str = "10.7.10.102"
    port: int = 19530
    hnsw_m: int = 16
    hnsw_ef_construction: int = 256
    hnsw_ef: int = 128


@dataclass
class StorageConfig:
    """JSON 文件存储配置"""
    data_dir: str = ""  # 空字符串表示用默认路径（项目根目录/data/）


@dataclass
class RetrievalConfig:
    """检索配置"""
    top_k: int = 5
    similarity_threshold: float = 0.5


@dataclass
class Settings:
    """全局配置入口"""
    llm: LLMConfig = field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    milvus: MilvusConfig = field(default_factory=MilvusConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)

    # 向后兼容：旧代码可能访问 settings.mysql
    @property
    def mysql(self):
        return self.storage


_settings: Optional[Settings] = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings


def reset_settings(settings: Optional[Settings] = None) -> Settings:
    global _settings
    _settings = settings if settings else Settings()
    return _settings
