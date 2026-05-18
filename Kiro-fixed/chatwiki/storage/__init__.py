from chatwiki.storage.json_client import JSONClient
from chatwiki.storage.milvus_client import MilvusClient

# 向后兼容：上层代码用 MySQLClient 名字引用的地方自动指向 JSONClient
MySQLClient = JSONClient
