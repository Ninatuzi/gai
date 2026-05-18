# ChatWiki

对话级记忆检索模块。结合**短期对话记忆 (Workspace Memory)** 和 **长期知识库 (Knowledge Wiki)**，为对话系统提供精准的上下文召回能力。

详细架构见 [docs/architecture.md](docs/architecture.md)。

## 核心特性

- **Workspace 短期记忆**：每个 chat 一个独立的 workspace，存储对话QA、摘要、向量
- **Knowledge Wiki 长期知识库**：从对话中蒸馏出结构化知识节点（title/summary/content/tags/relations）
- **智能路由 (Router)**：用 LLM 判断新问题应该从哪里检索（workspace / wiki / rag / 组合）
- **每轮自动蒸馏**：对话结束后自动判断是否提取知识并合并到 Wiki
- **HNSW 向量索引**：高召回率、低延迟，离线友好

## 技术栈

| 组件 | 用途 |
|------|------|
| Python 3.13 | 主语言 |
| OpenAI SDK | 调用 vLLM 部署的 DeepSeek/GTE |
| MySQL 8 | 结构化数据 |
| Milvus 2.5 + HNSW | 向量检索 |
| LangChain/LangGraph | （预留）流程编排 |

## 服务依赖

| 服务 | 地址 | 说明 |
|------|------|------|
| DeepSeek-32B | 10.0.6.89:8080 | LLM，vLLM/OpenAI兼容 |
| GTE-Chinese-Large | 10.7.10.102:8001 | Embedding，1024维 |
| MySQL | 10.7.10.102:3306 | 结构化数据 |
| Milvus | 10.7.10.102:19530 | 向量数据库 |

如需修改连接信息，编辑 `chatwiki/config/settings.py`。

## 安装与使用

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 检查服务连通性
python main.py health

# 3. 初始化数据库 & Milvus collection（首次运行）
python main.py init

# 4. 跑完整演示（5个workspace × 多轮对话）
python main.py demo

# 5. 交互式对话
python main.py chat my_chat_id
```

## 编程接口

```python
from chatwiki import ChatWikiClient

client = ChatWikiClient()
client.init()  # 首次需要

# 创建 workspace
ws_id = client.create_workspace(chat_id="user_001_chat_a")

# 写入对话（自动摘要+蒸馏）
client.add_turn(
    workspace_id=ws_id,
    user_query="锂电池正极材料有哪些？",
    assistant_response="主要有磷酸铁锂、三元材料、钴酸锂、锰酸锂...",
)

# 智能检索
ctx = client.retrieve(workspace_id=ws_id, query="那磷酸铁锂的优势？")
print(ctx.source)              # "workspace_only" or other route
print(ctx.workspace_results)   # 短期记忆命中
print(ctx.wiki_results)        # 长期知识库命中
print(ctx.rag_required)        # 是否需要外部RAG

# 删除 workspace（chat 结束时）
client.delete_workspace(ws_id)
```

## 项目结构

```
chatwiki/
├── config/         # 配置 + Prompt 模板
├── models/         # LLM / Embedding 客户端
├── storage/        # MySQL / Milvus 客户端
├── core/           # workspace / memory / wiki / router / distiller / retriever
├── api/            # 对外统一接口 ChatWikiClient
└── graph/          # （预留）LangGraph 工作流
docs/
└── architecture.md # 完整架构文档
main.py             # 演示入口
```
