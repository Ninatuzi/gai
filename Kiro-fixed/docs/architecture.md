# ChatWiki 技术架构文档

## 1. 项目概述

### 1.1 项目定位
ChatWiki 是一个**对话级记忆检索模块**，结合短期对话记忆和长期知识Wiki，为对话系统提供精准的上下文召回能力。

### 1.2 核心目标
- 精准判断新问题与历史对话的关联性
- 提供对话级短期记忆（Workspace）和长期知识库（Wiki）
- 作为通用接口模块，可插入现有对话系统
- 从对话中蒸馏知识，不断完善Wiki

### 1.3 业务场景
- 公司锂电池知识对话系统
- 现有RAG切块不够精准，需要更结构化的知识组织
- 员工对话时需要精准的上下文记忆

---

## 2. 系统架构

### 2.1 整体架构图

```
┌─────────────────────────────────────────────────────────┐
│                      用户对话入口                         │
└─────────────────────────┬───────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│                   路由判断层 (Router)                     │
│  判断策略：                                              │
│  1. 仅需短期记忆 → Workspace Memory                      │
│  2. 仅需长期知识 → Knowledge Wiki                        │
│  3. 两者结合 → Workspace + Wiki                          │
│  4. 都不够 → 转发到 RAG 系统                             │
│  5. 混合 → Wiki + RAG                                   │
└───┬──────────────┬──────────────┬───────────────────────┘
    │              │              │
    ▼              ▼              ▼
┌────────┐  ┌──────────┐  ┌──────────────┐
│Workspace│  │Knowledge │  │  RAG System  │
│ Memory  │  │   Wiki   │  │  (公司已有)   │
│(短期记忆)│  │(长期知识库)│  │              │
└────────┘  └──────────┘  └──────────────┘
    │              │
    │              ▲
    └──────────────┘
      知识蒸馏（对话 → Wiki）
```

### 2.2 核心模块

| 模块 | 职责 | 存储 |
|------|------|------|
| Workspace Memory | 当前对话的短期记忆，包含QA对、摘要 | Milvus + MySQL |
| Knowledge Wiki | 长期结构化知识库，类知识图谱的.md组织 | Milvus + MySQL + 文件系统 |
| Router | 路由判断，决定检索策略 | LLM判断 |
| Distiller | 从对话中蒸馏知识写入Wiki | LLM + 规则 |
| API Layer | 通用接口层，供外部系统调用 | - |

---

## 3. 技术选型

### 3.1 基础设施

| 组件 | 版本/规格 | 用途 |
|------|-----------|------|
| Python | 3.13.5 | 主开发语言 |
| LangChain + LangGraph | latest | LLM编排框架 |
| MySQL | 8.0 | 结构化数据存储（元数据、对话记录） |
| Milvus | 2.5.15 | 向量数据库（语义检索） |
| MinIO | 2023-03-20 | 对象存储（Milvus依赖） |
| etcd | 3.5.5 | 分布式配置（Milvus依赖） |

### 3.2 模型

| 模型 | 规格 | 用途 |
|------|------|------|
| DeepSeek-32B-f16 | 远程服务器部署 | 路由判断、摘要生成、知识蒸馏 |
| GTE-Chinese-Large | 本地部署(vLLM, port:8001) | 文本向量化，维度1024 |

### 3.3 向量索引
- **索引类型**: HNSW（Hierarchical Navigable Small World）
- **优势**: 高召回率、低延迟，适合离线场景
- **参数建议**: M=16, efConstruction=256, ef=128（后续可调优）

---

## 4. 数据模型设计

### 4.1 Workspace Memory（短期记忆）

#### MySQL 表结构

```sql
-- 工作空间表
CREATE TABLE workspace (
    workspace_id VARCHAR(36) PRIMARY KEY,  -- UUID
    chat_id VARCHAR(64) NOT NULL,           -- 关联的对话ID
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    status ENUM('active', 'archived', 'deleted') DEFAULT 'active',
    INDEX idx_chat_id (chat_id)
);

-- 对话轮次表
CREATE TABLE conversation_turn (
    turn_id VARCHAR(36) PRIMARY KEY,
    workspace_id VARCHAR(36) NOT NULL,
    turn_number INT NOT NULL,              -- 轮次序号（1-10）
    user_query TEXT NOT NULL,              -- 用户问题
    assistant_response TEXT NOT NULL,      -- 助手回答
    summary TEXT,                          -- 本轮摘要
    tags JSON,                            -- 标签
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (workspace_id) REFERENCES workspace(workspace_id) ON DELETE CASCADE,
    INDEX idx_workspace_turn (workspace_id, turn_number)
);
```

#### Milvus Collection: `workspace_memory`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(36) | 主键，对应turn_id |
| workspace_id | VARCHAR(36) | 工作空间ID |
| embedding | FLOAT_VECTOR(1024) | GTE向量 |
| text_type | VARCHAR(16) | "query" / "response" / "summary" |
| content | VARCHAR(4096) | 原始文本 |

### 4.2 Knowledge Wiki（长期知识库）

#### MySQL 表结构

```sql
-- Wiki知识节点表
CREATE TABLE wiki_node (
    node_id VARCHAR(36) PRIMARY KEY,
    title VARCHAR(256) NOT NULL,           -- 标题（如"福鼎肉片"）
    summary TEXT,                          -- 摘要
    content TEXT NOT NULL,                 -- 完整内容（Markdown格式）
    tags JSON,                            -- 标签数组
    category VARCHAR(128),                -- 分类
    source_type ENUM('distilled', 'manual', 'imported') DEFAULT 'distilled',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_category (category),
    FULLTEXT INDEX idx_content (title, content)
);

-- Wiki关系表（类知识图谱的关联）
CREATE TABLE wiki_relation (
    relation_id VARCHAR(36) PRIMARY KEY,
    source_node_id VARCHAR(36) NOT NULL,
    target_node_id VARCHAR(36) NOT NULL,
    relation_type VARCHAR(64) NOT NULL,    -- "belongs_to", "related_to", "part_of" 等
    weight FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (source_node_id) REFERENCES wiki_node(node_id) ON DELETE CASCADE,
    FOREIGN KEY (target_node_id) REFERENCES wiki_node(node_id) ON DELETE CASCADE,
    INDEX idx_source (source_node_id),
    INDEX idx_target (target_node_id)
);

-- 知识蒸馏记录表
CREATE TABLE distill_log (
    log_id VARCHAR(36) PRIMARY KEY,
    workspace_id VARCHAR(36) NOT NULL,
    turn_id VARCHAR(36),
    node_id VARCHAR(36),                   -- 写入/更新的wiki节点
    action ENUM('create', 'update', 'merge') NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
```

#### Milvus Collection: `wiki_knowledge`

| 字段 | 类型 | 说明 |
|------|------|------|
| id | VARCHAR(36) | 主键，对应node_id |
| embedding | FLOAT_VECTOR(1024) | GTE向量 |
| title | VARCHAR(256) | 标题 |
| tags | VARCHAR(512) | 标签JSON |
| category | VARCHAR(128) | 分类 |

---


## 5. 核心流程设计

### 5.1 对话写入流程

```
用户发送消息
    │
    ▼
① 接收 user_query + assistant_response
    │
    ▼
② GTE Embedding 向量化（query + response 分别向量化）
    │
    ▼
③ 写入 MySQL (conversation_turn 表)
    │
    ▼
④ 写入 Milvus (workspace_memory collection)
    │
    ▼
⑤ LLM 生成本轮摘要 + 标签
    │
    ▼
⑥ 更新 MySQL 的 summary 和 tags 字段
    │
    ▼
⑦ [异步] 知识蒸馏判断：本轮对话是否包含可提取的知识？
    │── 是 → 蒸馏写入 Wiki
    └── 否 → 结束
```

### 5.2 检索召回流程

```
新的用户问题
    │
    ▼
① GTE Embedding 向量化
    │
    ▼
② 路由判断（Router）
    │  - 输入：用户问题 + 最近2轮对话上下文
    │  - 输出：检索策略（workspace / wiki / rag / 混合）
    │
    ├─── workspace_only ────→ 搜索当前workspace的Milvus向量
    │                              │
    ├─── wiki_only ─────────→ 搜索wiki_knowledge的Milvus向量
    │                              │
    ├─── workspace_and_wiki ─→ 两者都搜，合并排序
    │                              │
    ├─── rag_only ──────────→ 转发到外部RAG系统
    │                              │
    └─── wiki_and_rag ──────→ Wiki + RAG 结果合并
    │
    ▼
③ 结果重排序 + 返回Top-K
```

### 5.3 知识蒸馏流程（每轮触发）

```
每轮对话写入完成后立即触发
    │
    ▼
① LLM 判断：本轮对话是否包含可提取的新知识？
    │
    ├── 否 → 结束
    │
    └── 是 ─→ ② LLM 提取知识点
                    │
                    ▼
               ③ 生成结构化知识
                  - title（标题）
                  - summary（摘要）
                  - content（详细内容，Markdown）
                  - tags（标签列表）
                  - relations（关联关系）
                    │
                    ▼
               ④ 查重：向量相似度检索现有Wiki
                    │
                    ├── 相似度 > 0.9 → 合并/更新已有节点
                    │
                    └── 相似度 < 0.9 → 创建新节点
                    │
                    ▼
               ⑤ 写入 MySQL + Milvus
                    │
                    ▼
               ⑥ 建立关联关系（wiki_relation）
```

### 5.4 路由判断策略 (Router)

Router 使用 LLM（DeepSeek-32B）进行意图判断：

```python
ROUTER_PROMPT = """
你是一个智能路由器，需要判断用户的新问题应该从哪里检索信息。

## 上下文
- 当前对话最近的摘要: {recent_summaries}
- 用户新问题: {user_query}

## 判断规则
1. workspace_only: 问题明显是对之前对话内容的追问/澄清/延伸
2. wiki_only: 问题是一个独立的知识性问题，跟当前对话无关
3. workspace_and_wiki: 问题既涉及之前讨论的内容，又需要额外的知识补充
4. rag_only: 问题需要检索专业文档（如锂电池技术规格等）
5. wiki_and_rag: 问题需要结合Wiki知识和专业文档

请只返回以下之一：workspace_only / wiki_only / workspace_and_wiki / rag_only / wiki_and_rag
"""
```

---

## 6. 项目结构

```
chatwiki/
├── config/
│   ├── __init__.py
│   ├── settings.py              # 全局配置（数据库连接、模型地址等）
│   └── prompts.py               # LLM Prompt 模板
├── core/
│   ├── __init__.py
│   ├── workspace.py             # Workspace 管理（创建/删除/查询）
│   ├── memory.py                # 短期记忆读写
│   ├── wiki.py                  # Wiki 知识库管理
│   ├── router.py                # 路由判断
│   ├── distiller.py             # 知识蒸馏
│   └── retriever.py             # 统一检索接口
├── models/
│   ├── __init__.py
│   ├── llm.py                   # DeepSeek LLM 封装
│   └── embedding.py             # GTE Embedding 封装
├── storage/
│   ├── __init__.py
│   ├── mysql_client.py          # MySQL 连接和操作
│   └── milvus_client.py         # Milvus 连接和操作
├── api/
│   ├── __init__.py
│   └── interface.py             # 对外统一接口（插入即用）
├── graph/
│   ├── __init__.py
│   └── workflow.py              # LangGraph 工作流定义
├── tests/
│   ├── test_memory.py
│   ├── test_wiki.py
│   └── test_router.py
├── requirements.txt
├── main.py                      # 入口/演示
└── README.md
```

---

## 7. 接口设计（API Layer）

### 7.1 统一接口 - 插入即用

```python
from chatwiki import ChatWikiClient

# 初始化（连接数据库、模型等）
client = ChatWikiClient(config_path="config/settings.yaml")

# === 对话写入 ===
# 每轮对话结束后调用
client.add_turn(
    workspace_id="ws_001",
    user_query="锂电池的正极材料有哪些？",
    assistant_response="锂电池正极材料主要有..."
)

# === 检索召回 ===
# 用户提问时调用，返回相关上下文
context = client.retrieve(
    workspace_id="ws_001",
    query="那负极呢？"
)
# context.source = "workspace_only"
# context.results = [...]

# === Workspace 管理 ===
client.create_workspace(chat_id="chat_123")
client.delete_workspace(workspace_id="ws_001")  # 对话删除时调用
client.list_workspaces()

# === Wiki 管理 ===
client.wiki_search(query="锂电池正极材料")
client.wiki_add_node(title="...", content="...", tags=[...])
```

---

## 8. 环境配置

### 8.1 连接信息

```yaml
# config/settings.yaml

# LLM 模型（DeepSeek-32B，vLLM部署，OpenAI兼容接口）
llm:
  base_url: "http://10.0.6.89:8080/v1"
  model_name: "DeepSeek_32B_f16"
  api_key: "EMPTY"                # 不需要key，已验证curl直接可用
  temperature: 0.1
  max_tokens: 2048

# Embedding 模型（GTE-Chinese-Large，vLLM部署）
embedding:
  base_url: "http://10.7.10.102:8001/v1"
  api_key: "qwertyuiop1A."
  model_name: "/root/embedding_model/nlp_gte_sentence-embedding_chinese-large"
  dimension: 1024

# MySQL
mysql:
  host: "10.7.10.102"
  port: 3306
  user: "root"
  password: "qwertyuiop1A."
  database: "chatwiki"

# Milvus
milvus:
  host: "10.7.10.102"
  port: 19530

# Workspace 配置
workspace:
  max_turns: -1            # 不限制轮次，workspace可以无限写入
  retrieval_window: 10     # 检索时最多回溯10轮
  max_workspaces: 5        # MVP阶段最大workspace数

# 检索配置
retrieval:
  top_k: 5
  similarity_threshold: 0.7

# HNSW 索引配置
hnsw:
  M: 16
  efConstruction: 256
  ef: 128
```

---

## 9. MVP 范围定义

### Phase 1: 基础能力（当前目标）

- [x] 项目骨架搭建
- [ ] MySQL / Milvus 连接封装
- [ ] GTE Embedding 封装
- [ ] DeepSeek LLM 封装
- [ ] Workspace 创建/删除/管理
- [ ] 对话写入（向量化 + 存储）
- [ ] 基础检索（向量相似度搜索）
- [ ] 路由判断（LLM Router）
- [ ] 简单知识蒸馏（对话 → Wiki节点）
- [ ] Wiki 基础CRUD
- [ ] 统一API接口
- [ ] 5个Workspace + 10轮对话 测试

### Phase 2: 增强（后续）

- [ ] Wiki 关系图谱完善
- [ ] 摘要自动生成优化
- [ ] 对接公司 RAG 系统
- [ ] 多路召回排序优化
- [ ] Workspace 自动清理策略

---

## 10. 技术要点 & 注意事项

### 10.1 HNSW 索引选择理由
- 离线服务器，不需要极致写入性能
- 查询精度高（相比IVF_FLAT）
- 支持增量插入，无需重建索引
- 内存占用可控（M=16时每个向量额外约 16*2*4 = 128 bytes）

### 10.2 知识组织方式（Wiki的.md逻辑）
```
wiki_nodes:
  - title: "锂电池正极材料"
    summary: "锂电池正极材料的分类和特性概述"
    tags: ["锂电池", "正极", "材料"]
    relations:
      - target: "磷酸铁锂" | type: "includes"
      - target: "三元材料" | type: "includes"
      - target: "锂电池" | type: "belongs_to"
  
  - title: "磷酸铁锂"
    summary: "磷酸铁锂(LFP)的特性和应用"
    tags: ["正极材料", "LFP", "安全性"]
    relations:
      - target: "锂电池正极材料" | type: "belongs_to"
      - target: "电池安全" | type: "related_to"
```

### 10.3 LangGraph 工作流
使用 LangGraph 编排主要流程（路由→检索→生成），便于：
- 可视化流程
- 状态管理
- 错误重试
- 后续扩展节点

---

## 11. 已确认配置

| # | 问题 | 结论 |
|---|------|------|
| 1 | DeepSeek 连接 | `http://10.0.6.89:8080/v1`（OpenAI兼容接口） |
| 2 | Milvus 连接 | `10.7.10.102:19530` |
| 3 | GTE embedding 维度 | 1024（GTE-Chinese-Large） |
| 4 | "十轮对话"含义 | 检索时最多回溯最近10轮，不限制workspace总轮次 |
| 5 | 知识蒸馏触发时机 | 每轮对话结束后都触发蒸馏 |
| 6 | Embedding 服务地址 | `http://10.7.10.102:8001/v1` |
| 7 | MySQL 地址 | `10.7.10.102:3306` |

## 12. 待确认事项

| # | 问题 | 状态 |
|---|------|------|
| 1 | DeepSeek 是否需要 api_key？ | ✅ 不需要，curl测试已验证 |
| 2 | RAG系统的对接接口格式 | 后续对接时确认 |
| 3 | GTE embedding 维度实际测试验证 | 开发时验证 |
