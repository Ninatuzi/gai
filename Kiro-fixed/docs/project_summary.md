# ChatWiki 项目总结

## 一、项目定位

ChatWiki 是一个**对话级记忆检索 Agent 模块**，基于 Harness/SKILL 架构，为公司的锂电池知识对话系统提供：
- 历史对话记忆（History Memory）：按话题模块自动分组的 Wiki 格式记录
- 智能检索：判断新问题是否跟历史对话有关，有关就返回相关 wiki
- 答案聚合：Wiki + RAG 结合生成最终回答
- 作为独立功能模块，后续接入公司现有系统

---

## 二、Leader 的核心要求

1. **不是做缓存层**，是做一个 Agent + Wiki 格式的 History Memory
2. **Wiki 格式**：每轮对话打包为 `{query, answer, knowledge_from_rag}`，外部有 module summary 作为父块
3. **模块分组**：连续同话题的对话归为一个模块，不同话题创建新模块
4. **Agent 框架**：像 Harness 一样，里面有多个 Skill（意图识别、问题改写、Wiki检索、RAG、保底等）
5. **Workspace 隔离**：每个对话框一个 workspace，删除对话框则清除该 workspace
6. **闲聊不污染**：闲聊不写入 Wiki 知识模块
7. **可以参考 Harness/SKILL 的做法**
8. **问题改写有同事在做**（我这边留了接口，当前用 LLM 自己做指代消解）
9. **RAG 接口暂时 Mock**，后续对接真实系统

---

## 三、已实现的技术方案

### 架构

```
用户提问
    │
    ▼
┌──────────────────────────────────────────────┐
│            Harness（Agent 管理器）             │
│  注册 Skill → 按流程调度 → 维护状态           │
└──────────────────────────────────────────────┘
         │         │         │         │         │         │
         ▼         ▼         ▼         ▼         ▼         ▼
   ┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐┌─────────┐
   │ 意图识别 ││ 问题改写 ││ Wiki    ││  RAG   ││ 答案聚合 ││ 保底直答 │
   │ Skill   ││ Skill   ││ Memory  ││ Skill  ││ Skill   ││ Skill   │
   └─────────┘└─────────┘└─────────┘└─────────┘└─────────┘└─────────┘
```

### 流程

```
意图识别 → chitchat? → 保底直答（不写wiki）
         → knowledge/followup → 问题改写（指代消解）
         → Wiki 检索（向量搜索 module summary）
         → 需要RAG? → RAG检索
         → 答案聚合（Wiki + RAG 综合）
         → 写入 Wiki（话题判断归模块）
         → 记录 chat_log（含闲聊，供改写参考）
```

### 存储

| 存储 | 存什么 | 文件/服务 |
|------|--------|-----------|
| JSON 文件 | workspace + modules + wikis + chat_log | `data/{workspace_id}.json` |
| Milvus | module 的 summary 向量（HNSW/COSINE） | 10.7.10.102:19530 |

### 6 个 Skill

| Skill | 文件 | 功能 |
|-------|------|------|
| IntentSkill | `skills/intent_tool.py` | 意图识别：knowledge_query / followup_query / chitchat |
| QueryRewriteSkill | `skills/query_rewrite_tool.py` | 指代消解：补全"它/那个/刚才说的" |
| WikiMemorySkill | `skills/wiki_memory_tool.py` | Wiki 检索 + 写入 + 模块管理 |
| RAGSkill | `skills/rag_tool.py` | RAG 检索（当前 Mock） |
| AggregatorSkill | `skills/aggregator_tool.py` | 答案聚合（Wiki + RAG → 最终回答） |
| FallbackSkill | `skills/fallback_tool.py` | 保底直答（带 15 轮对话记忆） |

### 模型

| 模型 | 地址 | 用途 |
|------|------|------|
| DeepSeek-32B-f16 | http://10.0.6.89:8080/v1 | LLM（意图/改写/话题判断/聚合） |
| BGE-M3 | http://10.7.10.102:8001/v1 | Embedding（1024维向量） |

---

## 四、当前代码位置

GitHub: https://github.com/Ninatuzi/Kiro/tree/feature/agent-rewrite

```
chatwiki/
├── config/settings.py       # 全局配置
├── config/prompts.py        # Prompt 模板
├── models/llm.py            # DeepSeek 客户端（含<think>去除）
├── models/embedding.py      # BGE-M3 客户端
├── storage/json_client.py   # JSON 文件存储
├── storage/milvus_client.py # Milvus 向量检索
├── skills/base.py           # BaseSkill + SkillInput + SkillOutput
├── skills/intent_tool.py    # 意图识别
├── skills/query_rewrite_tool.py  # 问题改写（指代消解）
├── skills/wiki_memory_tool.py    # History Memory
├── skills/rag_tool.py       # RAG（Mock）
├── skills/aggregator_tool.py     # 答案聚合
├── skills/fallback_tool.py  # 保底直答
├── agent/graph.py           # Harness（Agent管理器）
├── core/workspace.py        # Workspace 管理
├── api/interface.py         # ChatWikiAgent 对外接口
main.py                      # 演示脚本
```

---

## 五、已知问题

1. **话题判断只跟最新模块比较**：如果用户跳回之前的话题，会创建新模块而不是归到之前那个
2. **问题改写依赖上下文质量**：如果 chat_log 里的上下文不够，改写可能不准
3. **RAG 是 Mock**：只有 5 条固定的锂电池知识片段，关键词匹配
4. **每轮多次 LLM 调用**：意图识别 + 改写 + 话题判断 + 摘要生成 + 聚合 = 约 5 次 LLM 调用，每轮耗时 15-30 秒

---

## 六、接下来要做的

| 优先级 | 任务 | 说明 |
|--------|------|------|
| P0 | **迁移到 LangGraph** | 把 Harness 的手写状态机改为 LangGraph StateGraph |
| P1 | **对接真实 RAG** | 等 leader 给接口，改 `skills/rag_tool.py` |
| P1 | **对接问题改写接口** | 等同事接口，改 `skills/query_rewrite_tool.py` |
| P2 | 性能优化 | 减少 LLM 调用次数（合并意图+话题判断？） |
| P2 | 话题判断优化 | 支持归到历史模块（不只是最新模块） |
| P3 | LangGraph checkpoint | 利用 LangGraph 的状态持久化机制 |

---

## 七、服务器环境

| 项目 | 信息 |
|------|------|
| 服务器 | 10.7.10.102（离线 Linux） |
| Python | 3.13.5（conda 环境 byx_1） |
| DeepSeek | 10.0.6.89:8080（同事服务器） |
| BGE-M3 | 10.7.10.102:8001（本地 vLLM，需手动启动） |
| Milvus | 10.7.10.102:19530（Docker standalone） |
| openai SDK | 1.82.0 |
| pymilvus | 2.5.6 |
| vLLM | 0.11.0（跟 openai 1.82.0 有兼容性问题，BGE 不带 --task embed 参数启动） |

### 启动 BGE 服务

```bash
nohup vllm serve /root/embedding_model/bge-m3 \
  --port 8001 \
  --api-key qwertyuiop1A. \
  --gpu-memory-utilization 0.7 > /root/bge_vllm.log 2>&1 &
```

### 运行 ChatWiki

```bash
cd /root/BYX_2/Kiro-feature-agent-rewrite
rm -rf data/
python main.py init
python main.py demo
python main.py chat test_chat
```

---

## 八、关键设计决策记录

1. **DeepSeek 是推理模型**：输出有 `<think>` 块，所有 LLM 调用都通过 `_strip_think()` 去除
2. **max_tokens 要给足**：推理模型 `<think>` 占 200-800 tokens，所有调用至少 500+
3. **向量相似度兜底阈值 0.85**：LLM 话题判断说 new 但相似度 > 0.85 时仍归 same
4. **闲聊处理**：不写 wiki，不污染模块，但记到 chat_log（供改写参考）
5. **chat_log 窗口 15 轮**：Fallback 和问题改写能看到最近 15 轮对话（含闲聊）
6. **JSON 存储替代 MySQL**：leader 不想要数据库，每个 workspace 一个 JSON 文件
7. **Milvus 保留**：向量检索还是需要的（module summary 向量匹配）
