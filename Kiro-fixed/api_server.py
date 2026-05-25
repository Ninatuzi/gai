"""
ChatWiki API 服务
提供意图识别、记忆检索、记录对话三个接口，供 Dify 工作流调用。

启动：
    python api_server.py
    或
    uvicorn api_server:app --host 0.0.0.0 --port 8686 --reload

接口列表：
    POST /api/intent          意图识别（判断是否应关联上下文）
    POST /api/memory/search   记忆检索（从历史 Wiki 中检索相关知识）
    POST /api/memory/write    记录对话（写入 Wiki + 话题分组）
    GET  /api/health          健康检查
    POST /api/workspace/create  创建会话空间
"""

import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from chatwiki import ChatWikiAgent
from chatwiki.skills.base import SkillInput

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("chatwiki.api_server")

# ============================================================
# 全局 Agent 实例
# ============================================================

agent: Optional[ChatWikiAgent] = None


def get_agent() -> ChatWikiAgent:
    global agent
    if agent is None:
        agent = ChatWikiAgent()
        agent.init()
        logger.info("ChatWiki Agent 初始化完成")
    return agent


# ============================================================
# FastAPI App
# ============================================================

app = FastAPI(
    title="ChatWiki API",
    description="对话级记忆检索服务 - 为 Dify 提供意图识别、记忆管理能力",
    version="1.0.0",
)


# ============================================================
# 请求/响应模型
# ============================================================

# --- 意图识别 ---
class IntentRequest(BaseModel):
    workspace_id: str = Field(..., description="会话空间ID")
    query: str = Field(..., description="用户问题")


class IntentResponse(BaseModel):
    intent: str = Field(..., description="意图类型: knowledge_query / followup_query / chitchat")
    should_use_context: bool = Field(..., description="是否应关联上下文（Dify用此字段决定是否启用上下文）")
    message: str = Field(default="", description="补充说明")


# --- 记忆检索 ---
class MemorySearchRequest(BaseModel):
    workspace_id: str = Field(..., description="会话空间ID")
    query: str = Field(..., description="检索问题（建议传改写后的问题）")


class WikiResult(BaseModel):
    module_topic: str = Field(default="", description="所属话题模块")
    question: str = Field(default="", description="原始问题")
    summary: str = Field(default="", description="知识摘要（80-120字）")
    answer: str = Field(default="", description="历史回答片段")
    knowledge: List[str] = Field(default_factory=list, description="关联的RAG知识片段")
    module_score: float = Field(default=0.0, description="模块相关度")
    wiki_score: float = Field(default=0.0, description="Wiki条目相关度")


class MemorySearchResponse(BaseModel):
    found: bool = Field(..., description="是否命中历史记忆")
    modules_hit: int = Field(default=0, description="命中模块数")
    wiki_count: int = Field(default=0, description="命中Wiki条目数")
    results: List[WikiResult] = Field(default_factory=list, description="命中的Wiki记录列表")
    context_text: str = Field(default="", description="格式化的上下文文本（可直接塞入prompt）")


# --- 记录对话 ---
class MemoryWriteRequest(BaseModel):
    workspace_id: str = Field(..., description="会话空间ID")
    query: str = Field(..., description="用户问题")
    answer: str = Field(..., description="助手回答")
    intent: str = Field(default="knowledge_query", description="本轮意图类型")
    knowledge: List[str] = Field(default_factory=list, description="本轮RAG检索到的知识片段（可选）")


class MemoryWriteResponse(BaseModel):
    success: bool = Field(..., description="是否写入成功")
    module_id: str = Field(default="", description="归属模块ID")
    module_topic: str = Field(default="", description="归属模块话题名")
    wiki_id: str = Field(default="", description="Wiki条目ID")
    turn_number: int = Field(default=0, description="对话轮次")
    wiki_summary: str = Field(default="", description="生成的Wiki摘要")
    message: str = Field(default="", description="补充说明")


# --- 创建会话 ---
class WorkspaceCreateRequest(BaseModel):
    chat_id: str = Field(..., description="外部会话标识（如Dify的conversation_id）")


class WorkspaceCreateResponse(BaseModel):
    workspace_id: str = Field(..., description="创建的会话空间ID")


# --- 健康检查 ---
class HealthResponse(BaseModel):
    status: str = Field(..., description="服务状态")
    services: Dict[str, bool] = Field(default_factory=dict, description="各子服务状态")


# ============================================================
# 接口实现
# ============================================================

@app.post("/api/intent", response_model=IntentResponse, summary="意图识别")
async def api_intent(req: IntentRequest):
    """
    判断用户问题的意图类型。
    
    Dify 调用逻辑：
    - chitchat → 关闭上下文关联，直接回答
    - knowledge_query → 正常走 RAG
    - followup_query → 先做问题改写，再走 RAG
    """
    ag = get_agent()
    t0 = time.time()

    try:
        # 自动确保 workspace 存在
        ag.wiki_skill.mysql.ensure_workspace(req.workspace_id)

        # 获取上下文信息
        modules = ag.wiki_skill.get_modules(req.workspace_id)
        recent_modules = [m["topic"] for m in modules[:5]]
        recent_chat = ag.wiki_skill.get_recent_context(req.workspace_id, n=5)

        # 调用意图识别 Skill
        inp = SkillInput(
            query=req.query,
            workspace_id=req.workspace_id,
            context={"recent_modules": recent_modules, "recent_chat": recent_chat},
        )
        out = ag.intent_skill.run(inp)
        intent = out.data.get("intent", "knowledge_query")

        # 判断是否应关联上下文
        should_use_context = intent != "chitchat"

        elapsed = time.time() - t0
        logger.info("意图识别: query=%r, intent=%s, %.2fs", req.query[:30], intent, elapsed)

        return IntentResponse(
            intent=intent,
            should_use_context=should_use_context,
            message=f"耗时{elapsed:.1f}s",
        )

    except Exception as e:
        logger.error("意图识别异常: %s", e)
        # 异常时默认返回 knowledge_query（安全兜底）
        return IntentResponse(
            intent="knowledge_query",
            should_use_context=True,
            message=f"识别异常，默认knowledge_query: {str(e)}",
        )


@app.post("/api/memory/search", response_model=MemorySearchResponse, summary="记忆检索")
async def api_memory_search(req: MemorySearchRequest):
    """
    从历史 Wiki 中检索与当前问题相关的记忆。
    
    Dify 调用逻辑：
    - 把返回的 context_text 作为额外上下文塞入 LLM prompt
    - 或者取 results 列表自行组装 prompt
    """
    ag = get_agent()
    t0 = time.time()

    try:
        # 自动确保 workspace 存在
        ag.wiki_skill.mysql.ensure_workspace(req.workspace_id)

        inp = SkillInput(query=req.query, workspace_id=req.workspace_id)
        out = ag.wiki_skill.run(inp)

        found = out.data.get("found", False)
        modules_hit = out.data.get("modules_hit", 0)
        wiki_count = out.data.get("wiki_count", 0)
        context_text = out.data.get("context_text", "")
        wikis = out.data.get("wikis", [])

        # Fallback：向量检索没命中时，返回最近几轮原始对话作为上下文
        # 解决"把刚才的内容整理成表格"等操作指令语义搜不到的问题
        if not found and not context_text:
            recent_context = ag.wiki_skill.get_recent_context(req.workspace_id, n=5)
            if recent_context:
                context_text = recent_context
                found = True
                logger.info("记忆检索 fallback: 向量未命中，返回最近5轮对话")

        # 转换为响应格式
        results = []
        for w in wikis:
            knowledge = w.get("knowledge") or []
            if isinstance(knowledge, str):
                import json
                try:
                    knowledge = json.loads(knowledge)
                except Exception:
                    knowledge = []

            results.append(WikiResult(
                module_topic=w.get("_module_topic", ""),
                question=w.get("query", ""),
                summary=w.get("_wiki_summary") or w.get("summary", ""),
                answer=w.get("answer", "")[:500],
                knowledge=[str(k)[:300] for k in knowledge[:3]],
                module_score=w.get("_module_score", 0.0),
                wiki_score=w.get("_wiki_score", 0.0),
            ))

        elapsed = time.time() - t0
        logger.info("记忆检索: query=%r, found=%s, modules=%d, wikis=%d, %.2fs",
                   req.query[:30], found, modules_hit, wiki_count, elapsed)

        return MemorySearchResponse(
            found=found,
            modules_hit=modules_hit,
            wiki_count=wiki_count,
            results=results,
            context_text=context_text,
        )

    except Exception as e:
        logger.error("记忆检索异常: %s", e)
        return MemorySearchResponse(found=False, modules_hit=0, wiki_count=0, results=[], context_text="")


@app.post("/api/memory/write", response_model=MemoryWriteResponse, summary="记录对话")
async def api_memory_write(req: MemoryWriteRequest):
    """
    将一轮对话写入 Wiki 记忆系统。
    
    Dify 调用逻辑：
    - 每轮对话结束后调用
    - chitchat 只记 chat_log，不写入 wiki（不污染知识库）
    - knowledge_query / followup_query 写入 wiki + chat_log
    """
    ag = get_agent()
    t0 = time.time()

    try:
        # 自动确保 workspace 存在
        ag.wiki_skill.mysql.ensure_workspace(req.workspace_id)

        # 无论什么意图，都记录 chat_log
        ag.wiki_skill.log_chat(
            workspace_id=req.workspace_id,
            query=req.query,
            answer=req.answer,
            intent=req.intent,
        )

        # chitchat 不写入 wiki
        if req.intent == "chitchat":
            elapsed = time.time() - t0
            logger.info("记录对话(chitchat): query=%r, 仅记chat_log, %.2fs", req.query[:30], elapsed)
            return MemoryWriteResponse(
                success=True,
                message="闲聊已记录到chat_log，未写入wiki",
            )

        # 知识问答：写入 wiki
        write_result = ag.wiki_skill.write(
            workspace_id=req.workspace_id,
            query=req.query,
            answer=req.answer,
            knowledge=req.knowledge,
            intent=req.intent,
        )

        module_id = write_result.get("module_id", "")
        wiki_id = write_result.get("wiki_id", "")
        turn_number = write_result.get("turn_number", 0)
        wiki_summary = write_result.get("wiki_summary", "")

        # 获取模块话题名
        module_topic = ""
        if module_id:
            module = ag.wiki_skill.mysql.get_module(module_id)
            if module:
                module_topic = module.get("topic", "")

        elapsed = time.time() - t0
        logger.info("记录对话(wiki): query=%r, module=%s, turn=%d, %.2fs",
                   req.query[:30], module_topic, turn_number, elapsed)

        return MemoryWriteResponse(
            success=True,
            module_id=module_id,
            module_topic=module_topic,
            wiki_id=wiki_id,
            turn_number=turn_number,
            wiki_summary=wiki_summary,
            message=f"已归入模块[{module_topic}]",
        )

    except Exception as e:
        logger.error("记录对话异常: %s", e)
        return MemoryWriteResponse(success=False, message=f"写入失败: {str(e)}")


@app.post("/api/workspace/create", response_model=WorkspaceCreateResponse, summary="创建会话空间")
async def api_workspace_create(req: WorkspaceCreateRequest):
    """
    创建新的会话空间。Dify 新开对话时调用一次。
    如果 chat_id 已存在则返回现有空间。
    """
    ag = get_agent()
    try:
        ws_id = ag.create_workspace(req.chat_id)
        logger.info("创建/获取workspace: chat_id=%s, ws_id=%s", req.chat_id, ws_id)
        return WorkspaceCreateResponse(workspace_id=ws_id)
    except Exception as e:
        logger.error("创建workspace异常: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/health", response_model=HealthResponse, summary="健康检查")
async def api_health():
    """检查各子服务连通性"""
    ag = get_agent()
    status = ag.health_check()
    all_ok = all(status.values())
    return HealthResponse(
        status="ok" if all_ok else "degraded",
        services=status,
    )


# ============================================================
# 启动入口
# ============================================================

if __name__ == "__main__":
    # 预初始化 Agent（避免第一次请求慢）
    get_agent()
    logger.info("ChatWiki API 启动中...")
    uvicorn.run(app, host="0.0.0.0", port=7001)
