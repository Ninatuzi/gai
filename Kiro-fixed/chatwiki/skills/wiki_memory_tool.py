"""
Skill: History Memory（Wiki）
完整的历史记忆技能：
- Module 话题分组管理
- 向量检索相关模块
- 返回 wiki 列表（Q+A+Knowledge 格式）
- 写入新的 wiki 记录
"""

import logging
from typing import Any, Dict, List, Optional

from chatwiki.config import get_settings
from chatwiki.config.prompts import MODULE_SUMMARY_PROMPT, TOPIC_JUDGE_PROMPT
from chatwiki.models import EmbeddingClient, LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput
from chatwiki.storage import MilvusClient, MySQLClient

logger = logging.getLogger(__name__)


class WikiMemorySkill(BaseSkill):
    """
    History Memory Skill
    
    检索模式 (run):
        输入: query + workspace_id
        输出: 相关的历史 wiki 列表 + 格式化上下文

    写入模式 (write):
        将新的 Q+A+Knowledge 写入 wiki，自动归到对应话题模块
    """

    def __init__(self, mysql: MySQLClient, milvus: MilvusClient,
                 llm: LLMClient, embedding: EmbeddingClient):
        self.mysql = mysql
        self.milvus = milvus
        self.llm = llm
        self.embedding = embedding
        self.settings = get_settings()

    @property
    def name(self) -> str:
        return "history_memory"

    @property
    def description(self) -> str:
        return "从历史对话记忆中检索相关信息（Wiki格式：Q+A+Knowledge），支持话题自动分组"

    # ============================================================
    # Skill 标准接口：检索
    # ============================================================

    def run(self, skill_input: SkillInput) -> SkillOutput:
        """
        检索历史 wiki
        输出 data:
            - found: bool
            - modules_hit: int
            - wikis: List[Dict]  命中的 wiki 记录
            - context_text: str  格式化的上下文文本
        """
        workspace_id = skill_input.workspace_id
        query = skill_input.query
        top_k = self.settings.retrieval.top_k
        threshold = self.settings.retrieval.similarity_threshold

        if not workspace_id:
            return SkillOutput(success=False, message="缺少 workspace_id")

        # 1. 向量化 query
        try:
            vec = self.embedding.embed(query)
        except Exception as e:
            logger.error("Wiki检索向量化失败: %s", e)
            return SkillOutput(success=True, data={"found": False, "wikis": [], "context_text": ""})

        # 2. 搜索相关 module
        hits = self.milvus.search_modules(workspace_id=workspace_id, query_embedding=vec, top_k=top_k)
        hits = [h for h in hits if h["score"] >= threshold]

        if not hits:
            logger.info("Wiki检索: 无命中 (ws=%s, query=%r)", workspace_id, query[:30])
            return SkillOutput(success=True, data={"found": False, "modules_hit": 0, "wikis": [], "context_text": ""})

        # 3. 获取命中模块下的 wiki 记录
        all_wikis: List[Dict[str, Any]] = []
        for h in hits:
            module_wikis = self.mysql.get_wikis_by_module(h["module_id"])
            for w in module_wikis:
                w["_module_topic"] = h["topic"]
                w["_module_score"] = h["score"]
                # 解析 knowledge JSON
                if isinstance(w.get("knowledge"), str):
                    import json
                    try:
                        w["knowledge"] = json.loads(w["knowledge"])
                    except Exception:
                        w["knowledge"] = []
            all_wikis.extend(module_wikis)

        # 4. 格式化上下文
        context_text = self._format_context(all_wikis)

        logger.info("Wiki检索: 命中 %d 模块, %d 条wiki", len(hits), len(all_wikis))
        return SkillOutput(
            success=True,
            data={
                "found": True,
                "modules_hit": len(hits),
                "wikis": all_wikis,
                "context_text": context_text,
            },
            message=f"命中 {len(hits)} 个话题模块, {len(all_wikis)} 条历史记录",
        )

    # ============================================================
    # 写入接口
    # ============================================================

    def write(self, workspace_id: str, query: str, answer: str,
              knowledge: List[str], intent: str = "knowledge_query",
              match_query: str = None) -> Dict[str, Any]:
        """
        将一轮对话写入 wiki：
        1. 判断归属模块（同话题 or 新话题）
        2. 写入 wiki_record
        3. 更新模块 summary + 向量
        
        Args:
            query: 用户原话，写入 wiki 存储（展示用）
            match_query: 改写后的问题，用于话题归属的向量匹配（可选，默认用 query）
        
        Returns: {"wiki_id": str, "module_id": str, "turn_number": int}
        """
        # 1. 确定归属模块（用 match_query 做话题匹配，语义更完整）
        effective_query = match_query or query
        module_id = self._get_or_create_module(workspace_id, effective_query, is_followup=(intent == "followup_query"))

        # 2. 计算 turn_number
        turn_number = self.mysql.get_turn_count(workspace_id) + 1

        # 3. 写入 wiki_record
        wiki_id = self.mysql.create_wiki(
            workspace_id=workspace_id,
            module_id=module_id,
            turn_number=turn_number,
            query=query,
            answer=answer,
            knowledge=knowledge,
        )

        # 4. 更新模块摘要 + 向量
        self._update_module_summary(module_id)

        logger.info("Wiki写入: wiki_id=%s, module_id=%s, turn=%d", wiki_id, module_id, turn_number)
        return {"wiki_id": wiki_id, "module_id": module_id, "turn_number": turn_number}

    # ============================================================
    # 查询接口
    # ============================================================

    def get_modules(self, workspace_id: str) -> List[Dict[str, Any]]:
        """获取 workspace 下的所有话题模块"""
        return self.mysql.get_modules_by_workspace(workspace_id)

    def get_wikis_by_module(self, module_id: str) -> List[Dict[str, Any]]:
        return self.mysql.get_wikis_by_module(module_id)

    def get_all_wikis(self, workspace_id: str) -> List[Dict[str, Any]]:
        return self.mysql.get_wikis_by_workspace(workspace_id)

    def get_recent_context(self, workspace_id: str, n: int = 15) -> str:
        """获取最近 N 轮完整对话（含闲聊），从 chat_log 读取，供问题改写和闲聊回忆使用"""
        logs = self.mysql.get_recent_chat_log(workspace_id, n)
        if not logs:
            # fallback: 从 wiki_record 取
            wikis = self.mysql.get_wikis_by_workspace(workspace_id)
            if not wikis:
                return ""
            last_n = wikis[-n:]
            return "\n".join(f"Q: {w['query']}\nA: {w['answer'][:150]}" for w in last_n)
        return "\n".join(f"Q: {l['query']}\nA: {l['answer'][:150]}" for l in logs)

    def log_chat(self, workspace_id: str, query: str, answer: str, intent: str) -> None:
        """记录一轮对话到 chat_log（含闲聊），供问题改写参考使用"""
        try:
            turn = self.mysql.get_chat_log_count(workspace_id) + 1
            self.mysql.log_chat(
                workspace_id=workspace_id,
                turn_number=turn,
                query=query,
                answer=answer,
                intent=intent,
            )
        except Exception as e:
            logger.warning("log_chat 写入失败: %s", e)

    # ============================================================
    # 内部方法
    # ============================================================

    def _get_or_create_module(self, workspace_id: str, query: str, is_followup: bool = False) -> str:
        """
        判断归属模块：用向量相似度在所有历史模块中找最匹配的。
        1. 向量化 query，在 Milvus 中搜索所有模块
        2. 如果最高相似度 > topic_merge_threshold → 用 LLM 确认是否归入
        3. LLM 确认 same → 归入该模块
        4. 否则创建新模块
        """
        modules = self.mysql.get_modules_by_workspace(workspace_id)

        if not modules:
            return self._create_new_module(workspace_id, query)

        # 向量检索所有模块，找最相似的
        try:
            query_vec = self.embedding.embed(query)
            hits = self.milvus.search_modules(
                workspace_id=workspace_id,
                query_embedding=query_vec,
                top_k=5,
            )
        except Exception as e:
            logger.warning("话题归属向量检索失败: %s，fallback到最新模块", e)
            latest = self.mysql.get_latest_module(workspace_id)
            if latest:
                return latest["module_id"]
            return self._create_new_module(workspace_id, query)

        if not hits:
            return self._create_new_module(workspace_id, query)

        # 取相似度最高的模块
        best_hit = hits[0]
        best_score = best_hit["score"]
        topic_merge_threshold = 0.55  # 向量相似度门槛：低于此值直接创建新模块

        if best_score < topic_merge_threshold:
            logger.info("话题判断: 最高相似度 %.2f < %.2f，创建新模块", best_score, topic_merge_threshold)
            return self._create_new_module(workspace_id, query)

        # 相似度足够高（>= 0.65），用 LLM 精确判断
        is_same = self._judge_same_topic(
            current_topic=best_hit["topic"],
            current_summary=best_hit.get("summary") or best_hit["topic"],
            user_query=query,
        )

        if is_same:
            logger.info("话题判断: 归入模块 %s (topic=%s, score=%.2f)",
                       best_hit["module_id"][:8], best_hit["topic"], best_score)
            return best_hit["module_id"]

        # LLM 判 new，但如果相似度非常高（> 0.85）仍然归入（LLM可能判错）
        if best_score > 0.85:
            logger.info("话题判断: LLM判new但相似度 %.2f > 0.85，强制归入模块 %s",
                       best_score, best_hit["module_id"][:8])
            return best_hit["module_id"]

        # 检查第二、第三命中是否有更合适的（避免只看 top1）
        for hit in hits[1:3]:
            if hit["score"] < topic_merge_threshold:
                break
            is_same_alt = self._judge_same_topic(
                current_topic=hit["topic"],
                current_summary=hit.get("summary") or hit["topic"],
                user_query=query,
            )
            if is_same_alt:
                logger.info("话题判断: 归入备选模块 %s (topic=%s, score=%.2f)",
                           hit["module_id"][:8], hit["topic"], hit["score"])
                return hit["module_id"]

        return self._create_new_module(workspace_id, query)

    def _create_new_module(self, workspace_id: str, first_query: str) -> str:
        topic = first_query[:20].replace("\n", " ")
        module_id = self.mysql.create_module(workspace_id, topic=topic, summary="")
        try:
            vec = self.embedding.embed(topic)
            self.milvus.upsert_module(
                module_id=module_id, workspace_id=workspace_id,
                topic=topic, summary="", embedding=vec,
            )
        except Exception as e:
            logger.warning("初始模块向量创建失败: %s", e)
        logger.info("创建新模块: id=%s, topic=%s", module_id, topic)
        return module_id

    def _judge_same_topic(self, current_topic: str, current_summary: str, user_query: str) -> bool:
        """
        纯 LLM 判断（向量相似度已在 _get_or_create_module 中处理）
        """
        prompt = TOPIC_JUDGE_PROMPT.format(
            current_topic=current_topic,
            current_summary=current_summary[:500],
            user_query=user_query,
        )
        try:
            out = self.llm.chat(prompt, max_tokens=500, temperature=0.0)
            last_line = out.strip().splitlines()[-1].strip().lower() if out.strip() else ""
            return "same" in last_line
        except Exception as e:
            logger.warning("话题判断LLM调用失败: %s", e)
            return True  # 失败时保守归到当前模块

    def _update_module_summary(self, module_id: str) -> None:
        module = self.mysql.get_module(module_id)
        if not module:
            return
        wikis = self.mysql.get_wikis_by_module(module_id)
        if not wikis:
            return

        conversations = "\n".join(f"Q: {w['query']}\nA: {w['answer'][:200]}" for w in wikis)
        prompt = MODULE_SUMMARY_PROMPT.format(conversations=conversations)

        try:
            data = self.llm.chat_json(prompt, max_tokens=2048)
            topic = (data.get("topic") or module["topic"])[:256]
            summary = (data.get("summary") or "")[:2048]
        except Exception as e:
            logger.warning("模块摘要生成失败: %s", e)
            topic = module["topic"]
            summary = f"讨论了{len(wikis)}轮关于{topic}的问题"

        self.mysql.update_module(module_id, topic=topic, summary=summary, wiki_count=len(wikis))

        embed_text = f"{topic}。{summary}"
        try:
            vec = self.embedding.embed(embed_text)
            self.milvus.upsert_module(
                module_id=module_id, workspace_id=module["workspace_id"],
                topic=topic, summary=summary, embedding=vec,
            )
        except Exception as e:
            logger.error("模块向量更新失败: %s", e)

    def _format_context(self, wikis: List[Dict[str, Any]]) -> str:
        if not wikis:
            return ""
        parts = []
        current_module = None
        for w in wikis:
            module_topic = w.get("_module_topic", "")
            if module_topic != current_module:
                current_module = module_topic
                score = w.get("_module_score", 0)
                parts.append(f"\n--- 话题: {module_topic} (相关度: {score:.2f}) ---")
            knowledge_str = ""
            knowledge = w.get("knowledge") or []
            if knowledge:
                knowledge_str = "\n  参考知识: " + " | ".join(str(k)[:100] for k in knowledge[:3])
            parts.append(
                f"  [第{w['turn_number']}轮] Q: {w['query']}\n"
                f"  A: {w['answer'][:300]}"
                f"{knowledge_str}"
            )
        return "\n".join(parts)
