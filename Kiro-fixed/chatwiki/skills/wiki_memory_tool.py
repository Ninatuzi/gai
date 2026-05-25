"""
Skill: History Memory（Wiki）
完整的历史记忆技能：
- Module 话题分组管理（第一级检索）
- Wiki summary 向量精筛（第二级检索）
- 返回命中 wiki 的 knowledge 字段 + summary
- 写入新的 wiki 记录（含 summary 向量化）
"""

import logging
from typing import Any, Dict, List, Optional

from chatwiki.config import get_settings
from chatwiki.config.prompts import MODULE_SUMMARY_PROMPT, TOPIC_JUDGE_PROMPT, WIKI_SUMMARY_PROMPT
from chatwiki.models import EmbeddingClient, LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput
from chatwiki.storage import MilvusClient, MySQLClient

logger = logging.getLogger(__name__)


class WikiMemorySkill(BaseSkill):
    """
    History Memory Skill（两级检索版）
    
    检索模式 (run):
        第一级: 向量搜索 Module（找相关话题模块）
        第二级: 在命中 Module 内向量搜索 Wiki summary（找最相关的具体 wiki）
        返回: 命中 wiki 的 knowledge + summary + answer

    写入模式 (write):
        生成 wiki summary → 向量化存入 Milvus → 写入 JSON → 更新模块摘要
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
        return "两级检索：Module级找话题 → Wiki级找具体知识点，返回精确的 knowledge 片段"

    # ============================================================
    # Skill 标准接口：两级检索
    # ============================================================

    def run(self, skill_input: SkillInput) -> SkillOutput:
        """
        两级检索：
        1. Module 向量检索 → 命中相关模块
        2. Wiki summary 向量检索 → 在命中模块内找最相关的 wiki
        3. 拿出命中 wiki 的 knowledge + summary + answer
        """
        workspace_id = skill_input.workspace_id
        query = skill_input.query
        top_k = self.settings.retrieval.top_k
        threshold = self.settings.retrieval.similarity_threshold

        if not workspace_id:
            return SkillOutput(success=False, message="缺少 workspace_id")

        # 1. 向量化 query
        try:
            query_vec = self.embedding.embed(query)
        except Exception as e:
            logger.error("Wiki检索向量化失败: %s", e)
            return SkillOutput(success=True, data={"found": False, "wikis": [], "context_text": ""})

        # 2. 第一级：搜索相关 module
        module_hits = self.milvus.search_modules(
            workspace_id=workspace_id, query_embedding=query_vec, top_k=top_k
        )
        module_hits = [h for h in module_hits if h["score"] >= threshold]

        if not module_hits:
            logger.info("Wiki检索: 无命中模块 (ws=%s, query=%r)", workspace_id, query[:30])
            return SkillOutput(success=True, data={
                "found": False, "modules_hit": 0, "wikis": [], "context_text": ""
            })

        # 2.5 动态截断：如果第一名遥遥领先，只保留得分接近的模块
        module_hits = self._dynamic_truncate_modules(module_hits)

        # 3. 第二级：在命中模块内搜索 wiki summaries
        module_ids = [h["module_id"] for h in module_hits]
        wiki_hits = self.milvus.search_wiki_summaries(
            module_ids=module_ids,
            query_embedding=query_vec,
            top_k=5,  # 全局取 top5 最相关的 wiki
        )

        if not wiki_hits:
            # fallback: 如果 wiki_summaries 里没数据（可能是旧数据），用老方式全拿
            logger.info("Wiki检索: wiki_summaries 无命中，fallback 到全量获取")
            return self._fallback_run(module_hits, query_vec)

        # 4. 根据命中的 wiki_id 拿出完整 wiki 记录
        all_wikis: List[Dict[str, Any]] = []
        for wh in wiki_hits:
            wiki_record = self._get_wiki_by_id(wh["wiki_id"], wh["module_id"])
            if wiki_record:
                wiki_record["_wiki_score"] = wh["score"]
                wiki_record["_wiki_summary"] = wh["summary"]
                # 找到对应 module 的 topic
                for mh in module_hits:
                    if mh["module_id"] == wh["module_id"]:
                        wiki_record["_module_topic"] = mh["topic"]
                        wiki_record["_module_score"] = mh["score"]
                        break
                # 解析 knowledge JSON
                if isinstance(wiki_record.get("knowledge"), str):
                    import json
                    try:
                        wiki_record["knowledge"] = json.loads(wiki_record["knowledge"])
                    except Exception:
                        wiki_record["knowledge"] = []
                all_wikis.append(wiki_record)

        # 5. 格式化上下文（用 knowledge + summary，而不是完整 answer）
        context_text = self._format_context(all_wikis)

        logger.info("Wiki检索(两级): 命中 %d 模块, 精筛 %d 条wiki", len(module_hits), len(all_wikis))
        return SkillOutput(
            success=True,
            data={
                "found": True,
                "modules_hit": len(module_hits),
                "wikis": all_wikis,
                "wiki_count": len(all_wikis),
                "context_text": context_text,
            },
            message=f"命中 {len(module_hits)} 个模块, 精筛 {len(all_wikis)} 条wiki",
        )

    def _fallback_run(self, module_hits, query_vec) -> SkillOutput:
        """fallback: wiki_summaries 为空时用老方式（全量获取模块下所有 wiki）"""
        all_wikis: List[Dict[str, Any]] = []
        for h in module_hits:
            module_wikis = self.mysql.get_wikis_by_module(h["module_id"])
            for w in module_wikis:
                w["_module_topic"] = h["topic"]
                w["_module_score"] = h["score"]
                if isinstance(w.get("knowledge"), str):
                    import json
                    try:
                        w["knowledge"] = json.loads(w["knowledge"])
                    except Exception:
                        w["knowledge"] = []
            all_wikis.extend(module_wikis)

        context_text = self._format_context(all_wikis)
        return SkillOutput(
            success=True,
            data={
                "found": bool(all_wikis),
                "modules_hit": len(module_hits),
                "wikis": all_wikis,
                "wiki_count": len(all_wikis),
                "context_text": context_text,
            },
        )

    def _get_wiki_by_id(self, wiki_id: str, module_id: str) -> Optional[Dict[str, Any]]:
        """根据 wiki_id 从 module 的 wikis 列表中查找"""
        wikis = self.mysql.get_wikis_by_module(module_id)
        for w in wikis:
            if w.get("wiki_id") == wiki_id:
                return w
        return None

    # ============================================================
    # 写入接口
    # ============================================================

    def write(self, workspace_id: str, query: str, answer: str,
              knowledge: List[str], intent: str = "knowledge_query",
              match_query: str = None) -> Dict[str, Any]:
        """
        将一轮对话写入 wiki：
        1. 判断归属模块
        2. 生成 wiki summary（80-120字）
        3. 写入 wiki_record
        4. 向量化 summary 存入 Milvus wiki_summaries
        5. 更新模块 summary + 向量
        
        Returns: {"wiki_id": str, "module_id": str, "turn_number": int, "wiki_summary": str}
        """
        # 1. 确定归属模块
        effective_query = match_query or query
        module_id = self._get_or_create_module(workspace_id, effective_query, is_followup=(intent == "followup_query"))

        # 2. 计算 turn_number
        turn_number = self.mysql.get_turn_count(workspace_id) + 1

        # 2.5 过滤 answer 中的 <think> 标签（DeepSeek 输出可能带思考过程）
        import re
        clean_answer = re.sub(r'<think>[\s\S]*?</think>', '', answer, flags=re.DOTALL).strip()
        # 兜底：如果没有闭合的 </think>，去掉 <think> 及之后的所有内容直到正文开始
        if '<think>' in clean_answer:
            clean_answer = re.sub(r'<think>[\s\S]*', '', clean_answer).strip()
        # 如果过滤后为空（整个回答都是 think），保留原文
        if not clean_answer:
            clean_answer = answer

        # 3. 生成 wiki summary（80-120字）
        wiki_summary = ""
        try:
            summary_prompt = WIKI_SUMMARY_PROMPT.format(
                query=query,
                answer=clean_answer[:500],
            )
            wiki_summary = self.llm.chat(summary_prompt, max_tokens=500, temperature=0.0).strip()
            # 过滤 summary 中的 <think> 标签
            wiki_summary = re.sub(r'<think>[\s\S]*?</think>', '', wiki_summary, flags=re.DOTALL).strip()
            if '<think>' in wiki_summary:
                wiki_summary = re.sub(r'<think>[\s\S]*', '', wiki_summary).strip()
            # 限制长度 150 字（允许超出 120 一点）
            wiki_summary = wiki_summary[:150]
            # 如果过滤后为空（LLM 整段输出都是 think），用兜底摘要
            if not wiki_summary:
                wiki_summary = f"{query}。{clean_answer[:80]}" if clean_answer else query[:100]
        except Exception as e:
            logger.warning("Wiki摘要生成失败: %s", e)
            wiki_summary = f"{query}。{clean_answer[:80]}" if clean_answer else query[:100]

        # 4. 写入 wiki_record
        wiki_id = self.mysql.create_wiki(
            workspace_id=workspace_id,
            module_id=module_id,
            turn_number=turn_number,
            query=query,
            answer=clean_answer,
            knowledge=knowledge,
            summary=wiki_summary,
        )

        # 5. 向量化 summary 存入 Milvus wiki_summaries
        try:
            summary_vec = self.embedding.embed(wiki_summary)
            self.milvus.upsert_wiki_summary(
                wiki_id=wiki_id,
                workspace_id=workspace_id,
                module_id=module_id,
                summary=wiki_summary,
                embedding=summary_vec,
            )
        except Exception as e:
            logger.warning("Wiki summary 向量存储失败: %s", e)

        # 6. 更新模块摘要 + 向量
        self._update_module_summary(module_id)

        logger.info("Wiki写入: wiki_id=%s, module_id=%s, turn=%d, summary=%r",
                   wiki_id, module_id, turn_number, wiki_summary[:40])
        return {"wiki_id": wiki_id, "module_id": module_id, "turn_number": turn_number,
                "wiki_summary": wiki_summary}

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
        """获取最近 N 轮完整对话（含闲聊），从 chat_log 读取"""
        logs = self.mysql.get_recent_chat_log(workspace_id, n)
        if not logs:
            wikis = self.mysql.get_wikis_by_workspace(workspace_id)
            if not wikis:
                return ""
            last_n = wikis[-n:]
            return "\n".join(f"Q: {w['query']}\nA: {w['answer'][:150]}" for w in last_n)
        return "\n".join(f"Q: {l['query']}\nA: {l['answer'][:150]}" for l in logs)

    def log_chat(self, workspace_id: str, query: str, answer: str, intent: str) -> None:
        """记录一轮对话到 chat_log（含闲聊）"""
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
        """判断归属模块：向量搜索所有历史模块找最匹配的"""
        modules = self.mysql.get_modules_by_workspace(workspace_id)

        if not modules:
            return self._create_new_module(workspace_id, query)

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

        best_hit = hits[0]
        best_score = best_hit["score"]
        topic_merge_threshold = 0.55

        if best_score < topic_merge_threshold:
            logger.info("话题判断: 最高相似度 %.2f < %.2f，创建新模块", best_score, topic_merge_threshold)
            return self._create_new_module(workspace_id, query)

        is_same = self._judge_same_topic(
            current_topic=best_hit["topic"],
            current_summary=best_hit.get("summary") or best_hit["topic"],
            user_query=query,
        )

        if is_same:
            logger.info("话题判断: 归入模块 %s (topic=%s, score=%.2f)",
                       best_hit["module_id"][:8], best_hit["topic"], best_score)
            return best_hit["module_id"]

        if best_score > 0.85:
            logger.info("话题判断: LLM判new但相似度 %.2f > 0.85，强制归入模块 %s",
                       best_score, best_hit["module_id"][:8])
            return best_hit["module_id"]

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

    def _dynamic_truncate_modules(self, module_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        动态截断模块召回列表：
        - 如果第一名远超其他模块（分差 > score_gap_threshold），只保留第一名
        - 否则保留所有得分与第一名差距在 score_gap_threshold 以内的模块
        
        避免同领域弱相关模块被大量召回，减少第二级搜索噪音。
        """
        if len(module_hits) <= 1:
            return module_hits

        gap_threshold = self.settings.retrieval.score_gap_threshold
        best_score = module_hits[0]["score"]

        # 保留与第一名得分差距在阈值内的模块
        truncated = [h for h in module_hits if (best_score - h["score"]) <= gap_threshold]

        if len(truncated) < len(module_hits):
            logger.info(
                "模块动态截断: %d → %d (best=%.2f, gap_threshold=%.2f, 淘汰: %s)",
                len(module_hits), len(truncated), best_score, gap_threshold,
                ", ".join(f"{h['topic']}({h['score']:.2f})" for h in module_hits if h not in truncated),
            )

        return truncated

    def _judge_same_topic(self, current_topic: str, current_summary: str, user_query: str) -> bool:
        """纯 LLM 判断"""
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
            return True

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
        """格式化检索结果：优先展示 knowledge，辅以 summary"""
        if not wikis:
            return ""
        parts = []
        current_module = None
        for w in wikis:
            module_topic = w.get("_module_topic", "")
            if module_topic != current_module:
                current_module = module_topic
                score = w.get("_module_score", 0)
                parts.append(f"\n--- 话题: {module_topic} (模块相关度: {score:.2f}) ---")

            wiki_score = w.get("_wiki_score", 0)
            wiki_summary = w.get("_wiki_summary") or w.get("summary", "")
            parts.append(f"  [第{w['turn_number']}轮 | wiki相关度: {wiki_score:.2f}]")
            parts.append(f"  问题: {w['query']}")
            if wiki_summary:
                parts.append(f"  摘要: {wiki_summary}")

            # 展示 knowledge（核心价值）
            knowledge = w.get("knowledge") or []
            if knowledge:
                parts.append("  相关知识:")
                for k in knowledge[:3]:
                    parts.append(f"    - {str(k)[:200]}")

            # 兜底：如果没有 knowledge，展示 answer 片段
            if not knowledge:
                parts.append(f"  回答: {w['answer'][:300]}")
            parts.append("")

        return "\n".join(parts)
