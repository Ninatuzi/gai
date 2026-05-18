"""
Skill: 问题改写（指代消解 + 上下文增强）
两种模式：
1. followup_query: 经典指代消解，补全代词和省略
2. knowledge_query: 上下文增强，如果问题涉及前文概念则补充关键词

例如：
  [followup] 上下文: Q:硅碳负极和石墨负极哪个容量高? A:硅碳负极...
             用户问: 它的体积膨胀怎么解决？
             改写后: 硅碳负极的体积膨胀问题怎么解决？

  [knowledge] 上下文: 讨论了NCM811的能量密度、快充对电池影响
              用户问: 快充下会不会加剧析锂？
              改写后: NCM811在快充条件下会不会加剧析锂现象？
"""

import logging

from chatwiki.config.prompts import QUERY_REWRITE_PROMPT, QUERY_ENHANCE_PROMPT
from chatwiki.models import LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


class QueryRewriteSkill(BaseSkill):
    """
    问题改写 Skill（指代消解 + 上下文增强）
    输入 context 中字段：
        - recent_context: str  最近几轮对话文本（供改写参考）
        - intent: str  当前意图（followup_query / knowledge_query）
    输出 data：
        - rewritten_query: str  改写后的问题
    """

    def __init__(self, llm: LLMClient = None):
        self._llm = llm

    @property
    def name(self) -> str:
        return "query_rewrite"

    @property
    def description(self) -> str:
        return "对问题进行改写：followup 做指代消解，knowledge 做上下文增强"

    def run(self, skill_input: SkillInput) -> SkillOutput:
        query = skill_input.query
        recent_context = skill_input.context.get("recent_context", "")
        intent = skill_input.context.get("intent", "knowledge_query")

        # 如果没有 LLM 或没有上下文，直接返回原问题
        if not self._llm or not recent_context:
            return SkillOutput(
                success=True,
                data={"rewritten_query": query},
                message="无上下文，未改写",
            )

        if intent == "followup_query":
            # followup: 一定需要改写（因为有指代/省略）
            return self._rewrite_followup(query, recent_context)
        else:
            # knowledge_query: 只在有上下文关联时做增强
            return self._enhance_knowledge(query, recent_context)

    def _rewrite_followup(self, query: str, recent_context: str) -> SkillOutput:
        """followup_query 的指代消解"""
        prompt = QUERY_REWRITE_PROMPT.format(
            context=recent_context[-2000:],
            user_query=query,
        )
        try:
            rewritten = self._llm.chat(prompt, max_tokens=500, temperature=0.0)
            rewritten = rewritten.strip()
            if rewritten and len(rewritten) >= 3 and rewritten != query:
                logger.info("问题改写: %r -> %r", query[:30], rewritten[:30])
                return SkillOutput(
                    success=True,
                    data={"rewritten_query": rewritten},
                    message=f"已改写: {rewritten[:50]}",
                )
        except Exception as e:
            logger.warning("问题改写失败: %s", e)

        return SkillOutput(
            success=True,
            data={"rewritten_query": query},
            message="改写失败，使用原问题",
        )

    def _enhance_knowledge(self, query: str, recent_context: str) -> SkillOutput:
        """
        knowledge_query 的上下文增强：
        判断问题是否涉及前文讨论过的概念，如果是则补充关键信息提升检索效果
        """
        # 快速判断：如果问题已经很长很完整（> 20字且不含代词），大概率不需要增强
        if len(query) > 20 and not self._has_context_dependency(query):
            return SkillOutput(
                success=True,
                data={"rewritten_query": query},
                message="问题已完整，无需增强",
            )

        # 有可能需要增强，调用 LLM
        prompt = QUERY_ENHANCE_PROMPT.format(
            context=recent_context[-2000:],
            user_query=query,
        )
        try:
            enhanced = self._llm.chat(prompt, max_tokens=500, temperature=0.0)
            enhanced = enhanced.strip()
            if enhanced and len(enhanced) >= 3:
                # 如果 LLM 返回的跟原问题一样，说明不需要增强
                if enhanced == query or enhanced.rstrip("？?。.") == query.rstrip("？?。."):
                    return SkillOutput(
                        success=True,
                        data={"rewritten_query": query},
                        message="LLM判断无需增强",
                    )
                logger.info("上下文增强: %r -> %r", query[:30], enhanced[:30])
                return SkillOutput(
                    success=True,
                    data={"rewritten_query": enhanced},
                    message=f"已增强: {enhanced[:50]}",
                )
        except Exception as e:
            logger.warning("上下文增强失败: %s", e)

        return SkillOutput(
            success=True,
            data={"rewritten_query": query},
            message="增强失败，使用原问题",
        )

    @staticmethod
    def _has_context_dependency(query: str) -> bool:
        """判断问题是否有上下文依赖（代词、指示词、省略等）"""
        indicators = [
            "它", "他", "她", "它们", "他们", "这个", "那个", "这些", "那些",
            "上面", "前面", "刚才", "之前", "上述", "上面说的", "前面提到",
        ]
        for p in indicators:
            if p in query:
                return True
        # 问题太短（可能省略了主语）
        if len(query) <= 10:
            return True
        return False
