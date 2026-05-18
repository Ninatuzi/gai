"""
Skill: 问题改写（指代消解）
用 LLM 补全问题中的代词和省略，使问题适合独立检索

例如：
  上下文: Q:硅碳负极和石墨负极哪个容量高? A:硅碳负极...
  用户问: 它的体积膨胀怎么解决？
  改写后: 硅碳负极的体积膨胀问题怎么解决？
"""

import logging

from chatwiki.config.prompts import QUERY_REWRITE_PROMPT
from chatwiki.models import LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


class QueryRewriteSkill(BaseSkill):
    """
    问题改写 Skill（指代消解）
    输入 context 中字段：
        - recent_context: str  最近几轮对话文本（供改写参考）
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
        return "对问题进行改写，补全指代词（它/那个/刚才说的），使问题可独立理解"

    def run(self, skill_input: SkillInput) -> SkillOutput:
        query = skill_input.query
        recent_context = skill_input.context.get("recent_context", "")

        # 如果没有 LLM 或没有上下文，直接返回原问题
        if not self._llm or not recent_context:
            return SkillOutput(
                success=True,
                data={"rewritten_query": query},
                message="无上下文，未改写",
            )

        # 判断是否需要改写（包含代词或省略）
        needs_rewrite = self._needs_rewrite(query)
        if not needs_rewrite:
            return SkillOutput(
                success=True,
                data={"rewritten_query": query},
                message="问题已完整，无需改写",
            )

        # LLM 改写
        prompt = QUERY_REWRITE_PROMPT.format(
            context=recent_context[-2000:],
            user_query=query,
        )
        try:
            rewritten = self._llm.chat(prompt, max_tokens=500, temperature=0.0)
            rewritten = rewritten.strip()
            # 校验：改写结果不能为空或太短
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

    @staticmethod
    def _needs_rewrite(query: str) -> bool:
        """简单判断是否需要改写（包含代词、指示词、省略等）"""
        # 代词
        pronouns = ["它", "他", "她", "它们", "他们", "这个", "那个", "这些", "那些",
                    "上面", "前面", "刚才", "之前", "上述"]
        for p in pronouns:
            if p in query:
                return True
        # 问题太短（可能省略了主语）
        if len(query) <= 8:
            return True
        return False
