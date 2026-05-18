"""
Skill: 答案聚合
根据 Wiki + RAG 的检索结果综合生成最终回答
"""

import logging

from chatwiki.config.prompts import ANSWER_AGGREGATE_PROMPT, ANSWER_DIRECT_PROMPT
from chatwiki.models import LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


class AggregatorSkill(BaseSkill):
    """
    答案聚合 Skill
    输入 context 中字段：
        - wiki_context: str  Wiki 检索的格式化文本
        - rag_context: str   RAG 检索的格式化文本
    输出 data:
        - answer: str  最终回答
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @property
    def name(self) -> str:
        return "answer_aggregation"

    @property
    def description(self) -> str:
        return "综合 Wiki 历史记忆和 RAG 知识库检索结果，生成最终回答"

    def run(self, skill_input: SkillInput) -> SkillOutput:
        query = skill_input.query
        wiki_context = skill_input.context.get("wiki_context", "")
        rag_context = skill_input.context.get("rag_context", "")

        has_wiki = bool(wiki_context)
        has_rag = bool(rag_context)

        if not has_wiki and not has_rag:
            # 都没有，直接回答
            answer = self._direct_answer(query)
        else:
            prompt = ANSWER_AGGREGATE_PROMPT.format(
                wiki_context=wiki_context or "（无）",
                rag_context=rag_context or "（无）",
                user_query=query,
            )
            try:
                answer = self.llm.chat(prompt, max_tokens=2048)
                answer = answer.strip() if answer else "抱歉，我暂时无法回答。"
            except Exception as e:
                logger.error("答案聚合失败: %s", e)
                answer = f"抱歉，生成回答时出错: {e}"

        return SkillOutput(success=True, data={"answer": answer}, message="回答已生成")

    def _direct_answer(self, query: str) -> str:
        prompt = ANSWER_DIRECT_PROMPT.format(user_query=query)
        try:
            return self.llm.chat(prompt, max_tokens=1024).strip()
        except Exception as e:
            return f"抱歉，回答失败: {e}"
