"""
Skill: 意图识别
判断用户问题类型：knowledge_query / followup_query / chitchat
"""

import logging
from typing import List, Optional

from chatwiki.config.prompts import INTENT_PROMPT
from chatwiki.models import LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


class IntentSkill(BaseSkill):
    """
    意图识别 Skill
    输入 context 中可选字段：
        - recent_modules: List[str]  最近的话题模块名列表
    输出 data：
        - intent: str  ("knowledge_query" / "followup_query" / "chitchat")
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @property
    def name(self) -> str:
        return "intent_recognition"

    @property
    def description(self) -> str:
        return "识别用户问题的意图类型：知识查询、追问、闲聊"

    def run(self, skill_input: SkillInput) -> SkillOutput:
        query = skill_input.query
        recent_modules = skill_input.context.get("recent_modules", [])

        modules_text = ", ".join(recent_modules) if recent_modules else "（暂无历史话题）"
        prompt = INTENT_PROMPT.format(recent_modules=modules_text, user_query=query)

        try:
            out = self.llm.chat(prompt, max_tokens=500, temperature=0.0)
            intent = self._parse_intent(out)
            logger.info("意图识别: query=%r -> %s", query[:30], intent)
            return SkillOutput(success=True, data={"intent": intent}, message=f"意图: {intent}")
        except Exception as e:
            logger.warning("意图识别失败: %s", e)
            return SkillOutput(success=True, data={"intent": "knowledge_query"}, message="意图识别失败，默认knowledge_query")

    @staticmethod
    def _parse_intent(text: str) -> str:
        text_lower = text.strip().lower()
        last_line = text_lower.splitlines()[-1].strip() if text_lower.splitlines() else text_lower
        for intent in ["followup_query", "knowledge_query", "chitchat"]:
            if intent in last_line:
                return intent
        for intent in ["followup_query", "knowledge_query", "chitchat"]:
            if intent in text_lower:
                return intent
        return "knowledge_query"
