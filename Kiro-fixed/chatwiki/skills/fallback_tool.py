"""
Skill: 保底直答
当其他途径都无法提供有效信息时，LLM 直接回答
也用于处理闲聊意图
闲聊时带上最近对话记录，让 LLM 能回忆之前聊过的内容
"""

import logging

from chatwiki.models import LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)

FALLBACK_PROMPT_WITH_CONTEXT = """根据对话记录回答用户的问题。

## 最近的对话记录
{chat_history}

## 用户新问题
{user_query}

## 要求
1. 如果用户问的内容在对话记录中有提到，直接引用回答
2. 如果是闲聊/问候，自然回应，给出有趣或有用的新内容
3. 不要重复你上一轮已经说过的回答，要给出不同的内容
4. 使用中文，简洁有条理"""

FALLBACK_PROMPT_NO_CONTEXT = """直接回答用户的问题。

## 用户问题
{user_query}

## 要求
1. 如果是闲聊/问候，自然回应
2. 如果是知识性问题，基于你的知识回答
3. 使用中文，简洁有条理"""


class FallbackSkill(BaseSkill):
    """
    保底直答 Skill
    输入 context 中可选字段：
        - chat_history: str  最近对话记录（供回忆参考）
    输出 data:
        - answer: str
    """

    def __init__(self, llm: LLMClient):
        self.llm = llm

    @property
    def name(self) -> str:
        return "fallback_direct_answer"

    @property
    def description(self) -> str:
        return "保底机制：由LLM直接回答（带最近对话记录，可回忆之前聊的内容）"

    def run(self, skill_input: SkillInput) -> SkillOutput:
        query = skill_input.query
        chat_history = skill_input.context.get("chat_history", "")

        if chat_history:
            prompt = FALLBACK_PROMPT_WITH_CONTEXT.format(
                chat_history=chat_history,
                user_query=query,
            )
        else:
            prompt = FALLBACK_PROMPT_NO_CONTEXT.format(user_query=query)

        try:
            answer = self.llm.chat(prompt, max_tokens=1024).strip()
            if not answer:
                answer = "你好！有什么我可以帮助你的吗？"
        except Exception as e:
            logger.error("保底回答失败: %s", e)
            answer = "抱歉，我暂时无法回答。"
        return SkillOutput(success=True, data={"answer": answer}, message="保底直答")
