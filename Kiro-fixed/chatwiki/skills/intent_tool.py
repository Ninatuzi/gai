"""
Skill: 意图识别
判断用户问题类型：knowledge_query / followup_query / chitchat
"""

import logging
import re
from typing import List, Optional

from chatwiki.config.prompts import INTENT_PROMPT
from chatwiki.models import LLMClient
from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)

# 专业领域关键词（命中任一即直接判定为 knowledge_query，跳过 LLM）
# 覆盖锂电池/电池/材料/化学等领域的核心术语
KNOWLEDGE_KEYWORDS = [
    # 电池核心概念
    "SEI", "CEI", "析锂", "热失控", "过充", "过放", "内阻", "容量衰减",
    "循环寿命", "库伦效率", "能量密度", "功率密度", "充放电", "倍率",
    # 材料
    "正极", "负极", "电解液", "隔膜", "集流体", "粘结剂", "导电剂",
    "三元材料", "磷酸铁锂", "钴酸锂", "锰酸锂", "石墨", "硅碳",
    "NCM", "NCA", "LFP", "LCO", "LMO",
    # 工艺/现象
    "涂布", "辊压", "注液", "化成", "分容", "老化", "极片", "极耳",
    "电芯", "模组", "PACK", "BMS", "SOC", "SOH", "OCV",
    # 化学/物理
    "电化学", "嵌锂", "脱锂", "扩散系数", "离子电导率", "界面阻抗",
    "枝晶", "固态电解质", "聚合物", "陶瓷隔膜",
]

# 编译正则：不区分大小写匹配
_KNOWLEDGE_PATTERN = re.compile(
    "|".join(re.escape(kw) for kw in KNOWLEDGE_KEYWORDS),
    re.IGNORECASE
)


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
        recent_chat = skill_input.context.get("recent_chat", "")

        # ========== 关键词硬规则：包含专业术语直接判定为 knowledge_query ==========
        if _KNOWLEDGE_PATTERN.search(query):
            matched = _KNOWLEDGE_PATTERN.findall(query)
            logger.info("意图识别(关键词命中): query=%r, matched=%s -> knowledge_query",
                       query[:30], matched[:3])
            return SkillOutput(
                success=True,
                data={"intent": "knowledge_query"},
                message=f"关键词命中: {matched[:3]}",
            )

        # ========== LLM 判断：无明显专业关键词时走 LLM ==========
        modules_text = ", ".join(recent_modules) if recent_modules else "（暂无历史话题）"
        
        # 如果有最近对话记录，附加到 prompt 帮助判断
        chat_context = ""
        if recent_chat:
            chat_context = f"\n\n## 最近几轮对话\n{recent_chat}"
        
        prompt = INTENT_PROMPT.format(recent_modules=modules_text, user_query=query) + chat_context

        try:
            out = self.llm.chat(prompt, max_tokens=500, temperature=0.0)
            intent = self._parse_intent(out)
            logger.info("意图识别(LLM): query=%r -> %s", query[:30], intent)
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
