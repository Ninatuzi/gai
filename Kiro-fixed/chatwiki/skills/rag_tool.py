"""
Skill: RAG 检索（Mock）
后续对接公司真实 RAG 系统时只需修改 run() 内部实现
"""

import logging
from typing import Any, Dict, List

from chatwiki.skills.base import BaseSkill, SkillInput, SkillOutput

logger = logging.getLogger(__name__)


# Mock RAG 数据
MOCK_RAG_DATA = [
    {
        "id": "rag_001",
        "content": "析锂（Lithium Plating）是指锂离子在充电过程中未能正常嵌入负极石墨层间，"
                   "而是在负极表面以金属锂的形式沉积的现象。主要发生在低温快充、过充、"
                   "负极容量不足等条件下。析锂会导致电池容量衰减、内阻增大，严重时可能刺穿隔膜引发短路。",
        "source": "锂电池技术手册 Ch.5",
        "keywords": ["析锂", "lithium plating", "负极", "低温", "快充"],
    },
    {
        "id": "rag_002",
        "content": "磷酸铁锂(LiFePO4/LFP)正极材料的理论比容量为170mAh/g，实际比容量约140-160mAh/g。"
                   "工作电压平台为3.2V。热分解温度约600℃，具有优异的热稳定性和安全性。"
                   "循环寿命可达2000-6000次（80%容量保持率）。",
        "source": "正极材料技术规格书",
        "keywords": ["磷酸铁锂", "LFP", "比容量", "循环寿命", "热稳定性"],
    },
    {
        "id": "rag_003",
        "content": "三元材料(NCM811)的理论比容量约276mAh/g，实际可达200mAh/g以上。"
                   "工作电压3.6-3.7V。能量密度高但热稳定性相对较差，热分解温度约200℃。"
                   "循环寿命约800-1500次。高镍正极对湿度敏感，需在干燥环境下加工。",
        "source": "正极材料技术规格书",
        "keywords": ["三元材料", "NCM811", "能量密度", "高镍", "热稳定性"],
    },
    {
        "id": "rag_004",
        "content": "电解液中LiPF6在高温(>60℃)下会分解生成HF和PF5，HF会腐蚀正极材料和SEI膜。"
                   "常用添加剂包括：VC(碳酸亚乙烯酯)用于稳定SEI膜，FEC(氟代碳酸乙烯酯)用于硅基负极，"
                   "DTD(硫酸乙烯酯)用于高压正极保护。",
        "source": "电解液配方手册",
        "keywords": ["电解液", "LiPF6", "添加剂", "VC", "FEC", "SEI"],
    },
    {
        "id": "rag_005",
        "content": "BMS均衡策略分为被动均衡和主动均衡。被动均衡通过电阻放电将高电压电芯能量消耗，"
                   "电路简单但能量损耗大。主动均衡通过电感/电容/变压器将高电芯能量转移给低电芯，"
                   "效率高但电路复杂成本高。",
        "source": "BMS设计规范",
        "keywords": ["BMS", "均衡", "被动均衡", "主动均衡", "电芯"],
    },
]


class RAGSkill(BaseSkill):
    """
    RAG 检索 Skill（Mock）
    输出 data:
        - found: bool
        - chunks: List[Dict]  RAG 片段
        - context_text: str  格式化文本
    
    TODO: 对接真实 RAG 时替换 run() 内部实现
    """

    @property
    def name(self) -> str:
        return "rag_search"

    @property
    def description(self) -> str:
        return "从公司知识库（RAG）中检索专业文档片段"

    def run(self, skill_input: SkillInput) -> SkillOutput:
        """
        执行 RAG 检索
        TODO: 对接真实 RAG：
            resp = requests.post("http://rag-api/search", json={"query": query, "top_k": 3})
            chunks = resp.json()["results"]
        """
        query = skill_input.query
        query_lower = query.lower()

        # Mock: 关键词匹配
        scored = []
        for chunk in MOCK_RAG_DATA:
            score = sum(1 for kw in chunk["keywords"] if kw.lower() in query_lower or kw in query)
            if score > 0:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [c for _, c in scored[:3]]

        if not results:
            return SkillOutput(success=True, data={"found": False, "chunks": [], "context_text": ""})

        context_text = "\n\n".join(
            f"[RAG|{c['source']}] {c['content']}" for c in results
        )
        logger.info("RAG检索: 命中 %d 条", len(results))
        return SkillOutput(
            success=True,
            data={"found": True, "chunks": results, "context_text": context_text},
            message=f"RAG命中 {len(results)} 条",
        )
