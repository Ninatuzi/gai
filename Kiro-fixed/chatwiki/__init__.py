"""
ChatWiki - 对话级记忆检索 Agent
基于 Harness/SKILL 架构：
- Harness 负责调度
- 每个 Skill 是独立可插拔的能力单元
- Wiki 格式：Q + A + Knowledge (from RAG)
- Module 话题分组：自动识别话题归属
"""

__version__ = "0.3.0"

from chatwiki.api.interface import ChatWikiAgent

__all__ = ["ChatWikiAgent", "__version__"]
