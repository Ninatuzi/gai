"""
Skill 基类定义
所有 Skill 都继承 BaseSkill，统一 run(SkillInput) -> SkillOutput 接口
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class SkillInput:
    """统一的 Skill 输入"""
    query: str                                    # 用户问题（或改写后的问题）
    workspace_id: str = ""                        # 工作空间ID
    context: Dict[str, Any] = field(default_factory=dict)  # 额外上下文（前面 skill 的输出可以放这里）


@dataclass
class SkillOutput:
    """统一的 Skill 输出"""
    success: bool = True                          # 是否执行成功
    data: Dict[str, Any] = field(default_factory=dict)     # 输出数据（每个 skill 自定义内容）
    message: str = ""                             # 可读的说明信息


class BaseSkill(ABC):
    """
    Skill 基类
    所有 Skill 必须实现：
    - name: 技能名称
    - description: 技能描述（供 Harness 了解这个 skill 干什么）
    - run(): 执行逻辑
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Skill 名称"""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Skill 描述"""
        ...

    @abstractmethod
    def run(self, skill_input: SkillInput) -> SkillOutput:
        """
        执行 Skill

        Args:
            skill_input: 统一输入格式

        Returns:
            SkillOutput: 统一输出格式
        """
        ...
