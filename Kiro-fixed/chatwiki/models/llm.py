"""
DeepSeek LLM 客户端封装（复用，含 <think> 去除逻辑）
"""

import json
import logging
import re
from typing import Optional

from openai import OpenAI

from chatwiki.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    def __init__(self, settings=None):
        cfg = (settings or get_settings()).llm
        self.cfg = cfg
        self._client = OpenAI(base_url=cfg.base_url, api_key=cfg.api_key or "EMPTY")

    def chat(self, prompt: str, system: Optional[str] = None,
             temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        resp = self._client.chat.completions.create(
            model=self.cfg.model_name,
            messages=messages,
            temperature=self.cfg.temperature if temperature is None else temperature,
            max_tokens=self.cfg.max_tokens if max_tokens is None else max_tokens,
        )
        raw = resp.choices[0].message.content or ""
        return self._strip_think(raw)

    def chat_json(self, prompt: str, system: Optional[str] = None,
                  temperature: Optional[float] = None, max_tokens: Optional[int] = None) -> dict:
        text = self.chat(prompt, system=system, temperature=temperature, max_tokens=max_tokens)
        return self._extract_json(text)

    @staticmethod
    def _strip_think(text: str) -> str:
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
        return text.strip()

    @staticmethod
    def _extract_json(text: str) -> dict:
        text = LLMClient._strip_think(text).strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError as e:
                logger.error("JSON解析失败: %s, raw=%r", e, text[:500])
                raise
        raise ValueError(f"无法从LLM输出中提取JSON: {text[:200]!r}")

    def health_check(self) -> bool:
        try:
            out = self.chat("回复ok", max_tokens=20)
            return bool(out)
        except Exception as e:
            logger.error("LLM健康检查失败: %s", e)
            return False
