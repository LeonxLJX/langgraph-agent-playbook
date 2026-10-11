"""LLM 工厂：按 provider 构建 langchain 聊天模型，统一注入超时与重试。

生产要点：
- 不直接 new 模型，集中在这里管控超时 / 重试 / 模型名，避免各案例各写一套。
- 默认走 OpenAI 兼容接口；`OPENAI_BASE_URL` 可指向任意兼容网关或代理。
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from core.settings import Settings

if TYPE_CHECKING:
    from langchain_core.language_models.chat_models import BaseChatModel


def build_chat_model(settings: Settings) -> BaseChatModel:
    """根据配置构建 chat 模型。"""
    if settings.llm_provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        if not settings.anthropic_api_key:
            raise RuntimeError("llm_provider=anthropic 但未设置 ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
            api_key=settings.anthropic_api_key,
        )

    from langchain_openai import ChatOpenAI

    if not settings.openai_api_key:
        raise RuntimeError("llm_provider=openai 但未设置 OPENAI_API_KEY")
    return ChatOpenAI(
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
