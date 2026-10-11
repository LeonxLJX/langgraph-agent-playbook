"""LangSmith / 可观测性开关。

生产要点：
- 追踪必须在 import 阶段尽早开启，且只在显式配置时才开启，避免误把调试流量打进追踪计费。
- 关闭时不引入任何副作用（不打印、不阻塞）。
"""
from __future__ import annotations

import os

from core.settings import Settings


def enable_tracing(settings: Settings) -> bool:
    """按配置开启 LangSmith 追踪。返回是否已开启。"""
    if not settings.langsmith_tracing:
        return False
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    os.environ.setdefault("LANGCHAIN_TRACING", "true")
    if settings.langsmith_api_key:
        os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    if settings.langsmith_project:
        os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    return True
