"""基于 pydantic-settings 的集中配置。

所有案例都从这里读取配置，杜绝散落的 os.environ 与魔法字符串。
生产环境通过环境变量或 .env 文件注入。
"""
from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # ---- LLM ----
    # 默认使用 OpenAI 兼容接口（环境变量 OPENAI_API_KEY / OPENAI_BASE_URL 可指向任意兼容网关）
    llm_provider: str = Field(default="openai", description="openai | anthropic")
    llm_model: str = Field(default="gpt-4o-mini", description="默认模型")
    llm_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    llm_timeout_seconds: float = Field(default=60.0)
    llm_max_retries: int = Field(default=3, ge=0)
    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)
    anthropic_api_key: str | None = Field(default=None)

    # ---- 可观测性 ----
    langsmith_tracing: bool = Field(default=False, description="是否启用 LangSmith 追踪")
    langsmith_api_key: str | None = Field(default=None)
    langsmith_project: str | None = Field(default=None)
    log_level: str = Field(default="INFO")

    # ---- 持久化 ----
    # 不设置则回退到内存 checkpoint（生产建议 Postgres）
    postgres_uri: str | None = Field(default=None)
    # 每个 Agent 服务的服务名（用于日志/追踪/健康检查标签）
    service_name: str = Field(default="agent")

    # ---- 运行时 ----
    max_steps: int = Field(default=25, ge=1, description="Agent 最大执行步数，防死循环")
    recursion_limit: int = Field(default=50, ge=1, description="LangGraph 图执行最大递归深度")
    loop_threshold: int = Field(default=5, ge=2, description="连续相同工具调用达到该次数判定为循环")

    # ---- 工具调用治理（案例 04） ----
    max_invalid_tool_calls: int = Field(default=3, ge=1, description="累计非法工具调用达到该次数即强制收尾")

    # ---- 结构化输出（案例 11） ----
    max_struct_retries: int = Field(default=3, ge=1, description="结构化输出校验失败的最大重试次数")

    # ---- 上下文管理（案例 02） ----
    context_token_budget: int = Field(default=2000, ge=50, description="触发历史压缩的 token 阈值")
    keep_recent: int = Field(default=6, ge=1, description="压缩时保留的最近消息条数")

    @property
    def has_llm_key(self) -> bool:
        if self.llm_provider == "anthropic":
            return bool(self.anthropic_api_key)
        return bool(self.openai_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
