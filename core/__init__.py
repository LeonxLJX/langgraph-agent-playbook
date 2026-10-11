"""langgraph-agent-playbook 共享 core 包。

提供所有 15 个案例共用的生产级基础设施：
- settings  : 基于 pydantic-settings 的环境变量配置
- logging   : 结构化日志
- llm       : LLM 工厂（含限流退避 / 超时）
- checkpoint: Checkpoint 持久化工厂（内存 / Postgres）
- tracing   : LangSmith 追踪开关
- server    : FastAPI 服务脚手架（/health + /invoke）
- fake_llm  : 仅用于测试的可控假 LLM（离线跑通 Agent 逻辑）
"""

from core.settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
