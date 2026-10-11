"""FastAPI 服务脚手架：把每个案例的 LangGraph 应用暴露成可远程访问的 HTTP 服务。

生产要点：
- /health 用于云平台（Render/Fly）存活探针。
- /invoke 是标准入口，统一请求/响应模型，屏蔽各案例内部 State 差异。
- 服务可被 docker-compose / Render / Fly.io 直接拉起。
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeVar

from fastapi import FastAPI
from pydantic import BaseModel, Field

from core.logging import get_logger

log = get_logger("server")

# 一个“调用 Agent”的函数签名：输入消息文本与线程 id，返回 (回答, 元信息)
AgentRunner = Callable[[str, str | None], tuple[str, dict[str, Any]]]
T = TypeVar("T")


class InvokeRequest(BaseModel):
    message: str = Field(..., min_length=1, description="用户输入")
    thread_id: str | None = Field(default=None, description="会话/线程 id，用于 checkpoint 续跑")


class InvokeResponse(BaseModel):
    answer: str
    meta: dict[str, Any] = Field(default_factory=dict)


def create_app(
    runner: AgentRunner,
    service_name: str = "agent",
    extra_routes: list[Callable[[FastAPI], None]] | None = None,
) -> FastAPI:
    """构建一个标准 Agent 服务。

    runner: (message, thread_id) -> (answer, meta)。每个案例用 build_runner 包装自己的图。
    """
    app = FastAPI(title=service_name, version="1.0.0")
    extra_routes = extra_routes or []

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": service_name}

    @app.post("/invoke", response_model=InvokeResponse)
    def invoke(req: InvokeRequest) -> InvokeResponse:
        answer, meta = runner(req.message, req.thread_id)
        return InvokeResponse(answer=answer, meta=meta)

    for add in extra_routes:
        add(app)

    return app
