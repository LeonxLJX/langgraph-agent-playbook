"""
案例 14 —— 生产可观测性与成本控制
==================================
难点：Agent 上生产后是黑盒 —— 不知道每轮跑了哪些节点、花了多少 token、哪一步最慢、
      哪里报错。没有可观测性，线上问题只能靠猜；token 成本也无法预算。

生产级解法：分层可观测
  1. LangSmith 追踪：配置 LANGCHAIN_TRACING=true 后，整条图（含每次模型调用）自动上链，
     可在后台按 thread_id 回看每步输入输出与耗时。
  2. 节点级插桩：用一个装饰器给每个节点计时 + 记录 token 估算 + 捕获异常，
     落到结构化日志（core/logging 支持 JSON）。
  3. 运行级指标：每次 invoke 汇总 steps、节点调用数、估算 token、耗时，
     随结果一起返回（可喂给监控大盘做成本/延迟告警）。

运行：
  # 开启 LangSmith（可选）
  #   LANGCHAIN_TRACING=true LANGCHAIN_API_KEY=lsv2_xxx python main.py --message "..."
  python main.py --message "你好"          # 看 meta 里的观测指标
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

import time
from collections.abc import Callable
from functools import wraps
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case14")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def estimate_tokens(messages: list[BaseMessage]) -> int:
    return sum((len(m.content or "") // 3 + 20) for m in messages)


def make_telemetry():
    """返回节点插桩装饰器 + 指标存储。"""
    events: list[dict] = []

    def instrument(name: str, fn: Callable) -> Callable:
        @wraps(fn)
        def wrapped(state):
            start = time.perf_counter()
            try:
                out = fn(state)
                events.append({
                    "node": name, "status": "ok",
                    "ms": round((time.perf_counter() - start) * 1000, 1),
                })
                return out
            except Exception as exc:
                events.append({
                    "node": name, "status": "error",
                    "ms": round((time.perf_counter() - start) * 1000, 1),
                    "error": str(exc),
                })
                raise
        return wrapped

    return instrument, events


def build_graph(settings: Settings, model=None, instrument=None):
    if model is None:
        model = build_chat_model(settings)

    def agent_fn(state: AgentState) -> dict:
        ai = model.invoke(state["messages"])
        return {"messages": [ai]}

    def agent_node(state: AgentState) -> dict:
        return instrument("agent", agent_fn)(state) if instrument else agent_fn(state)

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    instrument, events = make_telemetry()
    graph = build_graph(settings, model=model, instrument=instrument)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        start = time.perf_counter()
        result = graph.invoke({"messages": [{"role": "user", "content": message}]}, config)
        duration_ms = round((time.perf_counter() - start) * 1000, 1)
        final = result["messages"][-1]

        meta = {
            "duration_ms": duration_ms,
            "steps": len(result["messages"]),
            "node_events": events,
            "est_tokens": estimate_tokens(result["messages"]),
            "tracing_enabled": settings.langsmith_tracing,
        }
        log.info("run metrics: %s", meta)
        return final.content, meta

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    # 只有显式开启时才会上报追踪，避免误把调试流量打进计费
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case14-observability")
