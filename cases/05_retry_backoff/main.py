"""
案例 05 —— 上游限流退避重试
============================
难点：Agent 依赖的上游 API（LLM、业务服务、第三方）会 429 限流 / 超时 / 抖动。
      直接重试往往打得更狠，造成二次雪崩；不重试又白白失败。

生产级解法：指数退避 + 抖动（exponential backoff + jitter）
  1. 只对“可重试”的异常重试（429 限流、5xx、超时、连接错），4xx 业务错不重试。
  2. 用 tenacity 的 wait_exponential_jitter：退避时间指数增长并加随机抖动，
     避免所有实例同时重试形成“惊群”。
  3. 设最大尝试次数，超限后 reraise，让上层（守卫节点）把错误转成可读消息，不崩图。
  4. 尊重上游 Retry-After（示例简化，生产请读取响应头）。

运行：
  python main.py --message "查一下日活指标"
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case05")


# ------------------------------------------------- 可重试异常定义
class UpstreamRateLimit(RuntimeError):
    """上游 429 限流（可重试）。"""

class UpstreamServerError(RuntimeError):
    """上游 5xx / 超时（可重试）。"""

class UpstreamBusinessError(RuntimeError):
    """上游 4xx 业务错误（不可重试）。"""


def _retryable(exc: BaseException) -> bool:
    """只重试“瞬时可恢复”的异常。业务错误不重试。"""
    return isinstance(exc, (UpstreamRateLimit, UpstreamServerError))


# ------------------------------------------- 可注入故障的上游客户端
class FlakyAnalyticsClient:
    """模拟一个前 N 次限流、之后正常的上游服务，便于观察重试。"""

    def __init__(self, fail_times: int = 2):
        self._fail_times = fail_times
        self.calls = 0

    def query(self, metric: str) -> str:
        self.calls += 1
        if self._fail_times > 0:
            self._fail_times -= 1
            raise UpstreamRateLimit("HTTP 429 Too Many Requests")
        return f"{metric}: 12345"


_upstream = FlakyAnalyticsClient(fail_times=2)


def _call_upstream_with_retry(metric: str) -> str:
    """核心：用 tenacity 做指数退避 + 抖动重试。"""
    @retry(
        retry=retry_if_exception(_retryable),
        wait=wait_exponential_jitter(initial=0.05, max=0.5),  # 退避 + 抖动，防惊群
        stop=stop_after_attempt(4),                            # 最多试 4 次
        reraise=True,                                          # 超限后把最后一次异常抛出去
        before_sleep=lambda s: log.warning(
            "上游重试 #%s (%s)，%.2fs 后重试", s.attempt_number, s.outcome.exception(), s.idle_for
        ),
    )
    def _do() -> str:
        return _upstream.query(metric)

    return _do()


@tool
def get_metric(metric: str) -> str:
    """查询业务指标。参数：metric 指标名。会真实演示限流重试。"""
    try:
        return _call_upstream_with_retry(metric)
    except (UpstreamRateLimit, UpstreamServerError) as exc:
        # 重试耗尽后，把错误转成语义明确的消息（不让它炸穿图）
        raise RuntimeError(f"上游服务暂时不可用（重试已耗尽）: {exc}") from exc


TOOLS: list[BaseTool] = [get_metric]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(settings: Settings, model=None, upstream: FlakyAnalyticsClient | None = None):
    global _upstream
    if upstream is not None:
        _upstream = upstream
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def tools_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        out: list[BaseMessage] = []
        for tc in getattr(last, "tool_calls", None) or []:
            t = tool_map.get(tc["name"])
            try:
                out.append(ToolMessage(content=str(t.invoke(tc.get("args", {}))), tool_call_id=tc["id"]))
            except Exception as exc:  # noqa: BLE001
                out.append(ToolMessage(content=f"[调用失败] {exc}", tool_call_id=tc["id"]))
        return {"messages": out}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke({"messages": [{"role": "user", "content": message}]}, config)
        final = result["messages"][-1]
        return final.content, {"upstream_calls": _upstream.calls, "steps": len(result["messages"])}

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case05-retry-backoff")
