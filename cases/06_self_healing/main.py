"""
案例 06 —— 节点异常自愈与状态一致性
====================================
难点：Agent 里某个节点（尤其模型调用）中途抛异常，整个图直接失败；
      而且之前已产生的副作用（工具已执行、消息已写）可能已经落盘，状态不一致。

生产级解法：错误边界 + 重试一次 + 优雅降级
  1. agent 节点包一层“错误边界”：调用失败先重试一次（瞬时抖动可自愈）。
  2. 重试仍失败 -> 记录错误到 state（errors，可观测），标记 failed，**不再产生残缺消息**。
  3. 路由到 fallback 节点：用不依赖模型的固定文案给用户一个优雅回应，
     而不是把堆栈抛给用户或让图崩溃。
  4. errors 走列表 reducer，一次运行多次失败的记录都能追溯。

运行：
  python main.py --message "你好"
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case06")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    errors: Annotated[list[str], add]  # 失败记录，跨节点可追溯
    failed: bool


def build_graph(settings: Settings, model=None):
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools([])

    def agent_node(state: AgentState) -> dict:
        """错误边界：调用失败先重试一次，再失败则标记 failed 走降级。"""
        last_error: str | None = None
        for attempt in range(1, 3):  # 原调用 + 1 次重试
            try:
                ai = model_with_tools.invoke(state["messages"])
                return {"messages": [ai], "failed": False}
            except Exception as exc:  # noqa: BLE001
                last_error = f"agent 调用失败(第{attempt}次): {exc}"
                log.warning("agent 节点异常，重试 #%d: %s", attempt, exc)
                # 追加一条反馈消息再重试，让模型/上下文带上失败信息
                state["messages"] = state["messages"] + [
                    SystemMessage(content=f"上一次模型调用失败：{last_error}。请重试。")
                ]
        # 重试仍失败：记录错误、标记 failed，不产生残缺 AI 消息
        return {"messages": [], "errors": [str(last_error)], "failed": True}

    def fallback_node(state: AgentState) -> dict:
        """不依赖模型的优雅降级文案。"""
        log.error("触发降级，已记录 %d 条错误: %s", len(state["errors"]), state["errors"])
        return {
            "messages": [
                AIMessage(content="抱歉，服务暂时出现了问题，请稍后再试。我们会持续关注并修复。")
            ]
        }

    def route(state: AgentState) -> str:
        if state.get("failed"):
            return "fallback"
        last = state["messages"][-1]
        return END if not getattr(last, "tool_calls", None) else "tools"  # type: ignore[return-value]

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("fallback", fallback_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"fallback": "fallback", END: END})

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}], "errors": [], "failed": False},
            config,
        )
        final = result["messages"][-1]
        return final.content, {"errors": result["errors"], "failed": bool(result["failed"])}

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case06-self-healing")
