"""
案例 03 —— 死循环熔断
======================
难点：Agent 陷入死循环 —— 要么反复调用同一个工具（拿到相同结果还继续调），
      要么条件路由在两个节点间来回振荡，直到把 token 烧光 / 请求打到超时。

生产级解法：双重熔断
  1. 步数上限：state 里维护 steps，超过 max_steps 就注入“强制收尾”消息，逼模型直接作答。
  2. 循环检测：连续 N 次“相同工具调用”判定为循环，注入停用提示，避免空转。
  3. 底层再用 recursion_limit 兜底，即使前两招都失效也不会无限跑。

运行：
  python main.py --message "帮忙查天气"
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case03")


@tool
def get_weather(city: str) -> str:
    """查询城市天气。参数：city 城市名。"""
    return f"{city}: 晴 22°C"

TOOLS: list[BaseTool] = [get_weather]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    steps: int
    loop_count: int
    forced: bool


def _prev_ai_with_tools(messages: list[BaseMessage]) -> AIMessage | None:
    for m in reversed(messages[:-1]):
        if getattr(m, "tool_calls", None):
            return m  # type: ignore[return-value]
    return None


def _same_calls(a: BaseMessage, b: BaseMessage | None) -> bool:
    if b is None:
        return False
    ta = [tuple(sorted((c["name"], str(sorted(c.get("args", {}).items())))) for c in a.tool_calls)]  # type: ignore[union-attr]
    tb = [tuple(sorted((c["name"], str(sorted(c.get("args", {}).items())))) for c in b.tool_calls)]  # type: ignore[union-attr]
    return ta == tb


def build_graph(settings: Settings, model=None):
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools(TOOLS)

    def agent_node(state: AgentState) -> dict:
        ai = model_with_tools.invoke(state["messages"])
        return {
            "messages": [ai],
            "steps": state["steps"] + 1,
            "forced": False,
        }

    def guard_node(state: AgentState) -> dict:
        """在“agent -> tools”之间把关，检测步数超限与循环。"""
        msgs = state["messages"]
        last = msgs[-1]
        if not getattr(last, "tool_calls", None):
            return {"loop_count": state["loop_count"]}

        prev = _prev_ai_with_tools(msgs)
        same = _same_calls(last, prev)
        loop_count = state["loop_count"] + 1 if same else 0

        forced = None
        if loop_count >= settings.loop_threshold:
            forced = (
                f"已检测到循环：你已连续 {loop_count} 次调用同一工具 {last.tool_calls[0]['name']}。"
                f"请停止调用工具，直接基于已有信息给出最终答复。"
            )
        elif state["steps"] >= settings.max_steps:
            forced = "已达到最大执行步数上限。请停止调用工具，直接给出最终答复。"

        if forced:
            log.warning("熔断触发: %s", forced)
            return {
                "messages": [SystemMessage(content=forced)],
                "loop_count": loop_count,
                "forced": True,
            }
        return {"loop_count": loop_count}

    def route_after_agent(state: AgentState) -> str:
        last = state["messages"][-1]
        return "guard" if getattr(last, "tool_calls", None) else END

    def route_after_guard(state: AgentState) -> str:
        # 熔断后去 agent 让模型收尾；否则正常执行工具
        return "agent" if state.get("forced") else "tools"

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("guard", guard_node)
    builder.add_node("tools", _tools_node(TOOLS))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent, {"guard": "guard", END: END})
    builder.add_conditional_edges("guard", route_after_guard, {"agent": "agent", "tools": "tools"})
    builder.add_edge("tools", "agent")

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def _tools_node(tools: list[BaseTool]):
    from langgraph.prebuilt import ToolNode

    return ToolNode(tools)


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}],
             "steps": 0, "loop_count": 0, "forced": False},
            config,
        )
        final = result["messages"][-1]
        meta = {
            "steps": result["steps"],
            "loop_count": result["loop_count"],
            "tripped": bool(result.get("forced")),
        }
        return final.content, meta

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case03-loop-guard")
