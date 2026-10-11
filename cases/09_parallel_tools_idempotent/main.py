"""
案例 09 —— 并行工具调用与幂等
==============================
难点：一次模型回复里往往带多个互相独立的工具调用。串行执行慢；
      而如果调用“相同签名”（同工具同参数），并发重放会造成重复副作用（重复下单/重复记账）。

生产级解法：去重幂等 + 并发执行 + 结果映射
  1. 按 (工具名, 参数签名) 去重：同一批里完全相同的调用只执行一次，结果共享 —— 幂等。
  2. 去重后的调用用线程池**并发**执行，显著缩短耗时。
  3. 结果按 tool_call_id 一一映射回原调用，保证模型能正确配对。
  4. 每个工具带执行计数器，测试可验证“相同调用只执行一次”。

运行：
  python main.py --message "查北京、北京、纽约三地天气"   # 北京会并发去重只执行一次
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case09")


# ------------------------------------------------ 带执行计数器的工具
_weather_calls = {"n": 0}
_time_calls = {"n": 0}


@tool
def get_weather(city: str) -> str:
    """查询城市天气。参数：city 城市名。"""
    _weather_calls["n"] += 1
    return f"{city}: 晴 22°C"

@tool
def get_time(timezone: str) -> str:
    """查询时区时间。参数：timezone 时区名。"""
    _time_calls["n"] += 1
    return f"{timezone}: 12:00"

TOOLS: list[BaseTool] = [get_weather, get_time]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def _signature(call) -> tuple:
    name = call["name"]
    args = str(sorted(call.get("args", {}).items()))
    return (name, args)


def build_graph(settings: Settings, model=None):
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def dispatch_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return {"messages": []}

        # 1) 幂等去重：同一签名只保留一个代表调用
        groups: dict[tuple, list] = {}
        for c in calls:
            groups.setdefault(_signature(c), []).append(c)
        unique_calls = [g[0] for g in groups.values()]

        # 2) 并发执行去重后的调用
        def run_one(c) -> str:
            t = tool_map.get(c["name"])
            if t is None:
                return f"[工具不存在] {c['name']}"
            try:
                return str(t.invoke(c.get("args", {})))
            except Exception as exc:  # noqa: BLE001
                return f"[调用失败] {exc}"

        with ThreadPoolExecutor(max_workers=min(8, len(unique_calls))) as ex:
            exec_results = {c["id"]: r for c, r in zip(unique_calls, ex.map(run_one, unique_calls))}

        # 3) 结果映射回所有调用（同签名共享结果，保证幂等且配对正确）
        id_result: dict[str, str] = {}
        for group in groups.values():
            shared = exec_results[group[0]["id"]]
            for c in group:
                id_result[c["id"]] = shared

        out = [ToolMessage(content=id_result[c["id"]], tool_call_id=c["id"]) for c in calls]
        return {"messages": out}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "dispatch" if getattr(last, "tool_calls", None) else END  # type: ignore[return-value]

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("dispatch", dispatch_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"dispatch": "dispatch", END: END})
    builder.add_edge("dispatch", "agent")

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke({"messages": [{"role": "user", "content": message}]}, config)
        final = result["messages"][-1]
        return final.content, {
            "weather_calls": _weather_calls["n"],
            "time_calls": _time_calls["n"],
        }

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case09-parallel-tools-idempotent")
