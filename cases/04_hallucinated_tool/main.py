"""
案例 04 —— 幻觉工具名拦截
==========================
难点：模型“幻觉”出一个不存在的工具名（或拼错），直接执行必然报错；
      更糟的是模型可能反复尝试同一个非法工具，浪费 token 还不出结果。

生产级解法：白名单校验 + 有界自纠
  1. 执行前先校验工具名是否在注册表 TOOLS 里 —— 非法调用不进执行器。
  2. 非法调用回传 [工具不存在] + 可用工具列表，让模型自己改正。
  3. 记录非法调用次数，达到 max_invalid_tool_calls 后注入强制收尾提示，防止无限自纠空转。
  4. 合法工具执行失败也用 try/except 兜底回传（与案例 01 同款守卫）。

运行：
  python main.py --message "发一封邮件给我"   # 故意触发非法工具
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case04")


@tool
def get_weather(city: str) -> str:
    """查询城市天气。参数：city 城市名。"""
    return f"{city}: 晴 22°C"

@tool
def get_time(timezone: str) -> str:
    """查询时区当前时间。参数：timezone 时区名。"""
    return f"{timezone}: 12:00"

TOOLS: list[BaseTool] = [get_weather, get_time]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    invalid_calls: int


def build_graph(settings: Settings, model=None):
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    def agent_node(state: AgentState) -> dict:
        ai = model_with_tools.invoke(state["messages"])
        return {"messages": [ai]}

    def dispatch_node(state: AgentState) -> dict:
        """白名单校验 + 有界自纠 + 守卫执行。"""
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        out: list[BaseMessage] = []
        invalid = 0
        for tc in tool_calls:
            name, call_id, args = tc["name"], tc["id"], tc.get("args", {})
            t = tool_map.get(name)
            if t is None:
                invalid += 1
                log.warning("非法工具调用: %s", name)
                out.append(ToolMessage(
                    content=f"[工具不存在] '{name}' 不在可用列表 {sorted(tool_map)}。请改用这些工具。",
                    tool_call_id=call_id,
                ))
                continue
            try:
                out.append(ToolMessage(content=str(t.invoke(args)), tool_call_id=call_id))
            except Exception as exc:  # noqa: BLE001
                out.append(ToolMessage(
                    content=f"[工具调用失败] {name} 参数 {args} 出错: {exc}。请修正后重试。",
                    tool_call_id=call_id,
                ))

        new_invalid = state["invalid_calls"] + invalid
        forced = new_invalid >= settings.max_invalid_tool_calls
        if forced:
            out.append(SystemMessage(
                content=(
                    f"你已累计 {new_invalid} 次调用不存在的工具。请立即停止，"
                    f"只用可用工具 {sorted(tool_map)}，或直接基于已有信息作答。"
                )
            ))
        return {"messages": out, "invalid_calls": new_invalid}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "dispatch" if getattr(last, "tool_calls", None) else END

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
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}], "invalid_calls": 0},
            config,
        )
        final = result["messages"][-1]
        meta = {
            "invalid_calls": result["invalid_calls"],
            "forced": result["invalid_calls"] >= settings.max_invalid_tool_calls,
        }
        return final.content, meta

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case04-hallucinated-tool")
