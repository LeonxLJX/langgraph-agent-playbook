"""
案例 01 —— 工具调用参数解析崩溃
================================
难点：模型返回的 tool_call 里参数是非法 JSON / 缺少必填字段 / 字段类型不对，
      直接交给 langgraph 自带的 ToolNode 执行会抛异常、整个 Agent 中断。

生产级解法：
  1. 用「带保护的守卫工具节点」替代裸 ToolNode：
     - 工具名不存在  -> 回传"工具不存在"的 ToolMessage，让模型重新选择
     - 参数校验失败  -> 回传带错误原因和可用参数的 ToolMessage，让模型自纠
     - 执行抛异常    -> 捕获并回传错误信息，而不是让整次运行崩溃
  2. 错误以消息形式回传给模型（而不是异常），模型能据此自我修正 —— 这是"自愈"的关键。
  3. 同时给图加 recursion_limit 兜底，极端情况下也不至于无限跑。

运行：
  python main.py --message "北京天气并转成摄氏温度"
  python main.py --serve --port 8000
"""
from __future__ import annotations

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, MessagesState, StateGraph

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case01")


# ---------------------------------------------------------------- 工具定义
@tool
def get_weather(city: str) -> str:
    """查询指定城市当前的天气，返回温度与描述。参数：city 城市名（必填字符串）。"""
    # 演示用假数据；生产替换为真实 API 调用
    table = {"北京": "晴 22°C", "上海": "多云 24°C", "广州": "雷阵雨 28°C"}
    return table.get(city, f"{city}: 暂无数据")

@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """把摄氏温度转为华氏。参数：celsius 摄氏温度（必填数值）。"""
    return celsius * 9 / 5 + 32

TOOLS: list[BaseTool] = [get_weather, celsius_to_fahrenheit]


def _tool_map() -> dict[str, BaseTool]:
    return {t.name: t for t in TOOLS}


# ------------------------------------------------- 带保护的守卫工具节点
def _guarded_tools_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
    """执行上一条 AIMessage 里的所有 tool_call，任何失败都以 ToolMessage 回传而不抛异常。"""
    last = state["messages"][-1]
    tool_calls = getattr(last, "tool_calls", None) or []
    if not tool_calls:
        return {"messages": []}

    tool_map = _tool_map()
    out: list[BaseMessage] = []
    for tc in tool_calls:
        name, call_id = tc["name"], tc["id"]
        args = tc.get("args", {})
        t = tool_map.get(name)
        if t is None:
            # 幻觉工具名：不崩溃，明确告诉模型可用集合
            log.warning("模型调用了不存在的工具 %r", name)
            out.append(ToolMessage(
                content=f"[工具不存在] '{name}' 不是可用工具。可用工具: {sorted(tool_map)}。请重新选择。",
                tool_call_id=call_id,
            ))
            continue
        try:
            # 关键：把失败圈在 try 里，校验/执行错误都变成消息而非异常
            result = t.invoke(args)
            out.append(ToolMessage(content=str(result), tool_call_id=call_id))
        except Exception as exc:  # noqa: BLE001 - 工具异常统一兜底
            log.warning("工具 %s 执行失败: %s (args=%s)", name, exc, args)
            out.append(ToolMessage(
                content=(
                    f"[工具调用失败] {name} 参数 {args} 校验/执行出错: {exc}。"
                    f"请修正参数后重新调用该工具。"
                ),
                tool_call_id=call_id,
            ))
    return {"messages": out}


# ---------------------------------------------------------------- 图构建
def build_graph(settings: Settings, model=None):
    """构建 LangGraph 应用。

    model 参数供测试注入 FakeChatModel；生产不传则用真实模型。
    """
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools(TOOLS)

    def agent_node(state: MessagesState) -> dict[str, list[BaseMessage]]:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def route(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END

    builder = StateGraph(MessagesState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", _guarded_tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    compiled = builder.compile()
    # 兜底：限制图执行深度，防死循环
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)
    return _make_runner(graph, settings)


def _make_runner(graph, settings: Settings) -> AgentRunner:
    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
        )
        final = result["messages"][-1]
        meta = {
            "steps": len(result["messages"]),
            "final_type": type(final).__name__,
        }
        return final.content, meta
    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(
        lambda: build_runner(_settings),
        service_name="case01-tool-call-parse",
    )
