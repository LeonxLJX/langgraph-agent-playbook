"""
案例 10 —— 持久化与断点续跑
============================
难点：Agent 跑长任务 / 多轮对话时，进程重启或并发请求会把状态丢掉；
      不持久化，一切“记住的东西”都随进程消失，也做不了断点续跑和时间旅行。

生产级解法：Checkpoint 持久化
  1. 给图编译时传入 checkpointer（内存或 Postgres），所有消息/状态按 thread_id 落库。
  2. 同一 thread_id 再次调用会自动带上历史 —— 换一个全新图实例 + 同一个库，记忆也在。
  3. 生产用 Postgres（core/checkpoint.py 统一入口）；本机无 PG 时回退内存做演示。
  4. 与案例 08 的 interrupt 天然配合：审批断点也是靠 checkpoint 存续。

运行：
  # 内存演示
  python main.py --message "记住我叫小王" --thread-id u1
  python main.py --message "我叫什么？"   --thread-id u1   # 记得
  python main.py --message "我叫什么？"   --thread-id u2   # 不记得（新线程）

  # 生产：先起 Postgres，设置 POSTGRES_URI 后重启服务也保留
  #   docker compose up -d postgres
  python main.py --serve --port 8010    # 配好 .env 的 POSTGRES_URI
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.checkpoint import build_checkpoint_saver, checkpoint_kind
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case10")


@tool
def remember(fact: str) -> str:
    """记录一个需要长期记住的事实。参数：fact 事实内容。"""
    return f"已记录：{fact}"

TOOLS: list[BaseTool] = [remember]


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(settings: Settings, model=None, saver=None):
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
            if t is None:
                out.append(ToolMessage(content=f"[工具不存在] {tc['name']}", tool_call_id=tc["id"]))
            else:
                try:
                    out.append(ToolMessage(content=str(t.invoke(tc.get("args", {}))), tool_call_id=tc["id"]))
                except Exception as exc:  # noqa: BLE001
                    out.append(ToolMessage(content=f"[调用失败] {exc}", tool_call_id=tc["id"]))
        return {"messages": out}

    def route(state: AgentState) -> str:
        last = state["messages"][-1]
        return "tools" if getattr(last, "tool_calls", None) else END  # type: ignore[return-value]

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("tools", tools_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"tools": "tools", END: END})
    builder.add_edge("tools", "agent")

    if saver is None:
        saver, need_setup = build_checkpoint_saver(settings)
        if need_setup:
            saver.setup()  # Postgres 建表
            log.info("已初始化 Postgres checkpoint 表")
    compiled = builder.compile(checkpointer=saver)
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model, saver=saver)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        tid = thread_id or "default"
        config = {"configurable": {"thread_id": tid}}
        result = graph.invoke({"messages": [{"role": "user", "content": message}]}, config)
        final = result["messages"][-1]
        return final.content, {
            "thread_id": tid,
            "checkpoint": checkpoint_kind(settings),
            "history_messages": len(result["messages"]),
        }

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case10-persistence-resume")
