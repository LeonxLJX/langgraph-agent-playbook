"""
案例 07 —— 多智能体路由与上下文隔离
====================================
难点：一个需求可能涉及“查资料”和“写文案”两类能力。把多个能力揉进一个 Agent
      上下文会互相污染；要让一个“调度者（Supervisor）”把请求路由给合适的专业 Agent，
      并且每个专业 Agent 的上下文互不泄露。

生产级解法：Supervisor 路由 + 专业 Agent 隔离上下文
  1. Supervisor 节点：用模型把用户意图分类成角色（research / writer），写入 state.active_agent。
  2. 条件路由：按 active_agent 把执行权交给对应专业节点。
  3. 每个专业节点用**自己的模型 + 自己的系统提示**，只接收“用户原始诉求”，
     不把调度者的中间结果带进去 —— 上下文天然隔离。
  4. 专业节点的产出作为最终答复返回。

运行：
  python main.py --message "写一段新品发布的宣传文案"
  python main.py --message "查一下市场竞品数据"
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case07")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    active_agent: str  # 路由结果：research | writer


def _last_human(messages: list[BaseMessage]) -> HumanMessage | None:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m
    return None


def build_graph(settings: Settings, model=None, research_model=None, writer_model=None):
    if model is None:
        model = build_chat_model(settings)
    if research_model is None:
        research_model = build_chat_model(settings)
    if writer_model is None:
        writer_model = build_chat_model(settings)

    def supervisor_node(state: AgentState) -> dict:
        router_prompt = SystemMessage(
            content=(
                "你是调度者。根据用户意图，只输出一个角色名：\n"
                "- research：需要查数据/资料/做分析；\n"
                "- writer：需要写文案/创作/润色。\n"
                "只输出 research 或 writer，不要输出其他内容。"
            )
        )
        ai = model.invoke([router_prompt, *state["messages"]])
        role = ai.content.strip().lower()
        if role not in {"research", "writer"}:
            role = "writer"
        log.info("路由到角色: %s", role)
        return {"active_agent": role}

    def research_node(state: AgentState) -> dict:
        query = _last_human(state["messages"])
        if query is None:
            return {"messages": [SystemMessage(content="缺少用户问题")], "active_agent": "research"}
        # 隔离：只用“研究专用系统提示 + 用户原始诉求”，不掺调度上下文
        ctx = [SystemMessage(content="你是研究助手：只做资料检索与数据分析，输出严谨、带来源的结论。"), query]
        ans = research_model.invoke(ctx)
        return {"messages": [ans], "active_agent": "research"}

    def writer_node(state: AgentState) -> dict:
        query = _last_human(state["messages"])
        if query is None:
            return {"messages": [SystemMessage(content="缺少用户问题")], "active_agent": "writer"}
        ctx = [SystemMessage(content="你是文案助手：负责广告文案、公众号文章等创作，输出可直接交付的成品。"), query]
        ans = writer_model.invoke(ctx)
        return {"messages": [ans], "active_agent": "writer"}

    def route(state: AgentState) -> str:
        role = state.get("active_agent", "")
        if role == "research":
            return "research"
        if role == "writer":
            return "writer"
        return END  # type: ignore[return-value]

    builder = StateGraph(AgentState)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("research", research_node)
    builder.add_node("writer", writer_node)
    builder.add_edge(START, "supervisor")
    builder.add_conditional_edges("supervisor", route, {"research": "research", "writer": "writer", END: END})
    builder.add_edge("research", END)
    builder.add_edge("writer", END)

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, research_model=None, writer_model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model, research_model=research_model, writer_model=writer_model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}], "active_agent": ""},
            config,
        )
        final = result["messages"][-1]
        return final.content, {"active_agent": result["active_agent"]}

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case07-multi-agent-routing")
