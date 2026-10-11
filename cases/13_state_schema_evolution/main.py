"""
案例 13 —— 复杂状态机图编排与 Schema 演进
==========================================
难点：生产图往往有多个分支 + 条件路由 + 回流，稍一复杂就容易：
  - 条件路由写错分支名 / 指向不存在的节点；
  - 加了新字段没做兼容，老 checkpoint 反序列化直接 KeyError；
  - 状态字段随手改，升级后线上历史状态全崩。

生产级解法：
  1. 拓扑集中声明（一张小表说清每个分支去向），杜绝“箭头满天飞”难排查。
  2. 状态带 schema_version，新增字段一律“可选 + .get(默认值)”向后兼容。
  3. 用 Annotated[list, add] 做结果聚合，多分支产出可合并。
  4. 条件路由函数返回明确分支名，并和 add_conditional_edges 的分支表对死。

运行：
  python main.py --message "写一篇新品文案"   # 走 writer 分支
  python main.py --message "查竞品数据"       # 走 research 分支
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from operator import add
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case13")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    branch: str                # 当前走向：research | writer | direct
    result: Annotated[list[str], add]  # 各分支产出的聚合
    schema_version: int        # 状态 schema 版本，用于演进兼容


def _last_user_text(messages: list[BaseMessage]) -> str:
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            return m.content or ""
    return ""


# 拓扑集中声明：节点 -> 它有哪些出边
TOPOLOGY = {
    "classify": {"research": "research", "writer": "writer", "direct": "direct", END: END},
    "research": {END: END},
    "writer": {END: END},
    "direct": {END: END},
}


def build_graph(settings: Settings, model=None, research_model=None, writer_model=None):
    if model is None:
        model = build_chat_model(settings)
    if research_model is None:
        research_model = build_chat_model(settings)
    if writer_model is None:
        writer_model = build_chat_model(settings)

    def classify_node(state: AgentState) -> dict:
        router = SystemMessage(
            content="判断用户意图，只输出 research(查资料/分析) 或 writer(写文案/创作) 或 direct(其他直接回答)。只输出一个词。"
        )
        ai = model.invoke([router, *state["messages"]])
        branch = ai.content.strip().lower()
        branch = branch if branch in {"research", "writer", "direct"} else "direct"
        return {"branch": branch}

    def research_node(state: AgentState) -> dict:
        q = _last_user_text(state["messages"])
        ans = research_model.invoke([SystemMessage("你是研究助手，输出严谨带来源的结论。"), HumanMessage(content=q)])
        return {"result": [f"[research] {ans.content}"], "messages": [ans]}

    def writer_node(state: AgentState) -> dict:
        q = _last_user_text(state["messages"])
        ans = writer_model.invoke([SystemMessage("你是文案助手，输出可直接交付的成品。"), HumanMessage(content=q)])
        return {"result": [f"[writer] {ans.content}"], "messages": [ans]}

    def direct_node(state: AgentState) -> dict:
        return {"result": ["[direct] 已回答"], "messages": [AIMessage(content="这个问题我可以直接回答。")]}

    def route_branch(state: AgentState) -> str:
        return state.get("branch", "direct")

    builder = StateGraph(AgentState)
    builder.add_node("classify", classify_node)
    builder.add_node("research", research_node)
    builder.add_node("writer", writer_node)
    builder.add_node("direct", direct_node)

    builder.add_edge(START, "classify")
    # 条件路由：分支表与 TOPOLOGY 里的出边对死，减少写错分支名的概率
    builder.add_conditional_edges("classify", route_branch, TOPOLOGY["classify"])
    for node in ("research", "writer", "direct"):
        builder.add_edge(node, END)

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, research_model=None, writer_model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model, research_model=research_model, writer_model=writer_model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke(
            {
                "messages": [{"role": "user", "content": message}],
                "branch": "direct",
                "result": [],
                "schema_version": 1,
            },
            config,
        )
        final = result["messages"][-1]
        return final.content, {"branch": result["branch"], "result": result["result"]}

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case13-state-schema-evolution")
