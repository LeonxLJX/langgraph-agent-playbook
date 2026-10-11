"""
案例 02 —— 上下文超长与 token 溢出
====================================
难点：多轮对话 / 反复工具调用会让 messages 无限增长，最终超模型上下文窗口，
      轻则报错、重则 token 费用爆炸。

生产级解法：历史消息压缩（summarization）
  1. 给 messages 用自定义 reducer：既能正常追加消息，又能用「重置标记」整体替换历史。
  2. 估算当前上下文 token 数（启发式：字符数/系数 + 每条消息固定开销），超过阈值触发压缩。
  3. 压缩节点把“旧消息”交给模型生成一段总结，替换成一条 SystemMessage 摘要，保留最近 K 条。
  4. 压缩后重新让模型基于摘要作答，历史“无缝延续”。

运行：
  python main.py --message "多轮对话，观察 messages 被压缩"
  python main.py --serve --port 8000
"""
# 注意：这里故意不用 `from __future__ import annotations`，
# 因为本地自定义 TypedDict 需要真实注解供 langgraph 求值（否则 NameError: Annotated）。

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case02")

_SUMMARY_SIGNAL = "__reset__"


# ------------------------------------------------------- 自定义 reducer
def _messages_reducer(left: list[BaseMessage], right: list[BaseMessage]) -> list[BaseMessage]:
    """messages 的 reducer：
    - 节点正常返回消息列表 -> 用 langgraph 的 add_messages 追加（按 id 去重）；
    - 节点返回 [{"__reset__": True, "messages": new}] -> 整体替换历史（压缩用）。
    """
    if (
        isinstance(right, list)
        and len(right) == 1
        and isinstance(right[0], dict)
        and right[0].get(_SUMMARY_SIGNAL)
    ):
        return right[0]["messages"]
    from langgraph.graph.message import add_messages

    return add_messages(left, right)


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], _messages_reducer]


# ---------------------------------------------------------- token 估算
def estimate_tokens(messages: list[BaseMessage]) -> int:
    """启发式 token 估算。生产可用 tiktoken 精确计数；这里用系数估算以保持零额外依赖。"""
    per_msg_overhead = 20
    return sum((len(m.content or "") // 3 + per_msg_overhead) for m in messages)


# ------------------------------------------------------------- 图构建
def build_graph(settings: Settings, model=None):
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools([])  # 本案例专注上下文管理，不涉及工具

    budget = settings.context_token_budget
    keep_recent = settings.keep_recent

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def summarize_node(state: AgentState) -> dict:
        msgs = state["messages"]
        # 保留最近的 keep_recent 条，其余交给模型总结
        to_summarize = msgs[:-keep_recent] if len(msgs) > keep_recent else msgs
        prompt = [
            SystemMessage(
                content=(
                    "请把下面这段历史对话压缩成一段不超过 80 字的中文总结，"
                    "保留关键事实、用户诉求、已确认的信息。只输出总结本身。"
                )
            ),
            *to_summarize,
        ]
        summary = model_with_tools.invoke(prompt).content
        new_history: list[BaseMessage] = [
            SystemMessage(content=f"【历史对话总结】{summary}"),
            *msgs[-keep_recent:],
        ]
        log.info("上下文压缩: %d 条 -> %d 条", len(msgs), len(new_history))
        return {"messages": [{_SUMMARY_SIGNAL: True, "messages": new_history}]}

    def route(state: AgentState) -> str:
        # 本案例没有工具，直接判断是否超预算触发压缩
        if estimate_tokens(state["messages"]) > budget:
            return "summarize"
        return END

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_node("summarize", summarize_node)
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route, {"summarize": "summarize", END: END})
    # 压缩后基于摘要再作答一次，避免压缩完直接收尾导致答案丢失
    builder.add_edge("summarize", "agent")

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}]},
            config,
        )
        final = result["messages"][-1]
        # 报告有没有触发压缩
        summarized = any(
            getattr(m, "type", "") == "system" and m.content.startswith("【历史对话总结】")
            for m in result["messages"]
        )
        meta = {
            "steps": len(result["messages"]),
            "compressed": summarized,
            "total_est_tokens": estimate_tokens(result["messages"]),
        }
        return final.content, meta

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(
        lambda: build_runner(_settings),
        service_name="case02-context-truncate",
    )
