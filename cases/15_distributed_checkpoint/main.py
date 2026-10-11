"""
案例 15 —— 高并发分布式 Checkpoint 与多实例安全
================================================
难点：生产是多个实例/进程同时打同一个共享 checkpoint（Postgres）。
      不加控制会出现：
  - 同一会话并发写入相互覆盖，状态损坏/丢消息；
  - 每个实例各持一份“假状态”，横向扩展后行为不一致。

生产级解法：共享持久化 + 按 thread_id 串行化
  1. 用 Postgres checkpoint（core/checkpoint.py 统一入口），所有实例读同一份状态，
     天然支持横向扩展与断点续跑（见案例 10）。
  2. 对“同一 thread_id”的写入用进程内锁串行化 —— 同一场对话的并发更新不互相覆盖；
     不同 thread_id 完全并行，互不阻塞。
  3. 提供多实例部署说明：连接同一个 POSTGRES_URI，锁只在写那一瞬保护一致性，
     跨进程的强一致由 Postgres 行级 upsert + 事务保证（生产可加分布式锁升级）。
  4. 本机无 Postgres 时回退内存 saver，但**内存 saver 不支持多实例**，仅演示。

运行：
  # 生产：起 Postgres 后多实例共享状态
  #   docker compose up -d postgres
  #   每个实例配 POSTGRES_URI=postgresql://agent:agent@<host>:5432/agent
  python main.py --message "记住 A=1" --thread-id conv1
  python main.py --message "A 是多少？" --thread-id conv1
  python main.py --serve --port 8015
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

import threading
from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from core import Settings, get_settings
from core.checkpoint import build_checkpoint_saver, checkpoint_kind
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case15")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


class ThreadIdLockRegistry:
    """按 thread_id 分桶的进程内锁：同一会话串行，不同会话并行。"""

    def __init__(self) -> None:
        self._locks: dict[str, threading.Lock] = {}
        self._guard = threading.Lock()

    def get(self, thread_id: str) -> threading.Lock:
        with self._guard:
            return self._locks.setdefault(thread_id, threading.Lock())


def build_graph(settings: Settings, model=None, saver=None):
    if model is None:
        model = build_chat_model(settings)

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model.invoke(state["messages"])]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    if saver is None:
        saver, need_setup = build_checkpoint_saver(settings)
        if need_setup:
            saver.setup()
            log.info("已初始化 Postgres checkpoint 表")
    compiled = builder.compile(checkpointer=saver)
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model, saver=saver)
    locks = ThreadIdLockRegistry()

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        tid = thread_id or "default"
        config = {"configurable": {"thread_id": tid}}
        # 同一会话的写入串行化，防止并发覆盖；不同会话互不阻塞
        with locks.get(tid):
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
    run_agent_cli(lambda: build_runner(_settings), service_name="case15-distributed-checkpoint")
