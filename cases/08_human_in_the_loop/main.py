"""
案例 08 —— 人类介入审批（Human-in-the-Loop）
==============================================
难点：Agent 想执行“不可逆/高风险”动作（转账、删数据、发邮件、下单）。直接执行代价巨大，
      必须**暂停下来，等真人确认后再决定执行或拒绝**。

生产级解法：langgraph 的 interrupt / Command 断点续跑
  1. 定义 SENSITIVE_TOOLS 白名单，命中“敏感工具”的调用一律先进审批节点。
  2. dispatch 节点调用 interrupt({proposal})：图在此暂停，把提案交到外部（API/UI）。
  3. 外部返回 Command(resume={"approve": True/False}) 续跑：
     - approve=True  -> 真正执行工具；
     - approve=False -> 回传 [未批准]，工具**不执行**（零副作用）。
  4. 需要 checkpointer（内存或 Postgres）才能断点续跑 —— 生产用 Postgres（见案例 10）。

运行：
  python main.py --message "转账 100 元给小王"   # 会先暂停要审批
  python main.py --serve --port 8000             # 云端通过 /invoke 获得提案、/resume 续跑
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import Command, interrupt

from core import Settings, get_settings
from core.checkpoint import build_checkpoint_saver
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.tracing import enable_tracing

log = get_logger("case08")


# ---------------------------------------------- 副作用计数器（验证审批才执行）
_transfers_done = {"n": 0}


@tool
def get_balance(account: str) -> str:
    """查询账户余额。参数：account 账户名。"""
    return f"{account}: ¥10,000"

@tool
def transfer_money(account: str, amount: float) -> str:
    """向账户转账。参数：account 收款账户，amount 金额。敏感操作，需人工审批。"""
    _transfers_done["n"] += 1  # 只有审批通过才会真正执行
    return f"已向 {account} 转账 ¥{amount}"

TOOLS: list[BaseTool] = [get_balance, transfer_money]
SENSITIVE_TOOLS: set[str] = {"transfer_money"}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(settings: Settings, model=None, saver=None):
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools(TOOLS)
    tool_map = {t.name: t for t in TOOLS}

    def agent_node(state: AgentState) -> dict:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def dispatch_node(state: AgentState) -> dict:
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        sensitive = [c for c in calls if c["name"] in SENSITIVE_TOOLS]
        out: list[BaseMessage] = []

        if sensitive:
            proposal = [{"tool": c["name"], "args": c.get("args", {})} for c in sensitive]
            # 暂停：把提案交给外部，等待真人决策
            decision = interrupt({"type": "approval", "proposal": proposal})
            approved = bool(decision and decision.get("approve"))
            log.info("人工审批: %s -> %s", proposal, "通过" if approved else "拒绝")
            for c in sensitive:
                if approved:
                    try:
                        out.append(ToolMessage(content=str(tool_map[c["name"]].invoke(c["args"])), tool_call_id=c["id"]))
                    except Exception as exc:  # noqa: BLE001
                        out.append(ToolMessage(content=f"[执行失败] {exc}", tool_call_id=c["id"]))
                else:
                    out.append(ToolMessage(content="[未批准] 用户拒绝了该操作，未执行。", tool_call_id=c["id"]))
            return {"messages": out}

        # 非敏感工具：守卫执行
        for c in calls:
            t = tool_map.get(c["name"])
            if t is None:
                out.append(ToolMessage(content=f"[工具不存在] {c['name']}", tool_call_id=c["id"]))
                continue
            try:
                out.append(ToolMessage(content=str(t.invoke(c.get("args", {}))), tool_call_id=c["id"]))
            except Exception as exc:  # noqa: BLE001
                out.append(ToolMessage(content=f"[调用失败] {exc}", tool_call_id=c["id"]))
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

    if saver is None:
        saver, _ = build_checkpoint_saver(settings)
    compiled = builder.compile(checkpointer=saver)
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def _stream(graph, payload, config):
    """流式执行，捕获 interrupt 提案与最终答复。"""
    answer = None
    pending = None
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                pending = update[0].value if update else None
            elif "messages" in update:
                msgs = update["messages"]
                if msgs and msgs[-1].type == "ai":
                    answer = msgs[-1].content
    return answer, pending


def build_approval_flow(settings: Settings, model=None, saver=None):
    """返回 (runner, approver)。runner 发起会话，approver 用审批结果续跑。"""
    graph = build_graph(settings, model=model, saver=saver)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        answer, pending = _stream(graph, {"messages": [{"role": "user", "content": message}]}, config)
        if pending is not None:
            return "", {"needs_approval": True, "pending_approval": pending}
        return answer or "(无答复)", {"needs_approval": False}

    def approver(thread_id: str, approve: bool) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id}}
        answer, _ = _stream(graph, Command(resume={"approve": approve}), config)
        return answer or "(已处理)", {"needs_approval": False, "decision": "approve" if approve else "reject"}

    return runner, approver


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    runner, _ = build_approval_flow(_settings)
    run_agent_cli(lambda: runner, service_name="case08-human-in-the-loop")
