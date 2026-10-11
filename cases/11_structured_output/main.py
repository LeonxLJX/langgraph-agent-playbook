"""
案例 11 —— 结构化输出校验重试
==============================
难点：业务要的不只是“一段话”，而是**带类型的结构化数据**（JSON/对象）。模型经常
      输出少字段、类型错、甚至是纯文本 —— 直接往下游传必然崩。

生产级解法：Pydantic 校验 + 带反馈重试
  1. 定义强类型 schema（pydantic BaseModel），作为输出的唯一真相。
  2. 对模型输出做 model_validate_json 校验；失败时**把校验错误作为反馈消息**塞回对话，
     让模型带着错误提示重试 —— 比简单“再来一次”收敛更快。
  3. 设置重试上限 max_struct_retries，防止无限自纠；超限后优雅返回 None 并记录 attempts。
  4. 生产可用 model.with_structured_output(schema) 让模型原生出结构，但**仍要本地校验**兜底。

运行：
  python main.py --message "北京今天天气如何"
  python main.py --serve --port 8000
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from typing import Annotated, TypedDict

from langchain_core.messages import BaseMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from core import Settings, get_settings
from core.cli import run_agent_cli
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner
from core.tracing import enable_tracing

log = get_logger("case11")


class WeatherReport(BaseModel):
    """结构化输出的唯一 schema。"""
    city: str = Field(..., description="城市名")
    celsius: float = Field(..., description="摄氏温度，数值")
    condition: str = Field(..., description="天气状况，如 晴/雨/多云")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    structured: dict | None  # 校验通过后的结果
    attempts: int


def build_graph(settings: Settings, model=None):
    if model is None:
        model = build_chat_model(settings)
    model_with_tools = model.bind_tools([])

    def extract_node(state: AgentState) -> dict:
        attempts = 0
        for _ in range(settings.max_struct_retries):
            attempts += 1
            resp = model_with_tools.invoke(state["messages"])
            content = resp.content
            try:
                if isinstance(content, str):
                    obj = WeatherReport.model_validate_json(content)
                else:
                    obj = WeatherReport.model_validate(content)
                log.info("结构化输出第 %d 次校验通过", attempts)
                return {"structured": obj.model_dump(), "attempts": attempts}
            except Exception as exc:  # noqa: BLE001
                log.warning("结构化输出校验失败(#%d): %s", attempts, exc)
                # 把校验错误作为反馈喂给模型，重试时带着错误修正
                feedback = ToolMessage(
                    content=(
                        f"输出不符合 schema: {exc}。请只返回合法 JSON，"
                        f'字段为 {{"city": 字符串, "celsius": 数字, "condition": 字符串}}。'
                    ),
                    tool_call_id=f"feedback_{attempts}",
                )
                state["messages"] = state["messages"] + [feedback]

        # 重试耗尽：优雅返回 None，不崩图
        log.error("结构化输出重试耗尽（%d 次）", attempts)
        return {"structured": None, "attempts": attempts}

    builder = StateGraph(AgentState)
    builder.add_node("extract", extract_node)
    builder.add_edge(START, "extract")
    builder.add_edge("extract", END)

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        config = {"configurable": {"thread_id": thread_id or "demo"}}
        result = graph.invoke(
            {"messages": [{"role": "user", "content": message}], "structured": None, "attempts": 0},
            config,
        )
        if result["structured"] is None:
            return "无法生成合规的结构化结果。", {"attempts": result["attempts"], "ok": False}
        obj = result["structured"]
        text = f"{obj['city']} {obj['celsius']}°C，{obj['condition']}"
        return text, {"structured": obj, "attempts": result["attempts"], "ok": True}

    return runner


if __name__ == "__main__":
    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)
    run_agent_cli(lambda: build_runner(_settings), service_name="case11-structured-output")
