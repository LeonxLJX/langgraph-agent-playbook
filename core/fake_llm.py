"""可控假 LLM —— 仅供离线测试使用。

目标：在**不调真实 API / 不配 key** 的前提下，把每个案例的 Agent 逻辑（重试、熔断、
schema 校验、路由、并行、持久化…）端到端跑通，从而在 CI 里可复现验证。

用法：
    fake = FakeChatModel(behavior=my_behavior)
    graph 内部用 fake 代替真实模型，behavior 决定模型“怎么说”。

behavior 签名：behavior(messages: list[BaseMessage], bound_tools: list) -> Response
Response 可以是：
  - str                         -> 模型只回文本
  - list[dict]                  -> 模型发起工具调用，每个 dict 形如 {"name":..., "args":{...}}
  - 抛异常                       -> 模拟上游失败（限流/网络错），用于重试测试

FakeChatModel 是合法的 langchain BaseChatModel 子类，可直接塞进 LangGraph 节点。
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.tools import BaseTool
from pydantic import Field

Behavior = Callable[[list[BaseMessage], list[BaseTool]], Any]
Response = str | list[dict] | BaseMessage


class FakeChatModel(BaseChatModel):
    """由脚本驱动、可注入故障的假模型。"""

    behavior: Callable[[list[BaseMessage], list[BaseTool]], Response]
    fake_tools: list[BaseTool] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "fake"

    def bind_tools(self, tools: Sequence, **kwargs: Any) -> FakeChatModel:
        """记录绑定的工具，保持 BaseChatModel 接口兼容。"""
        new = self.model_copy(deep=True)
        new.fake_tools = list(tools)
        return new

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeChatModel:
        # 结构化输出在测试里由 behavior 直接返回合法 dict 来模拟。
        return self

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> ChatResult:
        result = self.behavior(messages, self.fake_tools)
        if isinstance(result, str):
            msg: BaseMessage = AIMessage(content=result)
        elif isinstance(result, BaseMessage):
            msg = result
        elif isinstance(result, list):
            # 生成 tool_calls
            calls = []
            for i, call in enumerate(result):
                name = call["name"]
                args = call.get("args", {})
                calls.append(
                    {
                        "name": name,
                        "args": args,
                        "id": call.get("id", f"call_{i}"),
                        "type": "tool_call",
                    }
                )
            msg = AIMessage(content="", tool_calls=calls)
        else:
            raise TypeError(f"无法识别的 behavior 返回类型: {type(result)!r}")
        return ChatResult(generations=[ChatGeneration(message=msg)])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: Any = None,
        **kwargs: Any,
    ) -> Any:
        """模拟流式：文本结果按字符分批产出，便于测试 stream_mode='messages'。"""
        result = self.behavior(messages, self.fake_tools)
        if isinstance(result, str) and result:
            # 按 3 个字符一批，模拟 token 级增量
            for i in range(0, len(result), 3):
                yield ChatGenerationChunk(message=AIMessageChunk(content=result[i:i + 3]))
            return
        # 非文本（工具调用等）：退化为一次性产出
        gen = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        msg = gen.generations[0].message
        yield ChatGenerationChunk(message=AIMessageChunk(content=getattr(msg, "content", "") or ""))

    @property
    def bind_tools_enabled(self) -> bool:
        return True
