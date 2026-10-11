"""
案例 12 —— 流式输出与增量呈现
==============================
难点：模型生成长答案时，用户不想干等几十秒；前端希望像 ChatGPT 那样**边生成边出字**。
      但 Agent 里还夹着工具调用，流式处理稍不留神就把 token 碎片漏了。

生产级解法：graph.stream + stream_mode="messages"
  1. agent 节点用 model.stream()（同步流式）产出 token 碎片。
  2. 编译图后用 graph.stream(stream_mode="messages") 拿到**消息级增量**，
     既能看到模型在打字，也能感知工具调用阶段。
  3. FastAPI 里用 SSE 把增量推给前端（sse-starlette）。
  4. 前端把碎片拼接即为完整回答；测试验证拼接结果与一次性输出一致。

运行：
  python main.py --message "介绍一下扬州"      # 终端实时打印 token
  python main.py --serve --port 8000           # /invoke/stream 走 SSE
"""
# 本地自定义 TypedDict 需要真实注解供 langgraph 求值，故不用 `from __future__ import annotations`

from collections.abc import Generator
from typing import Annotated, TypedDict

from langchain_core.messages import AIMessage, BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from starlette.requests import Request
from starlette.responses import StreamingResponse

from core import Settings, get_settings
from core.llm import build_chat_model
from core.logging import get_logger, setup_logging
from core.server import AgentRunner, create_app
from core.tracing import enable_tracing

log = get_logger("case12")


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_graph(settings: Settings, model=None):
    if model is None:
        model = build_chat_model(settings)

    def agent_node(state: AgentState) -> dict:
        # 同步流式：token 碎片经 graph.stream(stream_mode='messages') 可见
        parts: list[str] = []
        for chunk in model.stream(state["messages"]):
            text = chunk.content or ""
            if text:
                parts.append(text)
        return {"messages": [AIMessage(content="".join(parts))]}

    builder = StateGraph(AgentState)
    builder.add_node("agent", agent_node)
    builder.add_edge(START, "agent")
    builder.add_edge("agent", END)

    compiled = builder.compile()
    compiled.recursion_limit = settings.recursion_limit
    return compiled


def iter_stream(graph, message: str, thread_id: str) -> Generator[str, None, None]:
    """产出消息级增量文本（stream_mode='messages'）。

    注意：该模式会同时发出「token 碎片」和「节点返回的完整消息」，
    这里用“累积去重”跳过最后那条与已流内容完全相同的完整消息，避免前端重复拼接。
    """
    config = {"configurable": {"thread_id": thread_id}}
    seen = ""
    for msg_chunk, _meta in graph.stream(
        {"messages": [{"role": "user", "content": message}]},
        config,
        stream_mode="messages",
    ):
        text = getattr(msg_chunk, "content", "") or ""
        if not text:
            continue  # 流结束的边界 chunk
        if text == seen:
            continue  # 节点返回的完整消息：token 已流过，跳过防重复
        seen += text
        yield text


def build_runner(settings: Settings, model=None, saver=None) -> AgentRunner:
    graph = build_graph(settings, model=model)

    def runner(message: str, thread_id: str | None = None) -> tuple[str, dict]:
        tid = thread_id or "demo"
        chunks = list(iter_stream(graph, message, tid))
        return "".join(chunks), {"stream_chunks": len(chunks)}

    return runner


def _add_stream_route(graph):
    from fastapi import FastAPI

    def _register(app: FastAPI) -> None:
        @app.post("/invoke/stream")
        async def invoke_stream(req_body: dict, request: Request):
            message = req_body.get("message", "")
            tid = req_body.get("thread_id") or "demo"

            def gen():
                for piece in iter_stream(graph, message, tid):
                    yield f"data: {piece}\n\n"
                yield "data: [DONE]\n\n"

            return StreamingResponse(gen(), media_type="text/event-stream")

    return _register


def build_stream_app(settings: Settings, model=None):
    graph = build_graph(settings, model=model)
    runner = build_runner(settings, model=model)
    return create_app(
        runner,
        service_name="case12-streaming",
        extra_routes=[_add_stream_route(graph)],
    )


if __name__ == "__main__":
    import argparse

    _settings = get_settings()
    setup_logging(_settings)
    enable_tracing(_settings)

    parser = argparse.ArgumentParser(prog="case12-streaming")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    if args.serve:
        import uvicorn

        uvicorn.run(build_stream_app(_settings), host="0.0.0.0", port=args.port)
    else:
        runner = build_runner(_settings)
        answer, meta = runner("介绍一下扬州")
        print(answer)
        print("\n[stream chunks]", meta["stream_chunks"])
