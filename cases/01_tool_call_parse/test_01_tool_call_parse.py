"""案例 01 离线测试：验证工具调用参数解析的守卫逻辑（不花 token）。"""
from __future__ import annotations

from core.fake_llm import FakeChatModel
from tests import load_main


def _make_model(handlers):
    """handlers 是按顺序执行的 behavior 列表；用后弹出。"""
    it = iter(handlers)

    def behavior(messages, bound_tools):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("测试脚本：模型被调用次数超过预期")

    return FakeChatModel(behavior=behavior)


def test_happy_path():
    """正常：先工具调用，再给出最终答复。"""
    main = load_main("01_tool_call_parse")
    model = _make_model([
        [{"name": "get_weather", "args": {"city": "北京"}}],
        "北京天气：晴 22°C。",
    ])
    graph = main.build_graph(main.get_settings(), model=model)
    result = graph.invoke({"messages": [{"role": "user", "content": "北京天气？"}]})

    assert result["messages"][-1].content == "北京天气：晴 22°C。"
    # 中间确实执行了一次工具调用并回传了 ToolMessage
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_msgs) == 1
    assert "晴" in tool_msgs[0].content


def test_guard_on_missing_required_arg():
    """难点核心：参数缺必填字段时，不能崩，而要回传错误让模型自纠。"""
    main = load_main("01_tool_call_parse")
    # 第一次：模型漏传 city（args 为空）
    model = _make_model([
        [{"name": "get_weather", "args": {}}],
        "抱歉，我漏掉了城市名，我补上：北京天气是晴 22°C。",
    ])
    graph = main.build_graph(main.get_settings(), model=model)
    result = graph.invoke({"messages": [{"role": "user", "content": "北京天气？"}]})

    # 不崩溃、正常出答案
    assert result["messages"][-1].type == "ai"
    # 错误信息以 ToolMessage 回传
    err_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert err_msgs, "缺少错误回传的 ToolMessage"
    assert "[工具调用失败]" in err_msgs[0].content


def test_guard_on_hallucinated_tool_name():
    """难点核心：模型幻觉出不存在的工具名，拦截并提示可用集合。"""
    main = load_main("01_tool_call_parse")
    model = _make_model([
        [{"name": "get_moon_phase", "args": {}}],
        "我没有这个工具，我改用 get_weather：北京晴。",
    ])
    graph = main.build_graph(main.get_settings(), model=model)
    result = graph.invoke({"messages": [{"role": "user", "content": "北京天气？"}]})

    assert result["messages"][-1].type == "ai"
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert "[工具不存在]" in tool_msgs[0].content
    assert "get_weather" in tool_msgs[0].content
