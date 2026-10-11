"""案例 04 离线测试：验证幻觉工具名拦截与有界自纠（不花 token）。"""
from __future__ import annotations

from core.fake_llm import FakeChatModel
from tests import load_main


def _model(handlers):
    it = iter(handlers)

    def behavior(messages, bound_tools):
        try:
            return next(it)
        except StopIteration:
            raise AssertionError("模型调用次数超过预期")

    return FakeChatModel(behavior=behavior)


def _run(main, settings, model):
    graph = main.build_graph(settings, model=model)
    return graph.invoke(
        {"messages": [{"role": "user", "content": "帮我发邮件给老板"}], "invalid_calls": 0}
    )


def test_hallucinated_tool_rejected_then_self_correct():
    """核心：模型先幻觉非法工具，被拦截并提示后，改用合法工具成功收尾。"""
    main = load_main("04_hallucinated_tool")
    settings = main.Settings(max_invalid_tool_calls=3)

    model = _model([
        [{"name": "send_email", "args": {"to": "boss"}}],   # 幻觉工具
        [{"name": "get_weather", "args": {"city": "北京"}}],  # 自纠为合法工具
        "我没有发邮件工具，帮你查了北京天气：晴 22°C。",
    ])
    result = _run(main, settings, model)

    assert result["messages"][-1].type == "ai"
    assert "晴" in result["messages"][-1].content
    err = [m for m in result["messages"] if m.type == "tool" and "[工具不存在]" in m.content]
    assert err, "应当回传工具不存在提示"
    assert result["invalid_calls"] == 1


def test_invalid_calls_bounded_force_final():
    """核心：非法调用达到上限后强制收尾，不会无限自纠。"""
    main = load_main("04_hallucinated_tool")
    settings = main.Settings(max_invalid_tool_calls=2)

    model = _model([
        [{"name": "send_email", "args": {}}],
        [{"name": "send_email", "args": {}}],
        "我没有发邮件能力，直接回答：无法完成。",
    ])
    result = _run(main, settings, model)

    assert result["messages"][-1].type == "ai"
    stop = [m for m in result["messages"] if m.type == "system" and "累计" in m.content]
    assert stop, "应当注入强制收尾提示"
    assert result["invalid_calls"] == 2
    assert result["messages"][-1].content == "我没有发邮件能力，直接回答：无法完成。"
