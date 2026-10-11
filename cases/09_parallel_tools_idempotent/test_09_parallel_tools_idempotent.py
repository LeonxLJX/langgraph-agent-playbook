"""案例 09 离线测试：验证并行执行 + 幂等去重（不花 token）。"""
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


def test_dedupe_and_parallel_execute():
    main = load_main("09_parallel_tools_idempotent")
    main._weather_calls["n"] = 0
    main._time_calls["n"] = 0
    settings = main.Settings()

    # 模型一次返回：北京×2（重复，应去重只执行1次）+ UTC（不同签名，独立执行）
    model = _model([
        [
            {"name": "get_weather", "args": {"city": "北京"}},
            {"name": "get_weather", "args": {"city": "北京"}},
            {"name": "get_time", "args": {"timezone": "UTC"}},
        ],
        "三地都查好了。",
    ])
    graph = main.build_graph(settings, model=model)
    result = graph.invoke({"messages": [{"role": "user", "content": "查天气和时间"}]})

    # 幂等：相同调用只执行一次
    assert main._weather_calls["n"] == 1, "重复的北京调用应只执行一次"
    assert main._time_calls["n"] == 1
    # 所有调用都有结果回执
    tool_msgs = [m for m in result["messages"] if m.type == "tool"]
    assert len(tool_msgs) == 3, "3 个调用都应有回执"
    contents = {m.content for m in tool_msgs}
    assert "北京: 晴 22°C" in contents
    assert "UTC: 12:00" in contents
    assert result["messages"][-1].content == "三地都查好了。"


def test_two_different_calls_both_executed():
    main = load_main("09_parallel_tools_idempotent")
    main._weather_calls["n"] = 0
    main._time_calls["n"] = 0
    settings = main.Settings()

    model = _model([
        [
            {"name": "get_weather", "args": {"city": "北京"}},
            {"name": "get_weather", "args": {"city": "上海"}},
        ],
        "两个城市都查了。",
    ])
    graph = main.build_graph(settings, model=model)
    result = graph.invoke({"messages": [{"role": "user", "content": "查北京上海"}]})

    assert main._weather_calls["n"] == 2, "不同参数的两个调用应各自执行"
    assert len([m for m in result["messages"] if m.type == "tool"]) == 2
