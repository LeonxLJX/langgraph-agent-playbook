"""案例 03 离线测试：验证死循环熔断（不花 token）。"""
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


def _run(main, settings, model, message="帮忙查天气"):
    graph = main.build_graph(settings, model=model)
    return graph.invoke(
        {"messages": [{"role": "user", "content": message}],
         "steps": 0, "loop_count": 0, "forced": False}
    )


def test_repeat_tool_call_trips_loop_guard():
    """核心：模型反复调同一工具，连续 N 次后被熔断并给出最终答复。"""
    main = load_main("03_loop_guard")
    settings = main.Settings(loop_threshold=3, max_steps=20)

    tc = [{"name": "get_weather", "args": {"city": "北京"}}]
    # 模型前 4 次都发相同工具调用（第 4 次触发 loop_count>=3），第 5 次收尾
    model = _model([tc, tc, tc, tc, "北京晴，22°C。"])
    result = _run(main, settings, model)

    assert result["messages"][-1].type == "ai"
    assert "22°C" in result["messages"][-1].content
    assert result["steps"] <= 6, "不应无限制空转"
    # 熔断消息应注入过
    stop_msgs = [m for m in result["messages"] if m.type == "system" and "循环" in m.content]
    assert stop_msgs, "应当注入循环熔断提示"


def test_max_steps_cap_when_no_repeat():
    """核心：每次调用不同工具不构成“循环”，但步数超过上限也要熔断。"""
    main = load_main("03_loop_guard")
    settings = main.Settings(loop_threshold=100, max_steps=3)

    model = _model([
        [{"name": "get_weather", "args": {"city": "北京"}}],
        [{"name": "get_weather", "args": {"city": "上海"}}],
        [{"name": "get_weather", "args": {"city": "广州"}}],
        "已到上限，直接作答：三个城市天气都不错。",
    ])
    result = _run(main, settings, model)

    assert result["messages"][-1].type == "ai"
    assert "三个城市" in result["messages"][-1].content
    stop_msgs = [m for m in result["messages"] if m.type == "system" and "步数上限" in m.content]
    assert stop_msgs, "应当注入步数上限熔断提示"


def test_happy_path_no_guard():
    """正常路径：一步工具调用就出结果，不该误伤熔断。"""
    main = load_main("03_loop_guard")
    settings = main.Settings(loop_threshold=3, max_steps=20)
    model = _model([
        [{"name": "get_weather", "args": {"city": "北京"}}],
        "北京晴，22°C。",
    ])
    result = _run(main, settings, model)

    assert result["messages"][-1].type == "ai"
    assert result.get("forced") is False
    assert not [m for m in result["messages"] if m.type == "system" and "循环" in m.content]
