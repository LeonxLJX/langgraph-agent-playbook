"""案例 06 离线测试：验证节点异常自愈与优雅降级（不花 token）。"""
from __future__ import annotations

from core.fake_llm import FakeChatModel
from tests import load_main


def _flaky_model(call_plan):
    """call_plan: 每次调用行为序列，值为 'raise' 或返回文本。"""
    state = {"n": 0}

    def behavior(messages, bound_tools):
        idx = state["n"]
        state["n"] += 1
        action = call_plan[min(idx, len(call_plan) - 1)]
        if action == "raise":
            raise RuntimeError("上游模型瞬时故障")
        return action

    return FakeChatModel(behavior=behavior)


def _run(main, settings, model):
    graph = main.build_graph(settings, model=model)
    return graph.invoke(
        {"messages": [{"role": "user", "content": "你好"}], "errors": [], "failed": False}
    )


def test_self_heal_on_first_failure():
    """核心：第一次调用失败，重试后自愈，产出正常回答，不降级。"""
    main = load_main("06_self_healing")
    settings = main.Settings()
    model = _flaky_model(["raise", "自愈成功回答。"])
    result = _run(main, settings, model)

    assert result["failed"] is False
    assert result["messages"][-1].content == "自愈成功回答。"
    assert result["errors"] == []


def test_graceful_degrade_on_persistent_failure():
    """核心：两次都失败 -> 标记 failed、记录错误、走 fallback 优雅降级，不崩图。"""
    main = load_main("06_self_healing")
    settings = main.Settings()
    model = _flaky_model(["raise", "raise"])
    result = _run(main, settings, model)

    assert result["failed"] is True
    assert "服务暂时出现了问题" in result["messages"][-1].content
    assert len(result["errors"]) >= 1, "应记录至少一条错误"
    assert "模型调用失败" in result["errors"][0] or "agent" in result["errors"][0]


def test_normal_path_no_side_effects():
    """正常路径不误伤：一次成功，无错误、无降级。"""
    main = load_main("06_self_healing")
    settings = main.Settings()
    model = _flaky_model(["正常回答。"])
    result = _run(main, settings, model)

    assert result["failed"] is False
    assert result["messages"][-1].content == "正常回答。"
    assert result["errors"] == []
