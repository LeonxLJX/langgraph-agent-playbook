"""案例 05 离线测试：验证上游限流指数退避重试与耗尽兜底（不花 token）。"""
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


def test_retry_recovers_after_rate_limit():
    """核心：前 2 次限流被重试吸收，第 3 次成功，Agent 正常出结果。"""
    main = load_main("05_retry_backoff")
    upstream = main.FlakyAnalyticsClient(fail_times=2)
    settings = main.Settings()

    model = _model([
        [{"name": "get_metric", "args": {"metric": "DAU"}}],
        "今日 DAU 为 12345。",
    ])
    graph = main.build_graph(settings, model=model, upstream=upstream)
    result = graph.invoke({"messages": [{"role": "user", "content": "查日活"}]})

    assert result["messages"][-1].content == "今日 DAU 为 12345。"
    # 2 次失败 + 1 次成功 = 3 次真实上游调用
    assert upstream.calls == 3


def test_retry_exhausted_degrades_gracefully():
    """核心：上游持续不可用，重试耗尽后不崩图，转成可读错误消息让模型收尾。"""
    main = load_main("05_retry_backoff")
    upstream = main.FlakyAnalyticsClient(fail_times=100)
    settings = main.Settings()

    model = _model([
        [{"name": "get_metric", "args": {"metric": "DAU"}}],
        "指标查询暂时失败，我无法提供数据。",
    ])
    graph = main.build_graph(settings, model=model, upstream=upstream)
    result = graph.invoke({"messages": [{"role": "user", "content": "查日活"}]})

    assert result["messages"][-1].type == "ai"
    err = [m for m in result["messages"] if m.type == "tool" and "重试已耗尽" in m.content]
    assert err, "应出现重试耗尽的兜底错误消息"
    # 最多尝试 4 次（tenacity stop_after_attempt=4）
    assert upstream.calls == 4
