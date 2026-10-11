"""案例 14 离线测试：验证可观测性与成本指标采集（不花 token）。"""
from __future__ import annotations

from core.fake_llm import FakeChatModel
from tests import load_main


def test_run_collects_observability_metrics():
    main = load_main("14_observability")
    settings = main.Settings()  # tracing 默认关闭，不影响离线运行
    model = FakeChatModel(behavior=lambda m, b: "北京今天晴，22°C。")

    runner = main.build_runner(settings, model=model)
    answer, meta = runner("北京天气")

    assert answer == "北京今天晴，22°C。"
    assert meta["est_tokens"] > 0, "应有 token 估算"
    assert meta["steps"] >= 1
    assert meta["duration_ms"] >= 0
    # 节点级插桩至少记录了 agent 一次
    assert any(e["node"] == "agent" and e["status"] == "ok" for e in meta["node_events"])
    # tracing 未显式开启时为 False，不产生任何副作用
    assert meta["tracing_enabled"] is False


def test_instrument_captures_node_error():
    main = load_main("14_observability")
    settings = main.Settings()

    def boom(messages, bound_tools):
        raise RuntimeError("模型炸了")
    model = FakeChatModel(behavior=boom)

    graph = main.build_graph(settings, model=model, instrument=main.make_telemetry()[0])
    import pytest
    with pytest.raises(RuntimeError):
        graph.invoke({"messages": [{"role": "user", "content": "x"}]})
