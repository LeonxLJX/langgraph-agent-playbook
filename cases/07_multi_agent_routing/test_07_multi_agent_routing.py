"""案例 07 离线测试：验证 Supervisor 路由与专业 Agent 上下文隔离（不花 token）。"""
from __future__ import annotations

from core.fake_llm import FakeChatModel
from tests import load_main


def _model(fn):
    return FakeChatModel(behavior=fn)


def _not_expected(name):
    def behavior(messages, bound_tools):
        raise AssertionError(f"{name} 不应被调用（路由/隔离错误）")
    return behavior


def test_route_to_writer_and_isolate_research():
    main = load_main("07_multi_agent_routing")
    settings = main.Settings()

    supervisor = _model(lambda m, b: "writer")
    writer = _model(lambda m, b: "产品文案：新品上市，抢鲜体验！")
    research = _model(_not_expected("research"))

    graph = main.build_graph(settings, model=supervisor, research_model=research, writer_model=writer)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "写一段新品宣传文案"}], "active_agent": ""}
    )

    assert result["active_agent"] == "writer"
    assert "新品上市" in result["messages"][-1].content


def test_route_to_research_and_isolate_writer():
    main = load_main("07_multi_agent_routing")
    settings = main.Settings()

    supervisor = _model(lambda m, b: "research")
    research = _model(lambda m, b: "竞品A 市占率 30%，增速 12%。")
    writer = _model(_not_expected("writer"))

    graph = main.build_graph(settings, model=supervisor, research_model=research, writer_model=writer)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": "查一下竞品数据"}], "active_agent": ""}
    )

    assert result["active_agent"] == "research"
    assert "市占率" in result["messages"][-1].content
