"""案例 13 离线测试：验证复杂图编排分支路由与 schema 演进兼容（不花 token）。"""
from __future__ import annotations

from core.fake_llm import FakeChatModel
from tests import load_main


def _not_expected(name):
    def behavior(messages, bound_tools):
        raise AssertionError(f"{name} 不应被调用")
    return behavior


def test_routes_to_writer_branch_and_aggregates():
    main = load_main("13_state_schema_evolution")
    settings = main.Settings()
    classify = FakeChatModel(behavior=lambda m, b: "writer")
    writer = FakeChatModel(behavior=lambda m, b: "新品文案：限时首发，抢先体验！")
    research = FakeChatModel(behavior=_not_expected("research"))

    graph = main.build_graph(settings, model=classify, research_model=research, writer_model=writer)
    result = graph.invoke({
        "messages": [{"role": "user", "content": "写文案"}],
        "branch": "direct", "result": [], "schema_version": 1,
    })

    assert result["branch"] == "writer"
    assert "[writer]" in result["result"][0]
    assert "限时首发" in result["messages"][-1].content


def test_routes_to_research_branch():
    main = load_main("13_state_schema_evolution")
    settings = main.Settings()
    classify = FakeChatModel(behavior=lambda m, b: "research")
    research = FakeChatModel(behavior=lambda m, b: "竞品A 市占 30%。")
    writer = FakeChatModel(behavior=_not_expected("writer"))

    graph = main.build_graph(settings, model=classify, research_model=research, writer_model=writer)
    result = graph.invoke({
        "messages": [{"role": "user", "content": "查竞品"}],
        "branch": "direct", "result": [], "schema_version": 1,
    })

    assert result["branch"] == "research"
    assert "[research]" in result["result"][0]


def test_unknown_branch_falls_back_to_direct():
    main = load_main("13_state_schema_evolution")
    settings = main.Settings()
    classify = FakeChatModel(behavior=lambda m, b: "spam")  # 模型乱输出
    research = FakeChatModel(behavior=_not_expected("research"))
    writer = FakeChatModel(behavior=_not_expected("writer"))

    graph = main.build_graph(settings, model=classify, research_model=research, writer_model=writer)
    result = graph.invoke({
        "messages": [{"role": "user", "content": "随便"}],
        "branch": "direct", "result": [], "schema_version": 1,
    })

    assert result["branch"] == "direct"  # 未知分支安全回退，不崩
    assert "[direct]" in result["result"][0]
