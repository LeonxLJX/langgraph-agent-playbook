"""案例 10 离线测试：验证 checkpoint 持久化与跨实例续跑（不花 token）。"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

from core.fake_llm import FakeChatModel
from tests import load_main


def _remember_model():
    """模型1：第一次调用工具记住事实，之后返回确认（避免无限工具循环）。"""
    n = {"v": 0}

    def behavior(messages, bound_tools):
        if n["v"] == 0:
            n["v"] += 1
            return [{"name": "remember", "args": {"fact": "用户叫小王"}}]
        return "好的，记住了，你叫小王。"

    return FakeChatModel(behavior=behavior)


def _recall_model():
    """模型2：扫描历史，记得就回答，不记得就说不知道。"""
    def behavior(messages, bound_tools):
        text = " ".join(getattr(m, "content", "") or "" for m in messages)
        if "小王" in text:
            return "你叫小王。"
        return "我不记得你叫什么。"
    return FakeChatModel(behavior=behavior)


def test_state_persists_across_new_graph_instances():
    main = load_main("10_persistence_resume")
    settings = main.Settings()
    saver = InMemorySaver()

    # 用【两个不同图实例】+ 同一个 saver，模拟“进程重启/新部署后仍记得”
    graph1 = main.build_graph(settings, model=_remember_model(), saver=saver)
    r1 = graph1.invoke(
        {"messages": [{"role": "user", "content": "记住，我叫小王"}]},
        {"configurable": {"thread_id": "u1"}},
    )
    assert "记住" in r1["messages"][-1].content

    graph2 = main.build_graph(settings, model=_recall_model(), saver=saver)
    r2 = graph2.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        {"configurable": {"thread_id": "u1"}},
    )
    assert r2["messages"][-1].content == "你叫小王。", "同一线程应延续记忆"


def test_new_thread_has_no_memory():
    main = load_main("10_persistence_resume")
    settings = main.Settings()
    saver = InMemorySaver()

    main.build_graph(settings, model=_remember_model(), saver=saver).invoke(
        {"messages": [{"role": "user", "content": "记住，我叫小王"}]},
        {"configurable": {"thread_id": "u1"}},
    )

    graph2 = main.build_graph(settings, model=_recall_model(), saver=saver)
    r = graph2.invoke(
        {"messages": [{"role": "user", "content": "我叫什么？"}]},
        {"configurable": {"thread_id": "u2"}},  # 全新线程
    )
    assert r["messages"][-1].content == "我不记得你叫什么。", "不同线程不应串记忆"
