"""案例 02 离线测试：验证上下文超长时触发历史压缩（不花 token）。"""
from __future__ import annotations

from langchain_core.messages import SystemMessage

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


def test_trigger_summarization_on_overflow():
    main = load_main("02_context_truncate")
    # 预算取中间值：初始长输入(~175 tokens)超阈值触发压缩；压缩后(~70)回落收尾，不会二次触发。
    settings = main.Settings(context_token_budget=100, keep_recent=1)

    long_user_text = "天气" * 200  # 超预算的长输入
    model = _model([
        "收到。",                                        # call1: agent 简短回应
        "用户发来长文本，核心诉求是天气。",              # call2: summarize 生成总结
        "已按历史总结作答：天气问题已记录。",            # call3: 压缩后再次作答
    ])
    graph = main.build_graph(settings, model=model)
    result = graph.invoke(
        {"messages": [{"role": "user", "content": long_user_text}]}
    )

    # 触发了压缩：历史里出现摘要 SystemMessage
    summaries = [
        m for m in result["messages"]
        if isinstance(m, SystemMessage) and m.content.startswith("【历史对话总结】")
    ]
    assert summaries, "应当出现历史总结 SystemMessage"
    # 摘要已替换掉长原文，上下文明显变短
    hist_len = sum(len(m.content or "") for m in result["messages"])
    assert hist_len < len(long_user_text), "压缩后总字符数应远小于原始长输入"


def test_no_summarization_under_budget():
    main = load_main("02_context_truncate")
    settings = main.Settings(context_token_budget=100000, keep_recent=1)
    model = _model(["天气不错。"])
    graph = main.build_graph(settings, model=model)
    result = graph.invoke({"messages": [{"role": "user", "content": "你好"}]})

    summaries = [
        m for m in result["messages"]
        if isinstance(m, SystemMessage) and m.content.startswith("【历史对话总结】")
    ]
    assert not summaries, "预算充足时不应触发压缩"
    assert result["messages"][-1].content == "天气不错。"
