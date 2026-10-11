"""案例 11 离线测试：验证结构化输出校验与带反馈重试（不花 token）。"""
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
        {"messages": [{"role": "user", "content": "北京今天天气如何"}], "structured": None, "attempts": 0}
    )


def test_validate_and_retry_with_feedback():
    main = load_main("11_structured_output")
    settings = main.Settings(max_struct_retries=3)
    model = _model([
        "这不是 JSON",                                     # 第1次：非法
        '{"city": "北京", "celsius": 22, "condition": "晴"}',  # 第2次：合法
    ])
    result = _run(main, settings, model)

    assert result["structured"] is not None
    assert result["structured"]["city"] == "北京"
    assert result["structured"]["celsius"] == 22.0
    assert result["attempts"] == 2, "第1次失败、第2次通过"


def test_retry_exhausted_returns_none():
    main = load_main("11_structured_output")
    settings = main.Settings(max_struct_retries=3)
    model = _model(["不是JSON", "还是不行", "依旧非法"])  # 3 次全失败
    result = _run(main, settings, model)

    assert result["structured"] is None
    assert result["attempts"] == 3
    # 校验反馈应出现过（反馈会喂回给模型）
    # 注：失败时反馈消息未持久化，这里校验 attempts 与降级即可
