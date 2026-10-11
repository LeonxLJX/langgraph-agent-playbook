"""案例 08 离线测试：验证人工审批断点续跑（不花 token）。"""
from __future__ import annotations

from langgraph.checkpoint.memory import InMemorySaver

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


def test_approve_executes_sensitive_tool():
    main = load_main("08_human_in_the_loop")
    main._transfers_done["n"] = 0
    settings = main.Settings()
    saver = InMemorySaver()
    model = _model([
        [{"name": "transfer_money", "args": {"account": "小王", "amount": 100}}],
        "转账流程已结束。",
    ])
    runner, approver = main.build_approval_flow(settings, model=model, saver=saver)

    # 第一轮：应暂停，返回提案
    _answer, meta = runner("转账 100 元给小王", thread_id="t1")
    assert meta["needs_approval"] is True
    assert meta["pending_approval"]["proposal"][0]["tool"] == "transfer_money"
    assert main._transfers_done["n"] == 0, "审批前绝不能执行"

    # 审批通过后续跑：真正执行
    _answer2, meta2 = approver("t1", approve=True)
    assert meta2["decision"] == "approve"
    assert main._transfers_done["n"] == 1, "审批通过后应执行转账"


def test_reject_does_not_execute_sensitive_tool():
    main = load_main("08_human_in_the_loop")
    main._transfers_done["n"] = 0
    settings = main.Settings()
    saver = InMemorySaver()
    model = _model([
        [{"name": "transfer_money", "args": {"account": "小王", "amount": 100}}],
        "用户拒绝后，我取消了转账。",
    ])
    runner, approver = main.build_approval_flow(settings, model=model, saver=saver)

    _, meta = runner("转账 100 元给小王", thread_id="t2")
    assert meta["needs_approval"] is True

    answer, meta2 = approver("t2", approve=False)
    assert meta2["decision"] == "reject"
    assert main._transfers_done["n"] == 0, "拒绝后绝不能执行"
    # 模型应收到 [未批准] 回执
    assert "取消" in answer
