"""案例 15 离线测试：验证高并发下会话隔离与同会话串行安全（不花 token）。"""
from __future__ import annotations

import threading

from langgraph.checkpoint.memory import InMemorySaver

from core.fake_llm import FakeChatModel
from tests import load_main


def _echo_model():
    """模型：记住用户最后一句，用于验证状态隔离。"""
    from langchain_core.messages import HumanMessage

    def behavior(messages, bound_tools):
        last = ""
        for m in reversed(messages):
            if isinstance(m, HumanMessage):
                last = m.content
                break
        return f"已记住：{last}"
    return FakeChatModel(behavior=behavior)


def test_concurrent_distinct_threads_are_isolated():
    main = load_main("15_distributed_checkpoint")
    settings = main.Settings()
    saver = InMemorySaver()
    model = _echo_model()

    runner = main.build_runner(settings, model=model, saver=saver)

    results = {}
    errors = []

    def work(i):
        try:
            msg = f"消息#{i}-unique-{i}"
            ans, _meta = runner(msg, thread_id=f"conv-{i}")
            results[i] = (msg, ans)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"并发出错: {errors}"
    # 每个会话的答案只含它自己的输入，互不串台
    for i in range(20):
        msg, ans = results[i]
        assert msg in ans, f"会话 {i} 被其他会话污染: {ans}"


def test_same_thread_serialized_consistent():
    main = load_main("15_distributed_checkpoint")
    settings = main.Settings()
    saver = InMemorySaver()
    model = _echo_model()

    runner = main.build_runner(settings, model=model, saver=saver)

    answers = []

    def work():
        answers.append(runner("并发写", thread_id="same")[0])

    threads = [threading.Thread(target=work) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # 同会话串行化：至少能正常跑完且都包含最新一句
    for ans in answers:
        assert "并发写" in ans
