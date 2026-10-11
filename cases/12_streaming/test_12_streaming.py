"""案例 12 离线测试：验证流式增量输出（不花 token）。"""
from __future__ import annotations

from core.fake_llm import FakeChatModel
from tests import load_main


def test_stream_yields_incremental_chunks():
    main = load_main("12_streaming")
    settings = main.Settings()
    long_text = "北京今天晴，气温 22°C，适合出行。"
    model = FakeChatModel(behavior=lambda m, b: long_text)

    graph = main.build_graph(settings, model=model)
    chunks = list(main.iter_stream(graph, "北京天气", thread_id="s1"))

    assert chunks, "应产生流式增量"
    assert len(chunks) > 1, "应产生多个增量片段（而不是一次性给全）"
    assert "".join(chunks) == long_text, "拼接结果应等于完整回答"


def test_build_runner_assembles_stream():
    main = load_main("12_streaming")
    settings = main.Settings()
    text = "扬州是一座历史文化名城。"
    model = FakeChatModel(behavior=lambda m, b: text)

    runner = main.build_runner(settings, model=model)
    answer, meta = runner("介绍扬州", thread_id="s2")

    assert answer == text
    assert meta["stream_chunks"] > 1
