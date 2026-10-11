"""统一 CLI 入口：每个案例的 main.py 只负责“构建 runner”，剩下的启动逻辑共用。

支持两种模式：
  python main.py --message "..."            # 一次性跑一个输入，打印结果（学习用）
  python main.py --serve --port 8000        # 以 FastAPI 服务启动（云部署用）
"""
from __future__ import annotations

import argparse
from collections.abc import Callable

import uvicorn

from core.logging import get_logger
from core.server import AgentRunner, create_app

log = get_logger("cli")


def run_agent_cli(
    runner_factory: Callable[[], AgentRunner],
    service_name: str,
    argv: list[str] | None = None,
) -> None:
    """启动 Agent 的 CLI 或 HTTP 服务。

    runner_factory: 构建 (message, thread_id) -> (answer, meta) 的 runner 的可调用对象。
    """
    parser = argparse.ArgumentParser(prog=service_name, description="LangGraph Agent")
    parser.add_argument("--serve", action="store_true", help="以 HTTP 服务方式启动")
    parser.add_argument("--host", default="0.0.0.0", help="服务监听地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--message", help="一次性输入（不走服务模式时）")
    parser.add_argument("--thread-id", default=None, help="会话线程 id（用于 checkpoint 续跑）")
    args = parser.parse_args(argv)

    runner = runner_factory()

    if args.serve:
        app = create_app(runner, service_name=service_name)
        log.info("启动 HTTP 服务 %s 于 %s:%s", service_name, args.host, args.port)
        uvicorn.run(app, host=args.host, port=args.port)
        return

    message = args.message or input("请输入问题：").strip()
    answer, meta = runner(message, args.thread_id)
    print("\n==== 结果 ====")
    print(answer)
    if meta:
        print("\n==== 元信息 ====")
        for k, v in meta.items():
            print(f"  {k}: {v}")
