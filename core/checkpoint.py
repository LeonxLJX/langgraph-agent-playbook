"""Checkpoint 持久化工厂。

生产要点：
- 内存 InMemorySaver 只适合单进程演示，重启即丢。
- 生产建议 PostgresSaver：跨实例共享状态、支持断点续跑/时间旅行/人工介入。
- 这里做统一入口，未来可扩展 Redis / 云原生 saver。
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from core.settings import Settings

if TYPE_CHECKING:
    from langgraph.checkpoint.base import BaseCheckpointSaver


def build_checkpoint_saver(
    settings: Settings,
) -> tuple[BaseCheckpointSaver, bool]:
    """返回 (saver, 是否需要显式 setup)。

    PostgresSaver 需要先调用 setup()（建表）。调用方按返回的 bool 决定是否 setup。
    """
    if settings.postgres_uri:
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
            from psycopg import Connection
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "使用 Postgres checkpoint 需要安装: pip install langgraph-checkpoint-postgres psycopg[binary]"
            ) from exc

        conn: Connection = Connection.connect(settings.postgres_uri)
        saver = PostgresSaver(conn)
        return saver, True  # 需要 setup()

    from langgraph.checkpoint.memory import InMemorySaver

    return InMemorySaver(), False


def checkpoint_kind(settings: Settings) -> Literal["memory", "postgres"]:
    return "postgres" if settings.postgres_uri else "memory"
