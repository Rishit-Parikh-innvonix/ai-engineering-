"""Checkpoints: LangGraph saves the whole state after every step, keyed by a thread id, so a run can
be stopped (a crash, Ctrl+C, closing the laptop, waiting for a human) and resumed later - even from
a brand-new process - exactly where it left off.

SQLite keeps this in one file with no server. Our state holds pydantic models, and LangGraph only
turns them back into objects if their types are registered, so they are listed explicitly below
(otherwise newer LangGraph versions refuse to load them)."""

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

ALLOWED_TYPES = [
    ("brief.sources.types", "SourceDocument"),
    ("brief.types", "NumberedSource"),
    ("brief.types", "QueryPlan"),
    ("brief.types", "RetrievedBatch"),
]


def make_serde() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=ALLOWED_TYPES)


@asynccontextmanager
async def open_checkpointer(path: str | Path) -> AsyncIterator[AsyncSqliteSaver]:
    async with aiosqlite.connect(path) as connection:
        yield AsyncSqliteSaver(connection, serde=make_serde())


async def list_threads(path: str | Path) -> list[str]:
    """Thread ids in the database, most recently written first."""
    if not Path(path).exists():
        return []
    async with aiosqlite.connect(path) as connection:
        try:
            cursor = await connection.execute(
                "SELECT thread_id FROM checkpoints GROUP BY thread_id ORDER BY MAX(rowid) DESC"
            )
        except aiosqlite.OperationalError:  # no table yet: nothing has ever been checkpointed
            return []
        return [row[0] for row in await cursor.fetchall()]
