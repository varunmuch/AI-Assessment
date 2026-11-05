"""Token-aware sliding-window rate limiter stored in an on-disk SQLite database."""

import time
from collections.abc import Callable

import aiosqlite

LIMIT_TOKENS = 50_000
WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, db_path: str, clock: Callable[[], float] = time.time,
                 limit: int = LIMIT_TOKENS, window: float = WINDOW_SECONDS) -> None:
        self.db_path, self.clock, self.limit, self.window = db_path, clock, limit, window

    async def try_consume(self, tenant: str, tokens: int) -> bool:
        """Atomically check the tenant's rolling usage and record `tokens` if allowed."""
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        # One connection per call, so BEGIN IMMEDIATE serializes concurrent requests.
        async with aiosqlite.connect(self.db_path, timeout=30, isolation_level=None) as db:
            await db.execute("CREATE TABLE IF NOT EXISTS usage (tenant TEXT, ts REAL, tokens INTEGER)")
            await db.execute("BEGIN IMMEDIATE")
            try:
                now = self.clock()
                await db.execute("DELETE FROM usage WHERE ts <= ?", (now - self.window,))
                cursor = await db.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE tenant = ?", (tenant,)
                )
                (used,) = await cursor.fetchone()
                allowed = used + tokens <= self.limit
                if allowed:
                    await db.execute("INSERT INTO usage VALUES (?, ?, ?)", (tenant, now, tokens))
                await db.execute("COMMIT")
            except BaseException:
                await db.execute("ROLLBACK")
                raise
            return allowed
