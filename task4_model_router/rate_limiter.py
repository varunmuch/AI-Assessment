"""Token-aware sliding-window rate limiter stored in an on-disk SQLite database.

Each request reserves its estimated tokens up front. Once the provider answers,
the reservation is settled to the tokens actually used, or released if the
request failed.
"""

import time
from collections.abc import Callable

import aiosqlite

LIMIT_TOKENS = 50_000
WINDOW_SECONDS = 60.0


class RateLimiter:
    def __init__(self, db_path: str, clock: Callable[[], float] = time.time,
                 limit: int = LIMIT_TOKENS, window: float = WINDOW_SECONDS) -> None:
        self.db_path, self.clock, self.limit, self.window = db_path, clock, limit, window

    def _connect(self):
        return aiosqlite.connect(self.db_path, timeout=30, isolation_level=None)

    async def reserve(self, tenant: str, tokens: int) -> int | None:
        """Atomically check the tenant's rolling usage and reserve `tokens` if allowed.

        Returns a reservation id, or None if the request would exceed the limit.
        """
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        # One connection per call, so BEGIN IMMEDIATE serializes concurrent requests.
        async with self._connect() as db:
            await db.execute("CREATE TABLE IF NOT EXISTS usage (tenant TEXT, ts REAL, tokens INTEGER)")
            await db.execute("BEGIN IMMEDIATE")
            try:
                now = self.clock()
                await db.execute("DELETE FROM usage WHERE ts <= ?", (now - self.window,))
                cursor = await db.execute(
                    "SELECT COALESCE(SUM(tokens), 0) FROM usage WHERE tenant = ?", (tenant,)
                )
                (used,) = await cursor.fetchone()
                reservation = None
                if used + tokens <= self.limit:
                    cursor = await db.execute("INSERT INTO usage VALUES (?, ?, ?)", (tenant, now, tokens))
                    reservation = cursor.lastrowid
                await db.execute("COMMIT")
            except BaseException:
                await db.execute("ROLLBACK")
                raise
            return reservation

    async def settle(self, reservation: int, tokens: int) -> None:
        """Replace a reservation's estimate with the tokens actually used.

        The row keeps its original timestamp, so it leaves the window on schedule.
        """
        if tokens < 0:
            raise ValueError("tokens must be non-negative")
        async with self._connect() as db:
            await db.execute("UPDATE usage SET tokens = ? WHERE rowid = ?", (tokens, reservation))

    async def release(self, reservation: int) -> None:
        """Give back a reservation for a request that consumed nothing."""
        await self.settle(reservation, 0)
