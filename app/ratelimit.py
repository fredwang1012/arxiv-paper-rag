"""Spend limits for POST /api/ask: questions per client per hour, and questions per day across all clients.

In memory, per process: enough for one small server. Behind a proxy, run uvicorn with --proxy-headers (and
--forwarded-allow-ips set to the proxy) so the client address is the visitor's, not the proxy's.
"""
import threading
import time
from collections import defaultdict, deque

HOUR, DAY = 3600.0, 86400.0


class RateLimiter:
    def __init__(self, per_client_hour: int, per_day: int, clock=time.monotonic):
        self.per_client_hour, self.per_day, self._clock = per_client_hour, per_day, clock
        self._clients: dict[str, deque[float]] = defaultdict(deque)
        self._all: deque[float] = deque()
        self._lock = threading.Lock()

    def check(self, client: str) -> float | None:
        """Record one question and return None, or return the seconds to wait when a limit is reached."""
        now = self._clock()
        with self._lock:
            mine = self._clients[client]
            while mine and now - mine[0] >= HOUR:
                mine.popleft()
            while self._all and now - self._all[0] >= DAY:
                self._all.popleft()
            if self.per_day and len(self._all) >= self.per_day:
                return DAY - (now - self._all[0])
            if self.per_client_hour and len(mine) >= self.per_client_hour:
                return HOUR - (now - mine[0])
            mine.append(now)
            self._all.append(now)
            if len(self._clients) > 10_000:  # forget idle clients so the table cannot grow without bound
                for key in [k for k, v in self._clients.items() if not v]:
                    del self._clients[key]
            return None
