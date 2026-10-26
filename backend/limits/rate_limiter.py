"""Thread-safe, in-memory rate limiter.

Streamlit runs each user session in its own thread but inside one process, so a
module-level instance is shared by every session. All state changes happen under one
lock, which makes "check then record" atomic. State is per process: with several
server processes each one enforces its own limits (use a shared store such as Redis
if you ever scale out).
"""

from __future__ import annotations

import math
import threading
import time
from collections import deque
from collections.abc import Callable

from backend.core.constants import (
    CLIENT_MAX_REQUESTS,
    CLIENT_WINDOW_SECONDS,
    COOLDOWN_SECONDS,
    GLOBAL_MAX_REQUESTS,
    GLOBAL_WINDOW_SECONDS,
)
from backend.core.errors import RateLimitExceeded

_EVICT_THRESHOLD = 256  # only sweep idle clients once this many are tracked


class RateLimiter:
    """Cooldown between requests, a per-client window cap and a global window cap."""

    def __init__(
        self,
        cooldown: float = COOLDOWN_SECONDS,
        client_limit: int = CLIENT_MAX_REQUESTS,
        client_window: float = CLIENT_WINDOW_SECONDS,
        global_limit: int = GLOBAL_MAX_REQUESTS,
        global_window: float = GLOBAL_WINDOW_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._cooldown = cooldown
        self._client_limit = client_limit
        self._client_window = client_window
        self._global_limit = global_limit
        self._global_window = global_window
        self._clock = clock
        self._lock = threading.Lock()
        self._clients: dict[str, deque[float]] = {}
        self._global: deque[float] = deque()

    @staticmethod
    def _prune(hits: deque[float], now: float, window: float) -> None:
        while hits and now - hits[0] >= window:
            hits.popleft()

    def acquire(self, client_id: str) -> None:
        """Record one request for `client_id`, or raise RateLimitExceeded (nothing is recorded)."""
        with self._lock:
            now = self._clock()
            self._prune(self._global, now, self._global_window)
            hits = self._clients.get(client_id) or deque()
            self._prune(hits, now, self._client_window)

            if hits and now - hits[-1] < self._cooldown:
                wait = math.ceil(self._cooldown - (now - hits[-1]))
                raise RateLimitExceeded(f"Please wait {wait}s before generating again.")
            if len(hits) >= self._client_limit:
                wait = math.ceil(self._client_window - (now - hits[0]))
                minutes = int(self._client_window // 60)
                raise RateLimitExceeded(
                    f"You've reached the limit of {self._client_limit} requests per {minutes} minutes. "
                    f"Try again in about {math.ceil(wait / 60)} min."
                )
            if len(self._global) >= self._global_limit:
                raise RateLimitExceeded(
                    "The app has reached its usage limit for now. Please try again later."
                )

            hits.append(now)
            self._global.append(now)
            self._clients[client_id] = hits
            if len(self._clients) > _EVICT_THRESHOLD:
                self._evict_idle(now)

    def _evict_idle(self, now: float) -> None:
        """Forget clients with no hits left in their window so memory stays bounded."""
        for client_id in list(self._clients):
            hits = self._clients[client_id]
            self._prune(hits, now, self._client_window)
            if not hits:
                del self._clients[client_id]


# One limiter per process, created at import time (imports are serialised, so no race).
rate_limiter = RateLimiter()
