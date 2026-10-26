from __future__ import annotations

import threading

import pytest

from backend.core.errors import RateLimitExceeded
from backend.limits.rate_limiter import RateLimiter


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def make(clock: FakeClock, **overrides: float) -> RateLimiter:
    defaults = {
        "cooldown": 5,
        "client_limit": 3,
        "client_window": 60,
        "global_limit": 100,
        "global_window": 600,
    }
    return RateLimiter(**{**defaults, **overrides}, clock=clock)


def test_cooldown_blocks_rapid_requests_then_allows() -> None:
    clock = FakeClock()
    limiter = make(clock)
    limiter.acquire("a")
    with pytest.raises(RateLimitExceeded, match="wait 5s"):
        limiter.acquire("a")
    clock.now += 5
    limiter.acquire("a")


def test_clients_are_independent() -> None:
    limiter = make(FakeClock())
    limiter.acquire("a")
    limiter.acquire("b")


def test_per_client_window_cap_and_recovery() -> None:
    clock = FakeClock()
    limiter = make(clock)
    for _ in range(3):
        limiter.acquire("a")
        clock.now += 6
    with pytest.raises(RateLimitExceeded, match="limit of 3"):
        limiter.acquire("a")
    clock.now += 60  # oldest hits age out of the window
    limiter.acquire("a")


def test_global_cap_applies_across_clients() -> None:
    limiter = make(FakeClock(), global_limit=2)
    limiter.acquire("a")
    limiter.acquire("b")
    with pytest.raises(RateLimitExceeded, match="usage limit"):
        limiter.acquire("c")


def test_rejected_requests_are_not_counted() -> None:
    clock = FakeClock()
    limiter = make(clock, client_limit=1)
    limiter.acquire("a")
    for _ in range(5):
        with pytest.raises(RateLimitExceeded):
            limiter.acquire("a")
    clock.now += 61  # only the one accepted request needs to age out
    limiter.acquire("a")


def test_idle_clients_are_evicted_so_memory_stays_bounded() -> None:
    clock = FakeClock()
    limiter = make(clock, global_limit=10_000, cooldown=0, client_limit=10)
    for i in range(300):
        limiter.acquire(f"c{i}")
    clock.now += 61
    limiter.acquire("fresh")  # the sweep runs while more than the threshold are tracked
    assert len(limiter._clients) < 10


def test_global_cap_is_exact_under_concurrent_load() -> None:
    """64 threads race for 10 slots: the lock must make check-and-record atomic."""
    limiter = RateLimiter(cooldown=0, client_limit=100, client_window=60, global_limit=10, global_window=600)
    barrier = threading.Barrier(64)
    outcomes: list[bool] = []
    guard = threading.Lock()

    def worker(i: int) -> None:
        barrier.wait()
        try:
            limiter.acquire(f"client-{i}")
            ok = True
        except RateLimitExceeded:
            ok = False
        with guard:
            outcomes.append(ok)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(64)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert outcomes.count(True) == 10 and outcomes.count(False) == 54


def test_cooldown_is_exact_under_concurrent_clicks_from_one_session() -> None:
    limiter = RateLimiter(
        cooldown=60, client_limit=100, client_window=60, global_limit=100, global_window=600
    )
    barrier = threading.Barrier(32)
    results: list[bool] = []
    guard = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            limiter.acquire("same-session")
            ok = True
        except RateLimitExceeded:
            ok = False
        with guard:
            results.append(ok)

    threads = [threading.Thread(target=worker) for _ in range(32)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert results.count(True) == 1
