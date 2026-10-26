"""Offline latency of the app's own processing, Groq mocked.

    uv run python -m qe.bench_offline [--runs N]

Reports p50/p95/p99/max in milliseconds for check_plan, fit_to_session, prompt rendering and
the full generate_workout_plan path (happy path and the retry path), measured with
time.perf_counter. The mock answers instantly, so these numbers are the app's overhead only.
"""

from __future__ import annotations

import argparse
import logging
import statistics
import time
from collections.abc import Callable
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from backend.core.timing import fit_to_session
from backend.core.utils import check_plan
from backend.llm import agent_file
from backend.llm import client as groq_client
from backend.llm.planner import generate_workout_plan
from backend.llm.prompts import build_user_prompt, system_prompt
from tests.helpers import make_request
from tests.qe_plans import SAFE_ROWS, build_plan

LONG_ROWS = [
    *SAFE_ROWS,
    *(f"| Move {k} | 4 | 12 | 90 s | Brace |" for k in range(6)),
    "| Jump Squat | 3 | 10 | 60 s | x |",
]
PLAN_7 = build_plan(7, LONG_ROWS)  # 7 days x 11 rows, one banned exercise per day, overlong days
PLAN_1 = build_plan(1)


def percentile(sorted_ms: list[float], q: float) -> float:
    k = max(0, min(len(sorted_ms) - 1, round(q * (len(sorted_ms) - 1))))
    return sorted_ms[k]


def bench(
    name: str, fn: Callable[[], object], runs: int, errors: Callable[[], int] = lambda: 0
) -> dict[str, object]:
    for _ in range(min(50, runs)):  # warm-up
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return {
        "scenario": name,
        "n": runs,
        "p50": percentile(samples, 0.50),
        "p95": percentile(samples, 0.95),
        "p99": percentile(samples, 0.99),
        "max": samples[-1],
        "mean": statistics.fmean(samples),
        "errors": errors(),
    }


def fake(content: str) -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason="stop")
    return SimpleNamespace(choices=[choice], usage=None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=2000)
    runs = parser.parse_args().runs
    logging.disable(logging.CRITICAL)  # the retry path logs a warning per attempt

    request = make_request(days_per_week=7, session_minutes=45, limitations="bad knees")
    results = [
        bench("check_plan, 1 day", lambda: check_plan(PLAN_1, 1, ""), runs),
        bench("check_plan, 7 days + repair", lambda: check_plan(PLAN_7, 7, "bad knees"), runs),
        bench("fit_to_session, 1 day fits", lambda: fit_to_session(PLAN_1, 60), runs),
        bench("fit_to_session, 7 days trimmed", lambda: fit_to_session(PLAN_7, 30), runs),
        bench("prompts, cached agent file", lambda: (system_prompt(), build_user_prompt(request)), runs),
    ]

    def cold() -> None:
        agent_file.reload()
        system_prompt()
        build_user_prompt(request)

    results.append(bench("prompts, file re-read", cold, runs))

    failures = 0

    def run(responses: list[str], days: int) -> Callable[[], object]:
        def go() -> object:
            nonlocal failures
            client = MagicMock()
            client.chat.completions.create.side_effect = [fake(r) for r in responses]
            groq_client.get_client.cache_clear()
            with patch.object(groq_client, "Groq", return_value=client):
                try:
                    return generate_workout_plan(
                        "Build muscle", "Beginner", days, ["Home dumbbells"], 45, "bad knees", api_key="k"
                    )
                except Exception:
                    failures += 1
                    return None

        return go

    for name, responses, days in [
        ("generate, 1 day, mocked", [PLAN_1], 1),
        ("generate, 7 days, mocked", [PLAN_7], 7),
        ("generate, 7 days, retry (wrong day count first)", [build_plan(6), PLAN_7], 7),
    ]:
        failures = 0
        results.append(bench(name, run(responses, days), runs, lambda: failures))  # noqa: B023

    print(f"{'scenario':50} {'n':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8} {'errors':>6}  (ms)")
    for r in results:
        print(
            f"{r['scenario']:50} {r['n']:>6} {r['p50']:>8.3f} {r['p95']:>8.3f} {r['p99']:>8.3f} "
            f"{r['max']:>8.3f} {r['errors']:>6}"
        )


if __name__ == "__main__":
    main()
