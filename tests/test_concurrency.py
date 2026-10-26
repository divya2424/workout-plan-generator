"""Concurrency behaviour of the planner: many threads share one Groq client safely."""

from __future__ import annotations

import threading
from unittest.mock import MagicMock, patch

import backend.llm.client as groq_client
from backend.llm import planner
from tests.helpers import GOOD_PLAN, VALID, fake_response


def run_threads(count: int, target) -> None:
    threads = [threading.Thread(target=target) for _ in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)


def test_concurrent_first_calls_all_succeed_with_a_shared_client() -> None:
    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **_k: fake_response(GOOD_PLAN)
    results: list[str] = []
    with patch.object(groq_client, "Groq", return_value=client):
        run_threads(20, lambda: results.append(planner.generate_workout_plan(**VALID, api_key="k")))
    assert len(results) == 20
