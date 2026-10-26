from __future__ import annotations

import pytest

import backend.llm.client as groq_client
from backend.llm import retry


@pytest.fixture(autouse=True)
def _isolate_planner(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fresh client cache per test, and no waiting between retries.

    Only the backoff constant is zeroed: patching `time.sleep` itself would silently
    disable every sleep in the process, including the ones the concurrency tests rely on.
    """
    groq_client.get_client.cache_clear()
    monkeypatch.setattr(retry, "RETRY_BACKOFF_SECONDS", 0.0)
