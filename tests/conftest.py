import pytest

import backend.llm.client as groq_client


@pytest.fixture(autouse=True)
def _fresh_client_cache() -> None:
    """Each test gets its own (mocked) Groq client instead of one cached by an earlier test."""
    groq_client.get_client.cache_clear()
