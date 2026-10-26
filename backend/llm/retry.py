"""Retry policy: re-ask the model once on transient or malformed results."""

from __future__ import annotations

import logging
from collections.abc import Callable

from backend.core.constants import MAX_ATTEMPTS, READ_TIMEOUT_SECONDS
from backend.core.errors import RetryableError
from backend.llm.client import Messages, call_groq

logger = logging.getLogger(__name__)


def complete_with_retries(
    messages: Messages,
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    parse: Callable[[str], str],
) -> str:
    """Call the model and `parse` its text, retrying RetryableError up to MAX_ATTEMPTS times.

    `parse` validates the raw text and may itself raise RetryableError (e.g. wrong day count).
    If that error carries a `model_hint`, the next attempt tells the model what to fix.
    Permanent errors (bad key, rate limit, unknown model) are raised immediately.
    """
    attempt_messages = messages
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return parse(call_groq(attempt_messages, api_key, model, max_tokens, READ_TIMEOUT_SECONDS))
        except RetryableError as exc:
            logger.warning("attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            if attempt == MAX_ATTEMPTS:
                raise
            if exc.model_hint:
                # Corrective retry: say what was wrong (one short line, so the request stays small).
                correction = (
                    f"A previous draft was rejected. {exc.model_hint} Write the complete answer again."
                )
                attempt_messages = [*messages, {"role": "user", "content": correction}]
    raise AssertionError("unreachable: the loop returns or raises")
