"""Retry policy: re-ask the model on transient or malformed results, within a time budget."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from backend.core.constants import (
    MAX_ATTEMPTS,
    READ_TIMEOUT_SECONDS,
    RETRY_BACKOFF_SECONDS,
    TOTAL_BUDGET_SECONDS,
)
from backend.core.errors import PlanGenerationError, RetryableError
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
    """Call the model and `parse` its text, retrying RetryableError within a wall-clock budget.

    `parse` validates the raw text and may itself raise RetryableError (e.g. wrong day count).
    If that error carries a `model_hint`, the next attempt tells the model what to fix.
    Permanent errors (bad key, rate limit, unknown model) are raised immediately.
    """
    deadline = time.monotonic() + TOTAL_BUDGET_SECONDS
    last_error: PlanGenerationError | None = None
    attempt_messages = messages
    for attempt in range(1, MAX_ATTEMPTS + 1):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            raw = call_groq(
                attempt_messages, api_key, model, max_tokens, min(READ_TIMEOUT_SECONDS, remaining)
            )
            return parse(raw)
        except RetryableError as exc:
            last_error = exc
            logger.warning("attempt %d/%d failed: %s", attempt, MAX_ATTEMPTS, exc)
            if exc.model_hint:
                # Corrective retry: say what was wrong (one short line, so the request stays small).
                correction = (
                    f"A previous draft was rejected. {exc.model_hint} Write the complete answer again."
                )
                attempt_messages = [*messages, {"role": "user", "content": correction}]
            if attempt < MAX_ATTEMPTS:
                time.sleep(min(RETRY_BACKOFF_SECONDS, max(0.0, deadline - time.monotonic())))
    raise last_error or PlanGenerationError("Generating your plan took too long. Please try again.")
