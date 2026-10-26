"""Translate Groq SDK exceptions into friendly errors the UI can show as-is."""

from __future__ import annotations

import re

import groq

from backend.core.errors import PlanGenerationError, RetryableError

# (exception type, user message, worth retrying?). Order matters: subclasses come first.
_RULES: tuple[tuple[type[Exception], str, bool], ...] = (
    (
        groq.AuthenticationError,
        "Your Groq API key was rejected. Check that GROQ_API_KEY is set correctly.",
        False,
    ),
    (groq.RateLimitError, "Groq's rate limit was reached. Wait a minute and try again.", False),
    (
        groq.NotFoundError,
        "The model '{model}' isn't available on Groq. Set GROQ_MODEL to a current model.",
        False,
    ),
    (groq.APITimeoutError, "The request to Groq timed out. Please try again.", True),
    (groq.APIConnectionError, "Couldn't reach Groq. Check your internet connection and try again.", True),
)
_SERVER_ERROR_MIN_STATUS = 500
# A 429 for the daily token or request cap (shared with the live demo) lasts hours, not a minute.
_DAILY_CAP = re.compile(r"(tokens|requests) per day|\b(TPD|RPD)\b", re.IGNORECASE)
_DAILY_CAP_MESSAGE = (
    "Groq's daily token allowance for this app has been used up (it's shared with the live demo). "
    "Please try again later, it resets within 24 hours."
)


def translate_error(exc: Exception, model: str) -> PlanGenerationError:
    """Map an SDK or unexpected exception to a friendly error. Only transient failures are retryable."""
    if isinstance(exc, groq.RateLimitError) and _DAILY_CAP.search(f"{exc} {exc.body}"):
        return PlanGenerationError(_DAILY_CAP_MESSAGE)
    for exc_type, message, retryable in _RULES:
        if isinstance(exc, exc_type):
            error_cls = RetryableError if retryable else PlanGenerationError
            return error_cls(message.format(model=model))
    if isinstance(exc, groq.APIStatusError):
        error_cls = RetryableError if exc.status_code >= _SERVER_ERROR_MIN_STATUS else PlanGenerationError
        return error_cls(f"Groq returned an error (status {exc.status_code}). Please try again shortly.")
    return PlanGenerationError("Something went wrong while generating your plan.")
