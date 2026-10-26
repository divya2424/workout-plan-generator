"""Exception types shared by the backend. Every message is safe to show to the user."""

from __future__ import annotations


class PlanGenerationError(Exception):
    """Raised for any failure; `str(error)` is a user-friendly message."""


class RetryableError(PlanGenerationError):
    """A failure that is worth another attempt (flaky network, 5xx, truncated/malformed output)."""


class RateLimitExceeded(PlanGenerationError):
    """The caller (or the whole app) has used up its request allowance."""
