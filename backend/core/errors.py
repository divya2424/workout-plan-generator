"""Exception types shared by the backend. Every message is safe to show to the user."""


class PlanGenerationError(Exception):
    """Raised for any failure; `str(error)` is a user-friendly message."""


class RetryableError(PlanGenerationError):
    """A failure that is worth another attempt (flaky network, 5xx, truncated/malformed output).

    `model_hint`, when set, tells the model on the next attempt what was wrong with its answer,
    so a retry corrects the mistake instead of repeating it. It is never shown to the user.
    """

    def __init__(self, message: str, model_hint: str | None = None) -> None:
        super().__init__(message)
        self.model_hint = model_hint
