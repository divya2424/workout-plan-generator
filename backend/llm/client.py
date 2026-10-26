"""Transport layer: one shared Groq client and a single chat completion.

Retries live in retry.py; this module makes exactly one attempt per call.
"""

import logging
import time
from functools import lru_cache

import groq
from groq import Groq
from groq.types.chat import ChatCompletion, ChatCompletionMessageParam

from backend.core.constants import (
    CONNECT_TIMEOUT_SECONDS,
    READ_TIMEOUT_SECONDS,
    REASONING_EFFORT,
    REASONING_MODEL_PREFIXES,
    TEMPERATURE,
)
from backend.core.errors import RetryableError
from backend.llm.error_mapping import translate_error

Messages = list[ChatCompletionMessageParam]

logger = logging.getLogger(__name__)


@lru_cache(maxsize=2)
def get_client(api_key: str) -> Groq:
    """One pooled, thread-safe client per API key. SDK retries are off: retry.py owns the policy."""
    return Groq(api_key=api_key, max_retries=0, timeout=READ_TIMEOUT_SECONDS)


def call_groq(
    messages: Messages,
    api_key: str,
    model: str,
    max_tokens: int,
    read_timeout: float,
) -> str:
    """Send one chat completion request and return the text.

    Maps SDK errors to friendly messages. Transient failures and truncated output raise
    RetryableError; everything else raises a plain PlanGenerationError.
    """
    started = time.monotonic()
    try:
        response = get_client(api_key).chat.completions.create(
            model=model,
            messages=messages,
            temperature=TEMPERATURE,
            max_tokens=max_tokens,
            reasoning_effort=REASONING_EFFORT if model.startswith(REASONING_MODEL_PREFIXES) else groq.omit,
            timeout=groq.Timeout(read_timeout, connect=CONNECT_TIMEOUT_SECONDS),
        )
    except Exception as exc:  # anything unexpected should still not crash the app
        friendly = translate_error(exc, model)
        logger.warning(
            "llm_call failed type=%s status=%s latency=%.2fs retryable=%s",
            type(exc).__name__,
            getattr(exc, "status_code", None),
            time.monotonic() - started,
            isinstance(friendly, RetryableError),
        )
        raise friendly from exc

    return _extract_text(response, model, time.monotonic() - started)


def _extract_text(response: ChatCompletion, model: str, latency: float) -> str:
    """Pull the text out of a completion, rejecting malformed or truncated answers."""
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RetryableError("The AI returned a response in an unexpected format.") from exc

    logger.info(
        "llm_call ok model=%s latency=%.2fs finish=%s tokens=%s",
        model,
        latency,
        choice.finish_reason,
        response.usage.total_tokens if response.usage else None,
    )
    if choice.finish_reason == "length":
        raise RetryableError("The AI's answer was cut off before the plan was finished. Please try again.")
    return content or ""
