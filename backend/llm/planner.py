"""Public operations: generate a workout plan and swap one exercise.

Each operation validates its input, builds the prompt, and hands the model call to
retry.py. Nothing here imports Streamlit, so it can be tested and reused on its own.
Every failure is raised as a `PlanGenerationError` carrying a message that is safe to
show directly to the user.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from backend.core.constants import (
    BASE_MAX_TOKENS,
    DEFAULT_SESSION_MINUTES,
    MAX_SWAP_REASON_CHARS,
    MAX_TOKENS_CAP,
    SWAP_MAX_TOKENS,
    TOKENS_PER_DAY,
)
from backend.core.errors import PlanGenerationError
from backend.core.models import PlanRequest
from backend.core.timing import fit_to_session
from backend.core.utils import (
    check_plan,
    list_exercises,
    parse_swap_row,
    resolve_credentials,
    table_name,
    validate_inputs,
)
from backend.llm.client import Messages
from backend.llm.prompts import build_swap_prompt, build_user_prompt, swap_system_prompt, system_prompt
from backend.llm.retry import complete_with_retries

logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR = "Something went wrong while generating your plan."


def plan_max_tokens(days_per_week: int) -> int:
    """Output budget for a plan: it grows with the number of days (see constants.py)."""
    return min(MAX_TOKENS_CAP, BASE_MAX_TOKENS + TOKENS_PER_DAY * days_per_week)


def generate_workout_plan(
    goal: str,
    experience: str,
    days_per_week: int,
    equipment: Sequence[str],
    session_minutes: int = DEFAULT_SESSION_MINUTES,
    limitations: str = "",
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Generate a weekly workout plan in Markdown.

    Validates the inputs, builds the prompt, calls Groq and checks the response.
    Never raises anything but PlanGenerationError, whose message is safe to show to the user.
    """
    try:
        request = PlanRequest(goal, experience, days_per_week, tuple(equipment), session_minutes, limitations)
        problems = validate_inputs(request)
        if problems:
            raise PlanGenerationError(" ".join(problems))

        key, model_name = resolve_credentials(api_key, model)
        messages: Messages = [
            {"role": "system", "content": system_prompt()},
            {"role": "user", "content": build_user_prompt(request)},
        ]
        return complete_with_retries(
            messages,
            api_key=key,
            model=model_name,
            max_tokens=plan_max_tokens(request.days_per_week),
            parse=lambda raw: fit_to_session(
                check_plan(raw, request.days_per_week, request.limitations, equipment=request.equipment),
                request.session_minutes,
            ),
        )
    except PlanGenerationError:
        raise
    except Exception as exc:  # a bug or odd input must not crash the app
        logger.exception("unexpected error while generating a plan")
        raise PlanGenerationError(_UNEXPECTED_ERROR) from exc


def _find_rows(lines: list[str], exercise: str) -> list[int]:
    """Indexes of the table rows whose exercise name matches `exercise` (case-insensitive)."""
    target = exercise.lower()
    return [i for i, line in enumerate(lines) if (name := table_name(line)) and name.lower() == target]


def swap_exercise(
    plan: str,
    exercise: str,
    reason: str,
    request: PlanRequest,
    *,
    api_key: str | None = None,
    model: str | None = None,
) -> str:
    """Return `plan` with every table row for `exercise` replaced.

    Only the replacement row is requested from the model and patched into the plan, so a
    swap is small, fast, and cannot alter the rest of the plan. The row gets the same checks
    as a plan's rows, and the session times are recomputed for `request.session_minutes`
    (pass the request that produced the plan).
    """
    try:
        exercise = exercise.strip()
        lines = plan.splitlines()
        rows = _find_rows(lines, exercise)
        if not rows:
            raise PlanGenerationError(f"'{exercise}' doesn't appear in the current plan.")

        key, model_name = resolve_credentials(api_key, model)
        existing = list_exercises(plan)
        messages: Messages = [
            {"role": "system", "content": swap_system_prompt()},
            {
                "role": "user",
                "content": build_swap_prompt(
                    request, lines[rows[0]].strip(), existing, reason.strip()[:MAX_SWAP_REASON_CHARS]
                ),
            },
        ]
        new_row = complete_with_retries(
            messages,
            api_key=key,
            model=model_name,
            max_tokens=SWAP_MAX_TOKENS,
            parse=lambda raw: parse_swap_row(raw, exercise, existing, request.limitations, request.equipment),
        )
        for i in rows:
            lines[i] = new_row
        # The new row changes the day's length: recompute (and trim) it like a fresh plan.
        return fit_to_session("\n".join(lines), request.session_minutes)
    except PlanGenerationError:
        raise
    except Exception as exc:  # a bug or odd input must not crash the app
        logger.exception("unexpected error while swapping an exercise")
        raise PlanGenerationError(_UNEXPECTED_ERROR) from exc
