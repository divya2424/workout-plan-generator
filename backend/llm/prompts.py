"""Prompt builders. The prompt text itself lives in agents/workout-trainer.md (see agent_file.py).

The user prompt carries the client's profile as clearly labelled fields so the model
cannot "miss" a constraint buried in a sentence.
"""

from __future__ import annotations

from backend.core.constants import CLIENT_NOTES_TAG, DISCLAIMER_MARKER
from backend.core.models import PlanRequest
from backend.core.utils import describe_equipment
from backend.llm.agent_file import learned_rules_block, render


def _as_data(text: str) -> str:
    """User text without <client_notes> tags, so it can't close its own data block early.

    Stripped until none is left, so a nested "</client_no</client_notes>tes>" can't rebuild one.
    """
    while (stripped := CLIENT_NOTES_TAG.sub("", text)) != text:
        text = stripped
    return text.strip()


def system_prompt(learned_rules: list[str] | None = None) -> str:
    """The plan system prompt with the learned rules in it (or `learned_rules`, to try out new ones)."""
    return render(
        "system", learned_rules=learned_rules_block(learned_rules), disclaimer_marker=DISCLAIMER_MARKER
    )


def swap_system_prompt() -> str:
    return render("swap-system")


def build_user_prompt(request: PlanRequest) -> str:
    """Render the client's profile as a labelled brief for the model."""
    return render(
        "plan-request",
        goal=request.goal,
        experience=request.experience,
        days_per_week=request.days_per_week,
        session_minutes=request.session_minutes,
        equipment=describe_equipment(request.equipment),
        limitations=_as_data(request.limitations) or "None reported",
    )


def build_swap_prompt(
    request: PlanRequest,
    current_row: str,
    existing_exercises: list[str],
    reason: str,
) -> str:
    """Ask for a replacement table row for one exercise.

    Only the client's constraints, the row being replaced and the names already in the
    plan are sent, so the request stays small and the rest of the plan can't drift.
    """
    return render(
        "swap-request",
        current_row=current_row,
        goal=request.goal,
        experience=request.experience,
        equipment=describe_equipment(request.equipment),
        limitations=_as_data(request.limitations) or "None reported",
        reason=_as_data(reason) or "No reason given",
        existing=", ".join(existing_exercises),
    )
