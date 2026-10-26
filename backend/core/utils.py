"""Pure helpers used by the planner: input validation, prompt text, response checks, table parsing.

None of these call Groq or Streamlit, so they are cheap to test on their own.
"""

from __future__ import annotations

import os
from collections.abc import Collection

from backend.core.constants import (
    DAY_HEADING_PATTERN,
    DEFAULT_MODEL,
    DISCLAIMER_MARKER,
    EQUIPMENT_DUMBBELLS,
    EQUIPMENT_GYM,
    EQUIPMENT_NONE,
    EQUIPMENT_OPTIONS,
    EXPERIENCE_LEVELS,
    FALLBACK_DISCLAIMER,
    GOALS,
    LIMITATION_GUARDS,
    MAX_DAYS,
    MAX_LIMITATIONS_CHARS,
    MAX_SESSION_MINUTES,
    MIN_DAYS,
    MIN_SESSION_MINUTES,
    SWAP_ROW_CELLS,
    TABLE_SEPARATOR_PATTERN,
)
from backend.core.errors import PlanGenerationError, RetryableError
from backend.core.models import PlanRequest


def validate_inputs(request: PlanRequest) -> list[str]:
    """Return a list of friendly problems with the request (empty if all good)."""
    problems: list[str] = []
    if request.goal not in GOALS:
        problems.append("Please choose a fitness goal.")
    if request.experience not in EXPERIENCE_LEVELS:
        problems.append("Please choose your experience level.")
    if not isinstance(request.days_per_week, int) or not MIN_DAYS <= request.days_per_week <= MAX_DAYS:
        problems.append(f"Pick between {MIN_DAYS} and {MAX_DAYS} training days per week.")
    if not request.equipment:
        problems.append(f"Select at least one equipment option (choose '{EQUIPMENT_NONE}' if you have none).")
    elif any(item not in EQUIPMENT_OPTIONS for item in request.equipment):
        problems.append("One of the selected equipment options isn't recognised.")
    if not isinstance(request.session_minutes, int) or not (
        MIN_SESSION_MINUTES <= request.session_minutes <= MAX_SESSION_MINUTES
    ):
        problems.append(
            f"Session length should be between {MIN_SESSION_MINUTES} and {MAX_SESSION_MINUTES} minutes."
        )
    if len(request.limitations) > MAX_LIMITATIONS_CHARS:
        problems.append(f"Please keep injuries/limitations under {MAX_LIMITATIONS_CHARS} characters.")
    return problems


def describe_equipment(equipment: Collection[str]) -> list[str]:
    """Expand the equipment choices into an explicit list the model can't misread."""
    if EQUIPMENT_GYM in equipment:
        return [
            "Full commercial gym (barbells, dumbbells, benches, racks, cable stations, "
            "machines, pull-up bar, cardio machines)"
        ]
    if EQUIPMENT_DUMBBELLS in equipment:
        return [
            "A pair of adjustable dumbbells",
            "Bodyweight only otherwise (no bench, no pull-up bar, no barbell, no machines)",
        ]
    return ["Bodyweight only (no weights, no bench, no pull-up bar, no machines)"]


def check_plan(plan: str, days_per_week: int, limitations: str = "") -> str:
    """Validate the model's output and return a cleaned-up plan.

    Raises a retryable error if the response is empty, doesn't contain exactly the requested
    training days, or names an exercise that clearly conflicts with the user's limitations.
    Appends the disclaimer if the model forgot it.
    """
    plan = plan.strip()
    if not plan:
        raise RetryableError("The AI returned an empty response. Please try again.")

    day_numbers = [int(n) for n in DAY_HEADING_PATTERN.findall(plan)]
    if day_numbers != list(range(1, days_per_week + 1)):
        raise RetryableError(
            f"The AI returned a plan with {len(day_numbers)} training day(s) instead of "
            f"{days_per_week}. Please click Generate again."
        )

    conflicts = find_limitation_conflicts(list_exercises(plan), limitations)
    if conflicts:
        raise RetryableError(
            f"The AI included an exercise that conflicts with your limitations ({conflicts[0]}). "
            "Please try again."
        )

    plan = _fix_table_separators(plan)
    if limitations and f"{DISCLAIMER_MARKER}:" not in plan:
        plan = f"{plan}\n\n{FALLBACK_DISCLAIMER}"
    return plan


def _fix_table_separators(plan: str) -> str:
    """Rewrite slightly garbled table separator rows (e.g. `|---s|`) so the table still renders."""
    lines = plan.splitlines()
    for i, line in enumerate(lines):
        if TABLE_SEPARATOR_PATTERN.match(line):
            lines[i] = "|" + "---|" * len(_split_row(line))
    return "\n".join(lines)


def resolve_credentials(api_key: str | None, model: str | None) -> tuple[str, str]:
    """Pick the API key and model: explicit argument, then environment, then the default.

    A blank `GROQ_MODEL=` in `.env` counts as unset instead of becoming an empty model name.
    """
    key = api_key or os.getenv("GROQ_API_KEY")
    if not key:
        raise PlanGenerationError(
            "No Groq API key found. Add GROQ_API_KEY to a .env file (see .env.example)."
        )
    return key, model or os.getenv("GROQ_MODEL") or DEFAULT_MODEL


def _split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def table_name(line: str) -> str | None:
    """The exercise name if `line` is an exercise row of a Markdown table, else None.

    Lines that aren't table rows, and the header and separator rows, give None.
    """
    if not line.lstrip().startswith("|"):
        return None
    name = _split_row(line)[0].strip("* ")
    if not name or name.lower() == "exercise" or set(name) <= set("-: "):
        return None
    return name


def list_exercises(plan: str) -> list[str]:
    """Unique exercise names from the plan's tables, in order."""
    names = (table_name(line) for line in plan.splitlines())
    return list(dict.fromkeys(name for name in names if name))


def find_limitation_conflicts(exercise_names: list[str], limitations: str) -> list[str]:
    """Exercise names that clearly clash with a common limitation (see LIMITATION_GUARDS)."""
    if not limitations:
        return []
    conflicts: list[str] = []
    for trigger, banned, allowed in LIMITATION_GUARDS:
        if not trigger.search(limitations):
            continue
        for name in exercise_names:
            plain = name.replace("\u2011", "-").replace("\u2010", "-")
            if banned.search(plain) and not (allowed and allowed.search(plain)) and name not in conflicts:
                conflicts.append(name)
    return conflicts


def parse_swap_row(raw: str, old_name: str, existing: list[str], limitations: str = "") -> str:
    """Extract and validate the single replacement table row from the model's reply."""
    taken = {name.lower() for name in [*existing, old_name]}
    for line in raw.splitlines():
        name = table_name(line)
        cells = _split_row(line)
        if name is None or len(cells) != SWAP_ROW_CELLS:
            continue
        if name.lower() in taken:
            raise RetryableError(
                "The AI suggested an exercise that is already in your plan. Please try again."
            )
        if find_limitation_conflicts([name], limitations):
            raise RetryableError(
                f"The AI suggested '{name}', which conflicts with your limitations. Please try again."
            )
        return "| " + " | ".join(cells) + " |"
    raise RetryableError("The AI didn't return a usable replacement exercise. Please try again.")
