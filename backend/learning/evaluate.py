"""Score raw model drafts against the plan rules, before any code-level repair.

The app repairs many mistakes after the fact (see check_plan and timing.py). Here the same
checks only count them, so the learning loop can tell whether a prompt change made the model
itself better.
"""

from __future__ import annotations

from backend.core.constants import (
    DAY_HEADING_PATTERN,
    DISCLAIMER_MARKER,
    EQUIPMENT_DUMBBELLS,
    EQUIPMENT_GYM,
    EQUIPMENT_NONE,
    MIN_SESSION_FILL,
)
from backend.core.models import PlanRequest
from backend.core.timing import estimate_day_minutes
from backend.core.utils import (
    find_equipment_conflicts,
    find_limitation_conflicts,
    find_session_text_conflicts,
    is_stretch,
    join_and,
    list_exercises,
    non_training_days,
)

# Fixed test clients covering every goal and level, short and long sessions, and the guarded
# limitations. They are ours, not users', so their text is safe to show the model when learning.
PROFILES: tuple[PlanRequest, ...] = (
    PlanRequest("Lose fat", "Beginner", 3, (EQUIPMENT_NONE,), 15, ""),
    PlanRequest("Build muscle", "Intermediate", 4, (EQUIPMENT_DUMBBELLS,), 20, ""),
    PlanRequest("Improve endurance", "Intermediate", 3, (EQUIPMENT_DUMBBELLS,), 40, "bad knees"),
    PlanRequest("General fitness", "Beginner", 2, (EQUIPMENT_NONE,), 45, "sore wrist"),
    PlanRequest("Build muscle", "Advanced", 5, (EQUIPMENT_GYM,), 60, "no overhead pressing"),
    PlanRequest("Lose fat", "Beginner", 7, (EQUIPMENT_NONE,), 30, "lower back pain"),
)


def describe(request: PlanRequest) -> str:
    """One line naming a test client, e.g. "Lose fat, Beginner, 3 days x 15 min, No equipment"."""
    text = (
        f"{request.goal}, {request.experience}, {request.days_per_week} days x "
        f"{request.session_minutes} min, {' + '.join(request.equipment)}"
    )
    return f"{text}, limitation: {request.limitations}" if request.limitations else text


def find_issues(draft: str, request: PlanRequest) -> list[str]:
    """Every rule the draft breaks, in words the model can learn from (empty if none)."""
    issues: list[str] = []
    days = [int(n) for n in DAY_HEADING_PATTERN.findall(draft)]
    if days != list(range(1, request.days_per_week + 1)):
        issues.append(f"wrote {len(days)} training days instead of {request.days_per_week}")

    names = list_exercises(draft)
    if conflicts := find_limitation_conflicts(names, request.limitations):
        issues.append(f"programmed {', '.join(conflicts)} despite the limitation")
    if gear := find_equipment_conflicts(names, request.equipment):
        issues.append(f"programmed {', '.join(gear)}, which needs equipment the client doesn't have")
    if parts := find_session_text_conflicts(draft, request.limitations):
        issues.append(f"put movements banned by the limitation in the {join_and(parts)}")
    if stretches := [name for name in names if is_stretch(name)]:
        issues.append(f"put stretches in the exercise table ({', '.join(stretches)})")
    if empty := non_training_days(draft):
        named = join_and([f"Day {n}" for n in empty])
        issues.append(f"{named} had no strength or conditioning exercise (rest or stretching only)")

    target = request.session_minutes
    for day, minutes in enumerate(estimate_day_minutes(draft), start=1):
        if round(minutes) > target:
            issues.append(
                f"Day {day} adds up to about {round(minutes)} min, over the {target}-minute session"
            )
        elif minutes < MIN_SESSION_FILL * target:
            issues.append(
                f"Day {day} adds up to only about {round(minutes)} min of the {target}-minute session"
            )

    if request.limitations and DISCLAIMER_MARKER not in draft:
        issues.append("left out the disclaimer although the client listed a limitation")
    return issues
