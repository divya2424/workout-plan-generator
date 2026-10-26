"""Live QE checks against real Groq (`uv run pytest -m live tests/test_qe_live.py`, 2 calls)."""

import os

import pytest
from dotenv import load_dotenv

from backend.core.constants import DISCLAIMER_MARKER
from backend.core.timing import estimate_day_minutes
from backend.core.utils import (
    find_equipment_conflicts,
    find_limitation_conflicts,
    list_exercises,
    non_training_days,
)
from backend.llm.planner import generate_workout_plan
from tests.qe_plans import day_count

load_dotenv()
pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="GROQ_API_KEY not set"),
]


@pytest.mark.parametrize(
    ("days", "minutes", "equipment", "limitations"),
    [(1, 15, ["No equipment"], "bad knees"), (7, 60, ["Home dumbbells"], "sore wrist")],
)
def test_live_plan_meets_every_code_check(
    days: int, minutes: int, equipment: list[str], limitations: str
) -> None:
    plan = generate_workout_plan("Build muscle", "Beginner", days, equipment, minutes, limitations)
    assert day_count(plan) == days
    assert not find_limitation_conflicts(list_exercises(plan), limitations)
    assert f"{DISCLAIMER_MARKER}:" in plan
    assert not find_equipment_conflicts(list_exercises(plan), equipment)
    assert not non_training_days(plan)
    # A very short session can honestly stay over after trimming, but then the note must say so.
    over_note = next((line for line in plan.splitlines() if "still longer than your" in line), "")
    for day, m in enumerate(estimate_day_minutes(plan), start=1):
        assert round(m) <= minutes or f"Day {day} (" in over_note
