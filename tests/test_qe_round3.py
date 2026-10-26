"""QE round 2 re-test of the qe-bug-fixes build (fix round 1): regressions next to the fixes.

Each bug (S1-S9 in work/qe-bug-fixes/06-qe-report.md, "## Round 2") is a strict xfail, so the
suite stays green and the engineer has an exact target. Groq is mocked.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.core.constants import AGENT_FILE, MAX_ATTEMPTS
from backend.core.timing import estimate_day_minutes
from backend.core.utils import _fix_table_separators, check_plan, find_equipment_conflicts, list_exercises
from backend.llm import client as groq_client
from backend.llm.agent_file import load_sections
from backend.llm.planner import generate_workout_plan, swap_exercise
from tests.helpers import VALID, fake_response, make_request, mock_client
from tests.qe_plans import HEADER, SAFE_ROWS, build_plan

NONE = ["No equipment"]
GYM = ["Full gym"]


def one_day(warm: str, rows: tuple[str, ...] | list[str] = SAFE_ROWS) -> str:
    body = "\n".join(rows)
    return (
        f"## Day 1 - A\n{warm}\n\n{HEADER}\n{body}\n\n**Cool-down (5 min):** Easy walk.\n\n"
        "## Progression\n- Add a rep.\n"
    )


def warm_up_line(plan: str) -> str:
    return next(line for line in plan.splitlines() if "warm" in line.lower())


def generate(*responses: object, **inputs: object) -> str:
    groq_client.get_client.cache_clear()
    with patch.object(groq_client, "Groq", return_value=mock_client(*responses)):
        return generate_workout_plan(**{**VALID, **inputs}, api_key="k")  # type: ignore[arg-type]


def swap(plan: str, exercise: str, reply: str, **request: object) -> str:
    groq_client.get_client.cache_clear()
    with patch.object(groq_client, "Groq", return_value=mock_client(*[fake_response(reply)] * MAX_ATTEMPTS)):
        return swap_exercise(plan, exercise, "", make_request(**request), api_key="k")


# --- S1 (unsafe): "Rest ..." rows skip every exercise check ---------------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: REST_ROW_PATTERN '^rest\\s' makes 'Rest Pause X' and 'Rest 30 s, then X' rest rows, "
    "which skip the limitation guard and the equipment check",
)
@pytest.mark.parametrize(
    ("row", "limitations", "equipment", "banned"),
    [
        ("| Rest Pause Push-ups | 3 | 10 | 60 s | |", "sore wrist", (), "Push-ups"),
        ("| Rest Pause Barbell Squat | 3 | 8 | 90 s | |", "", NONE, "Barbell"),
        ("| Rest 30 s, then 10 jump squats | 3 | 10 | 30 s | |", "bad knees", (), "jump squats"),
    ],
)
def test_a_row_starting_with_rest_still_gets_the_exercise_checks(
    row: str, limitations: str, equipment: list[str], banned: str
) -> None:
    plan = check_plan(build_plan(1, [*SAFE_ROWS, row]), 1, limitations, equipment=equipment)
    assert not any(banned in line for line in plan.splitlines() if line.startswith("|"))


# --- S2 (unsafe, low realism): a negation word excuses the whole item ------------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: NEGATED_ITEM_PATTERN skips a whole item that starts with no/don't/never, so a banned "
    "movement later in that item ('no rest before 20 jumping jacks') is kept",
)
@pytest.mark.parametrize(
    "warm",
    [
        "**Warm-up (5 min):** Brisk walk, no rest before 20 jumping jacks.",
        "**Warm-up (5 min):** Brisk walk, don't rush the 10 jump squats.",
        "**Warm-up (5 min):** Brisk walk, never skip the walking lunges.",
    ],
)
def test_a_negation_word_does_not_excuse_a_banned_movement_it_does_not_negate(warm: str) -> None:
    line = warm_up_line(check_plan(one_day(warm), 1, "bad knees"))
    assert "jump" not in line.lower() and "lunge" not in line.lower()


@pytest.mark.xfail(strict=True, reason="BUG: a Notes cell 'Don't rush the jump squats' is kept for bad knees")
def test_a_negation_word_in_notes_does_not_excuse_a_banned_movement() -> None:
    rows = ["| Glute Bridge | 3 | 10 | 60 s | Don't rush the jump squats |", *SAFE_ROWS[1:]]
    assert "jump squats" not in check_plan(one_day("**Warm-up (5 min):** Walk.", rows), 1, "bad knees")


# --- S3 (wrong plan): new stretch / equipment terms catch strength moves ---------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: STRETCH_PATTERN 'arm swing' and 'hip circle' match the kettlebell 'One-Arm Swing' and a "
    "'Hip Circle Band Walk', which are then moved out of the main table",
)
@pytest.mark.parametrize("name", ["One-Arm Swing", "Single Arm Swing", "Hip Circle Band Walk"])
def test_a_strength_move_with_a_mobility_word_in_it_is_kept(name: str) -> None:
    rows = [
        f"| {name} | 3 | 15 | 60 s | |",
        "| Goblet Squat | 3 | 10 | 60 s | |",
        "| Bench Press | 3 | 8 | 90 s | |",
        "| Lat Pulldown | 3 | 10 | 60 s | |",
    ]
    assert name in list_exercises(check_plan(build_plan(1, rows), 1, "", equipment=GYM))


@pytest.mark.xfail(
    strict=True,
    reason="BUG: 'hip (ab|ad)ductor' bans the bodyweight 'Lying Hip Adductor Squeeze' (R7 class over-block)",
)
def test_a_bodyweight_adductor_squeeze_needs_no_equipment() -> None:
    assert find_equipment_conflicts(["Lying Hip Adductor Squeeze"], NONE) == []


# --- S4 (wrong plan, low): agent-file corruption the section check doesn't see -----------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: load_sections accepts a duplicated marker (the later copy silently replaces the section), "
    "an empty section and a plan-request without {limitations}",
)
@pytest.mark.parametrize(
    ("old", "new"),
    [
        # A second "system" section further down replaces the real system prompt with "Hello".
        (
            "<!-- prompt: learned-rules -->",
            "<!-- prompt: system -->\nHello\n\n<!-- prompt: learned-rules -->",
        ),
        # An empty swap-system section.
        ("<!-- prompt: swap-system -->", "<!-- prompt: swap-system -->\n<!-- prompt: swap-system-old -->"),
        # The client's limitations are no longer sent to the model.
        ("<client_notes>\n{limitations}\n</client_notes>", "<client_notes>\n</client_notes>"),
    ],
)
def test_a_corrupted_agent_file_is_rejected(tmp_path: Path, old: str, new: str) -> None:
    copy = tmp_path / "agent.md"
    shutil.copy(AGENT_FILE, copy)
    text = copy.read_text(encoding="utf-8")
    assert old in text
    copy.write_text(text.replace(old, new, 1), encoding="utf-8")
    with pytest.raises(ValueError):
        load_sections(copy)


# --- S5 (wrong session time): a circuit header with text in its Reps cell ---------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: circuit_header needs an empty/dash Reps cell, so 'Circuit (repeat 3x) | 3 | N/A' is an "
    "exercise: listed for swapping, and its rows are counted once instead of 3 times",
)
@pytest.mark.parametrize("reps", ["N/A", "see below"])
def test_a_circuit_header_with_words_in_the_reps_cell_is_still_a_header(reps: str) -> None:
    plan = build_plan(1, [f"| Circuit (repeat 3x, 60 s rest) | 3 | {reps} | 60 s | |", *SAFE_ROWS])
    dashed = build_plan(1, ["| Circuit (repeat 3x, 60 s rest) | | - | | |", *SAFE_ROWS])
    assert list_exercises(plan) == list_exercises(dashed)
    assert estimate_day_minutes(plan) == estimate_day_minutes(dashed)


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a header row's name is guarded for limitations but never checked for equipment",
)
def test_a_circuit_header_naming_missing_equipment_is_not_kept() -> None:
    header = "| Superset (repeat 3x): Barbell Squat + Pull-up | | | 90 s | |"
    plan = build_plan(1, [header, *SAFE_ROWS])
    try:
        out = check_plan(plan, 1, "", equipment=NONE)
    except Exception:  # a retry is fine too
        return
    assert "Barbell" not in out


# --- S6 (bad message): notes after a swap on a trimmed plan -----------------------------------

MOVES = [f"| Move {k} | 4 | 12 | 90 s | |" for k in range(1, 7)]


def trimmed_plan() -> str:
    plan = generate(fake_response(build_plan(1, MOVES)), days_per_week=1, session_minutes=30)
    assert "Day 1 shortened" in plan
    return plan


@pytest.mark.xfail(
    strict=True,
    reason="BUG: after a swap makes a shortened day run over, the plan says both 'Day 1 shortened ... to "
    "fit your 30-minute session' and 'Day 1 ... is still longer than your 30-minute session'",
)
def test_a_swap_does_not_leave_a_shortened_to_fit_note_on_a_day_that_no_longer_fits() -> None:
    plan = swap(
        trimmed_plan(), "Move 1", "| Easy Run | 1 | 25 min | - | |", days_per_week=1, session_minutes=30
    )
    assert "still longer" in plan
    assert "to fit your 30-minute session" not in plan


@pytest.mark.xfail(
    strict=True,
    reason="BUG: a swap that is trimmed again adds a second 'Day 1 shortened' note next to the old one",
)
def test_a_swap_does_not_add_a_second_shortened_note_for_the_same_day() -> None:
    plan = swap(
        trimmed_plan(), "Move 2", "| Wall Sit | 4 | 60 s | 90 s | |", days_per_week=1, session_minutes=30
    )
    assert plan.count("Day 1 shortened") == 1


# --- S7 (bad message, safe direction): the first warm-up item is never a negated cue ----------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: warm_cool_text puts the label rest ('(5 min)' or ':') before the first item, so "
    "NEGATED_ITEM_PATTERN never matches it and 'No jumping, march in place' is replaced",
)
@pytest.mark.parametrize(
    "warm", ["**Warm-up (5 min):** No jumping, march in place.", "**Warm-up:** No jumping, march in place."]
)
def test_a_warm_up_that_starts_with_a_negated_cue_is_kept(warm: str) -> None:
    assert warm_up_line(check_plan(one_day(warm), 1, "bad knees")) == warm


# --- S8 (cosmetic): separator repair under a header without a leading pipe --------------------


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _fix_table_separators uses the header's cell count only when it starts with '|', so "
    "'Exercise | Sets | ... |' over '|---+---+---+---+---|' still becomes '|---|'",
)
def test_a_plus_joined_separator_under_a_header_without_a_leading_pipe() -> None:
    table = "Exercise | Sets | Reps | Rest | Notes |\n|---+---+---+---+---|"
    assert _fix_table_separators(table).splitlines()[1] == "|---|---|---|---|---|"
