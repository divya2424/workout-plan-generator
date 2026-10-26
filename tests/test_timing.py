from __future__ import annotations

import re

import pytest

from backend.core.timing import _exercise_row, estimate_day_minutes, fit_to_session
from backend.core.utils import list_exercises

# Day 1 of a live plan for a 45-minute session. The model labelled it "~45 min";
# 8 exercises x 4 sets x (12 reps + 90 s rest) is really about 80 minutes.
OVERLONG_DAY = """## Day 1 - Full-Body Strength
**Warm-up (5 min):** Arm circles, hip circles, leg swings.

| Exercise | Sets | Reps / Time | Rest | Notes |
|---|---|---|---|---|
| Squat (bodyweight) | 4 | 12 | 90 s | Chest up |
| Push-up (knees on floor if needed) | 4 | 12 | 90 s | Elbows at 45 |
| Glute bridge | 4 | 12 | 90 s | Squeeze at the top |
| Bird-Dog | 4 | 12 each side | 90 s | Core tight |
| Side-step | 4 | 12 each side | 90 s | Hips level |
| Reverse plank | 4 | 12 | 90 s | Hold 12 s if easy |
| Plank to Downward Dog | 4 | 12 | 90 s | Flow |
| Superman | 4 | 12 | 90 s | Hold 2 s |

**Cool-down (5 min):** Gentle stretching.
**Session time:** ~45 min (8 exercises x 4 sets x 1.6 min = 16.8 min, rounded to 45 min).

## Progression
- Add a rep each week.
"""


def stated_minutes(plan: str) -> list[int]:
    return [int(m) for m in re.findall(r"\*\*Session time:\*\* about (\d+) min", plan)]


def day_minutes(plan: str) -> float:
    rows = [row for line in plan.splitlines() if (row := _exercise_row(line))]
    return 10 + sum(sets * per_set for sets, per_set in rows) / 60  # + 5 min warm-up, 5 min cool-down


def test_reads_reps_times_sides_and_rest() -> None:
    assert _exercise_row("| Squat | 4 | 12 | 90 s | x |") == (4, 12 * 3 + 90)
    assert _exercise_row("| Bird-Dog | 3 | 10 each side | 60 sec | x |") == (3, 10 * 3 * 2 + 60)
    assert _exercise_row("| Plank | 3 | 30\N{NARROW NO-BREAK SPACE}s | 30 s | x |") == (3, 30 + 30)
    assert _exercise_row("| Bike | 1 | 1-2 min | - | x |") == (1, 60)
    assert _exercise_row("| Burpees | 3 | AMRAP | 60 s | x |") is None  # unreadable rows are left alone
    assert _exercise_row("| Exercise | Sets | Reps / Time | Rest | Notes |") is None


def test_an_overlong_day_is_trimmed_to_fit_and_labelled_honestly() -> None:
    assert day_minutes(OVERLONG_DAY) > 75  # what the model called "~45 min"
    fitted = fit_to_session(OVERLONG_DAY, 45)
    assert day_minutes(fitted) <= 45
    assert stated_minutes(fitted) == [round(day_minutes(fitted))]
    assert "16.8 min" not in fitted  # the model's wrong arithmetic is gone
    assert "**Note:** Day 1 shortened (fewer sets) to fit your 45-minute session." in fitted
    assert "## Progression" in fitted


def test_a_day_that_fits_is_unchanged_apart_from_its_real_length() -> None:
    fitted = fit_to_session(OVERLONG_DAY, 90)
    assert "Note:" not in fitted
    assert fitted.count("| 4 |") == 8  # no sets touched
    assert stated_minutes(fitted) == [round(day_minutes(OVERLONG_DAY))]


def test_a_missing_session_line_is_added() -> None:
    no_line = "\n".join(line for line in OVERLONG_DAY.splitlines() if "Session time" not in line)
    assert len(stated_minutes(fit_to_session(no_line, 90))) == 1


def test_trimming_never_leaves_fewer_than_three_exercises() -> None:
    fitted = fit_to_session(OVERLONG_DAY, 15)  # impossible to fit 8 exercises; keep a usable day
    assert len([line for line in fitted.splitlines() if _exercise_row(line)]) >= 3


def test_trimming_stays_close_to_the_session_length() -> None:
    fitted = fit_to_session(OVERLONG_DAY, 45)
    assert 40 <= day_minutes(fitted) <= 45  # one set at a time, not a blunt proportional cut


# --- estimator notations (QE bug 5, 6d) -----------------------------------------------------


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("| Run | 4 | 400 m | 90 s |", (4, 210)),  # 400 m at 0.3 s/m, not 400 reps
        ("| Walk | 2 | 400 m | 60 s |", (2, 300)),  # a walk is twice as slow
        ("| Run | 2 | 1 km | 120 s |", (2, 420)),
        ("| Easy run | 1 | 1 mi | - |", (1, 1609 * 0.3)),
        ("| Squat | 3 | 10 | 1:30 |", (3, 120)),  # m:ss rest
        ("| Plank | 2 | 1:00 | 30 s |", (2, 90)),  # m:ss work
        ("| Squat | 3 | 10 (3 s down) | 60 s |", (3, 90)),  # a tempo note is not the set's length
        ("| Row | 3 | 10 per arm | 60 s |", (3, 120)),
        ("| Plank | 3 | 30 s (each side) | 30 s |", (3, 90)),  # a side note in parentheses still doubles
        ("| Bike | 1 | 5 min | - |", (1, 300)),  # "min" is minutes, never metres
    ],
)
def test_distances_clock_times_tempos_and_per_arm_are_read(row: str, expected: tuple[int, float]) -> None:
    assert _exercise_row(row) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("line", "minutes"),
    [
        ("**Warm-up:** 10 min easy jog.", 10),
        ("**Cool-down:** 5 min walk", 5),
        ("**Warm-up (5 min):** Brisk walk.", 5),
        ("**Warmup** 8 minutes of marching", 8),
        ("**Cool down (3 min):** Stretch.", 3),
    ],
)
def test_a_warm_up_or_cool_down_counts_in_any_written_form(line: str, minutes: int) -> None:
    rows = "| Exercise | Sets | Reps / Time | Rest | Notes |\n|---|---|---|---|---|\n| Squat | 1 | 10 | - | |"
    assert estimate_day_minutes(f"## Day 1 - A\n{line}\n\n{rows}\n")[0] == pytest.approx(minutes + 0.5)


# --- honest session lines and circuits (QE bugs 6, 13, 14) ----------------------------------

TABLE = "| Exercise | Sets | Reps / Time | Rest | Notes |\n|---|---|---|---|---|"


def day(n: int, rows: list[str], warm: int = 5, cool: int = 5) -> str:
    body = "\n".join(rows)
    return (
        f"## Day {n} - A\n**Warm-up ({warm} min):** March.\n\n{TABLE}\n{body}\n\n"
        f"**Cool-down ({cool} min):** Walk.\n**Session time:** about 45 min.\n"
    )


READABLE = ["| Squat | 3 | 10 | 60 s | |", "| Bridge | 3 | 10 | 60 s | |", "| Row | 3 | 10 | 60 s | |"]


def test_a_day_with_an_unreadable_row_says_at_least() -> None:
    fitted = fit_to_session(day(1, [*READABLE, "| Burpee | 3 | AMRAP | 60 s | |"]), 45)
    assert (
        "**Session time:** at least 24 min (estimated from the sets above; some rows have no fixed time)."
        in fitted
    )
    assert "about" not in fitted


def test_shortened_and_still_over_days_get_separate_honest_notes() -> None:
    long_rows = [f"| Move {k} | 4 | 12 | 90 s | |" for k in range(6)]
    plan = day(1, long_rows) + "\n" + day(2, READABLE, warm=12) + "\n" + day(3, READABLE, warm=14)
    fitted = fit_to_session(plan, 20)
    assert "**Note:** Day 1 shortened (fewer sets) to fit your 20-minute session." in fitted
    assert (
        "**Note:** Day 2 (about 22 min) and Day 3 (about 24 min) are still longer than your 20-minute "
        "session; shorten the warm-up or skip the last set if you're short of time."
    ) in fitted
    assert fitted.count("to fit your") == 1  # only Day 1 really fits


def test_a_still_over_day_with_unreadable_rows_says_at_least() -> None:
    fitted = fit_to_session(day(1, [*READABLE, "| Burpee | 3 | Max reps | 60 s | |"], warm=10), 15)
    assert "Day 1 (at least 20 min) is still longer than your 15-minute session" in fitted


def test_rows_under_a_circuit_header_count_per_round_and_are_never_trimmed() -> None:
    rows = [
        "| Circuit (repeat 4x, 60 s rest after each round): | - | - | - | |",
        "| - Squat | 1 | 15 | - | |",
        "| - **Glute bridge** | 1 | 20 | - | |",
    ]
    plan = day(1, rows)
    assert list_exercises(plan) == ["Squat", "Glute bridge"]  # header gone, "- " stripped
    assert estimate_day_minutes(plan)[0] == pytest.approx(10 + 4 * (45 + 60 + 60) / 60)
    fitted = fit_to_session(plan, 15)
    assert "| - Squat | 1 | 15 |" in fitted and "Day 1 (about 21 min) is still longer" in fitted


def test_a_second_header_starts_its_own_rounds() -> None:
    rows = [
        "| Circuit A (3 rounds) | - | - | - | |",
        "| Squat | 1 | 10 | - | |",
        "| Superset B (repeat 2x) | - | - | - | |",
        "| Bridge | 1 | 10 | - | |",
    ]
    assert estimate_day_minutes(day(1, rows, 0, 0))[0] == pytest.approx((3 * 30 + 2 * 30) / 60)


def test_a_circuit_ends_with_its_table() -> None:
    plan = day(1, ["| Circuit (repeat 3x) | - | - | - | |", "| Squat | 1 | 10 | - | |"], 0, 0)
    plan += f"\n{TABLE}\n| Bridge | 1 | 10 | - | |\n"
    assert estimate_day_minutes(plan)[0] == pytest.approx((3 * 30 + 30) / 60)


def test_a_session_line_added_after_a_table_gets_a_blank_line_first() -> None:
    plan = f"## Day 1 - A\n{TABLE}\n" + "\n".join(READABLE) + "\n\n## Progression\n- x\n"
    lines = fit_to_session(plan, 45).splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("**Session time"))
    assert lines[i - 1] == "" and lines[i - 2].startswith("| Row |")


def test_fitting_twice_gives_the_same_text() -> None:
    long_rows = [f"| Move {k} | 4 | 12 | 90 s | |" for k in range(6)]
    plan = day(1, long_rows) + "\n" + day(2, READABLE, warm=10)
    once = fit_to_session(plan, 15)
    assert fit_to_session(once, 15) == once
