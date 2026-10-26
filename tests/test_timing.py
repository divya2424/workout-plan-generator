from __future__ import annotations

import re

from backend.core.timing import _exercise_row, fit_to_session

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
