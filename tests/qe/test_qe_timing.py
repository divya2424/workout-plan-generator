"""QE pass on backend/core/timing.py: session-length estimate and trimming edge cases."""

import pytest

from backend.core.timing import _exercise_row, estimate_day_minutes, fit_to_session
from backend.core.utils import list_exercises
from tests.qe_plans import HEADER, SAFE_ROWS, build_plan, day_block, stated_minutes


def one_day(rows: list[str] | tuple[str, ...], warm: str = "5", cool: str = "5") -> str:
    return day_block(1, rows, warm, cool) + "\n## Progression\n- More.\n"


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("| A | 3 | 10 | 60 s | x |", (3, 90)),
        ("| A | 3 | 8-12 | 60 s | x |", (3, 84)),  # rep range: lower bound
        ("| A | 3 | 30-45 s | 15 s | x |", (3, 45)),  # time range: lower bound
        ("| A | 3 | 30\N{EN DASH}45 sec | 15 sec | x |", (3, 45)),
        ("| A | 2 | 1.5 min | 30 seconds | x |", (2, 120)),
        ("| A | 3 | 10 each side | 60 s | x |", (3, 120)),
        ("| A | 3 | 12/leg | 60 s | x |", (3, 132)),
        ("| A | 3-4 | 10 | 60 s | x |", (3, 90)),
        ("| A | 3 | 10 | - | x |", (3, 30)),
        ("| **A** | 3 | 10 | 60 s |", (3, 90)),  # 4 cells is enough
        ("| A | 1 | 20 min | - | Steady |", (1, 1200)),
    ],
)
def test_rows_the_estimator_reads(row: str, expected: tuple[int, float]) -> None:
    assert _exercise_row(row) == expected


@pytest.mark.parametrize(
    "row",
    [
        "| A | 3 | AMRAP | 60 s | x |",
        "| A | 3 | Max reps | 60 s | x |",
        "| A | - | 10 | 60 s | x |",
        "| A | 3 | 10 |",
        "| Exercise | Sets | Reps / Time | Rest | Notes |",
        "|---|---|---|---|---|",
        "Not a table row",
        "",
    ],
)
def test_rows_the_estimator_leaves_alone(row: str) -> None:
    assert _exercise_row(row) is None


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("| Run | 4 | 400 m | 90 s | x |", (4, 90 + 120)),  # a distance is read as 400 reps (20 min)
        ("| Squat | 3 | 10 | 1:30 | x |", (3, 30 + 90)),  # "1:30" rest is read as no rest
        ("| Squat | 3 | 10 (3 s down) | 60 s | x |", (3, 30 + 60)),  # a tempo is read as the whole set
        ("| Row | 3 | 10 per arm | 60 s | x |", (3, 60 + 60)),  # "per arm" is not counted twice
    ],
)
def test_rows_the_estimator_misreads(row: str, expected: tuple[int, float]) -> None:
    assert _exercise_row(row) == expected


def test_a_day_that_fits_keeps_its_sets_and_gets_an_honest_line() -> None:
    plan = one_day(SAFE_ROWS)
    fitted = fit_to_session(plan, 60)
    assert "Note:" not in fitted and stated_minutes(fitted) == [round(estimate_day_minutes(plan)[0])]


# 15 minutes is left out: with a 5 + 5 minute warm-up and cool-down, three exercises at 90 s rest
# can't fit, see test_note_never_claims_a_fit_that_did_not_happen.
@pytest.mark.parametrize(("days", "minutes"), [(1, 20), (3, 45), (7, 60), (1, 120)])
def test_every_day_fits_every_session_length(days: int, minutes: int) -> None:
    rows = [f"| Move {k} | 4 | 12 | 90 s | x |" for k in range(10)]
    fitted = fit_to_session(build_plan(days, rows), minutes)
    assert stated_minutes(fitted) == [round(m) for m in estimate_day_minutes(fitted)]
    assert all(m <= minutes for m in estimate_day_minutes(fitted))
    assert fitted.count("| Move 0 |") == days  # the first exercises survive


def test_trimming_is_idempotent() -> None:
    once = fit_to_session(build_plan(3, [f"| Move {k} | 5 | 15 | 90 s | x |" for k in range(8)]), 30)
    twice = fit_to_session(once, 30)
    assert stated_minutes(twice) == stated_minutes(once)
    assert [line for line in twice.splitlines() if line.startswith("| Move")] == [
        line for line in once.splitlines() if line.startswith("| Move")
    ]


def test_a_day_with_no_table_is_left_alone_and_gets_no_invented_time() -> None:
    plan = "## Day 1 - Walk\nWalk for 30 minutes.\n\n## Progression\n- Walk more.\n"
    assert fit_to_session(plan, 45) == plan.rstrip("\n")


def test_no_day_headings_at_all_is_harmless() -> None:
    assert fit_to_session("Just text.", 45) == "Just text."
    assert fit_to_session("", 45) == ""


def test_unreadable_rows_are_never_trimmed() -> None:
    rows = ["| Burpees | 10 | AMRAP | 120 s | x |"] * 3
    fitted = fit_to_session(one_day(rows), 15)
    assert fitted.count("| Burpees | 10 | AMRAP |") == 3


def test_unreadable_rows_do_not_produce_a_false_session_time() -> None:
    rows = ["| Burpees | 3 | AMRAP | 60 s | x |", "| Push-up | 3 | Max reps | 60 s | x |"]
    fitted = fit_to_session(one_day(rows), 45)
    assert "about 10 min" not in fitted  # 6 sets with a minute of rest each is not a 10-minute day


def test_a_distance_is_not_read_as_hundreds_of_reps() -> None:
    rows = ["| Run | 4 | 400 m | 90 s | Easy pace |", *SAFE_ROWS[:3]]
    fitted = fit_to_session(one_day(rows), 45)
    assert "| Run | 4 |" in fitted and "shortened" not in fitted


@pytest.mark.parametrize(
    ("rows", "warm"),
    [
        (SAFE_ROWS[:3], "10"),  # a long warm-up is never trimmed
        ([f"| Move {k} | 4 | 12 | 90 s | x |" for k in range(10)], "5"),  # 3 x 1 set still 16 min
    ],
)
def test_note_never_claims_a_fit_that_did_not_happen(rows: list[str], warm: str) -> None:
    fitted = fit_to_session(one_day(rows, warm=warm), 15)
    assert all(m <= 15 for m in stated_minutes(fitted)) or "to fit your 15-minute session" not in fitted


def test_warm_up_written_differently_still_counts() -> None:
    plan = f"## Day 1 - A\n**Warm-up:** 10 min easy jog.\n\n{HEADER}\n" + "\n".join(SAFE_ROWS) + "\n"
    rows_only = estimate_day_minutes(f"## Day 1 - A\n{HEADER}\n" + "\n".join(SAFE_ROWS))[0]
    assert estimate_day_minutes(plan)[0] >= rows_only + 10


def test_an_inserted_session_line_is_separated_from_the_table() -> None:
    plan = f"## Day 1 - A\n{HEADER}\n" + "\n".join(SAFE_ROWS) + "\n\n## Progression\n- x\n"
    lines = fit_to_session(plan, 45).splitlines()
    i = next(k for k, line in enumerate(lines) if line.startswith("**Session time"))
    assert not lines[i - 1].startswith("|")  # GFM keeps adding lines to a table until a blank line


def test_session_line_inside_a_later_section_is_not_touched() -> None:
    plan = one_day(SAFE_ROWS) + "\n## Safety Notes\n**Session time** matters.\n"
    assert "**Session time** matters." in fit_to_session(plan, 45)


def test_trimming_keeps_at_least_three_exercises_and_the_first_ones() -> None:
    rows = [f"| Move {k} | 5 | 20 | 120 s | x |" for k in range(8)]
    fitted = fit_to_session(one_day(rows), 15)
    assert list_exercises(fitted)[:3] == ["Move 0", "Move 1", "Move 2"]
    assert len(list_exercises(fitted)) >= 3


# Minimised from live call 4 (7 days x 60 min): the model wrote a circuit as a header row.
CIRCUIT_DAY = """## Day 1 - Conditioning
**Warm-up (5 min):** March in place.

| Exercise | Sets | Reps / Time | Rest | Notes |
|---|---|---|---|---|
| Circuit (repeat 4x, 60 s rest after each round): | - | - | - | Back to back |
| - Bodyweight squat | 1 | 15 | - | |
| - Glute bridge | 1 | 20 | - | |
| - Wall sit | 1 | 45 s | - | |

**Cool-down (5 min):** Easy walk.
**Session time:** about 30 min.
"""


def test_a_circuit_header_row_is_not_an_exercise() -> None:
    assert not any(name.startswith("Circuit") for name in list_exercises(CIRCUIT_DAY))


def test_circuit_rounds_count_towards_the_session_time() -> None:
    # 4 rounds x (45 + 60 + 45 s work + 60 s rest) + 10 min = 24 min; the estimate says 13.
    assert estimate_day_minutes(CIRCUIT_DAY)[0] >= 20


# --- regression tests (from round 2 / round 3) --------------------------------


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ("| Run | 4 | 400 metres | 90 s | |", (4, 210)),
        ("| Run | 4 | 400 meters | 90 s | |", (4, 210)),
        ("| Run | 1 | 1 kilometre | 0 s | |", (1, 300)),
        ("| Easy Run | 1 | 1 hour | 0 s | |", (1, 3600)),
    ],
)
def test_spelled_out_units_are_read(row: str, expected: tuple[int, float]) -> None:
    assert _exercise_row(row) == expected


def test_a_spelled_out_distance_is_not_trimmed() -> None:
    rows = ["| Run | 4 | 400 metres | 90 s | |", *SAFE_ROWS[:3]]
    out = fit_to_session(build_plan(1, rows), 45)
    assert "| Run | 4 |" in out and "shortened" not in out


def test_a_clock_time_in_the_warm_up_text_does_not_replace_its_minutes() -> None:
    body = "\n".join(SAFE_ROWS)
    base = (
        f"## Day 1 - A\n**Warm-up (8 min):** 5 min march, then arm swings.\n\n"
        f"| Exercise | Sets | Reps / Time | Rest | Notes |\n|---|---|---|---|---|\n{body}\n\n"
        f"**Cool-down (5 min):** Easy walk.\n\n## Progression\n- Add a rep.\n"
    )
    with_clock = base.replace("then arm swings", "then 0:30 arm swings")
    assert estimate_day_minutes(with_clock) == estimate_day_minutes(base)


def test_exercises_removed_by_trimming_are_named_in_a_note() -> None:
    rows = [f"| Move {k} | 4 | 12 | 90 s | |" for k in range(6)]
    out = fit_to_session(build_plan(1, rows), 30)
    dropped = [f"Move {k}" for k in range(6) if f"Move {k}" not in list_exercises(out)]
    notes = " ".join(line for line in out.splitlines() if line.startswith("**Note:**"))
    assert dropped and all(name in notes for name in dropped)


def test_a_rest_between_rounds_row_is_not_an_exercise() -> None:
    rows = [
        "| Squat | 3 | 10 | 10 s | circuit |",
        "| Push-up | 3 | 10 | 10 s | circuit |",
        "| Plank | 3 | 30 s | 10 s | circuit |",
        "| Rest between rounds | 3 | — | 30 s | circuit |",
    ]
    out = fit_to_session(build_plan(1, rows), 45)
    assert "Rest between rounds" not in list_exercises(out)
    assert "at least" not in out


@pytest.mark.parametrize("reps", ["N/A", "see below"])
def test_a_circuit_header_with_words_in_the_reps_cell_is_still_a_header(reps: str) -> None:
    plan = build_plan(1, [f"| Circuit (repeat 3x, 60 s rest) | 3 | {reps} | 60 s | |", *SAFE_ROWS])
    dashed = build_plan(1, ["| Circuit (repeat 3x, 60 s rest) | | - | | |", *SAFE_ROWS])
    assert list_exercises(plan) == list_exercises(dashed)
    assert estimate_day_minutes(plan) == estimate_day_minutes(dashed)
