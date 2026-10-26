"""Session-length arithmetic the model can't be trusted to do.

Live tests showed the model writing the requested length on its "Session time" line whatever
the plan added up to (one "45 min" day was really about 80). So the code estimates each day
itself, trims days that run over, and writes the real figure.
"""

from __future__ import annotations

import re

from backend.core.constants import (
    DAY_HEADING_PATTERN,
    MIN_EXERCISES_PER_DAY_AFTER_REPAIR,
    SECONDS_PER_REP,
)
from backend.core.utils import plain_text, split_row, table_name

_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-\s*\d+(?:\.\d+)?\s*)?(s|sec|secs|seconds?|min|mins|minutes?)\b", re.IGNORECASE
)
_NUMBER = re.compile(r"\d+")
_WARM_COOL = re.compile(r"\*\*(?:warm|cool)-?\s*(?:up|down)\s*\((\d+(?:\.\d+)?)\s*min", re.IGNORECASE)
_PER_SIDE = re.compile(r"each|per side|per leg|/side|/leg", re.IGNORECASE)
_SESSION_LINE = re.compile(r"^\s*\*\*Session time", re.IGNORECASE)
# (line index, sets, seconds per set including rest) for one exercise row.
Row = tuple[int, int, float]

_SECTION_HEADING = re.compile(r"^##\s")


def _duration_seconds(text: str) -> float | None:
    """Seconds in '30 s', '45 sec', '1 min' or '1-2 min' (lower bound); None if no time unit."""
    match = _DURATION.search(plain_text(text))
    if not match:
        return None
    value = float(match.group(1))
    return value * 60 if match.group(2).lower().startswith("m") else value


def _exercise_row(line: str) -> tuple[int, float] | None:
    """(sets, seconds per set including rest) for an exercise row, or None if it can't be read."""
    if table_name(line) is None:
        return None
    cells = split_row(line)
    if len(cells) < 4 or not (sets := _NUMBER.search(cells[1])):
        return None
    work = _duration_seconds(cells[2])
    if work is None:
        if not (reps := _NUMBER.search(cells[2])):
            return None
        work = int(reps.group()) * SECONDS_PER_REP
    if _PER_SIDE.search(cells[2]):
        work *= 2
    return int(sets.group()), work + (_duration_seconds(cells[3]) or 0)


def _with_sets(line: str, sets: int) -> str:
    cells = split_row(line)
    cells[1] = str(sets)
    return "| " + " | ".join(cells) + " |"


def _trim(rows: list[Row], budget: float) -> tuple[list[Row], list[int]]:
    """Take away one set at a time until the day fits `budget` seconds, staying as close to it as possible.

    Sets come off the exercise with the most sets (then the longest sets). Below 2 sets, the last
    exercise is dropped instead, while more than MIN_EXERCISES_PER_DAY_AFTER_REPAIR remain; only
    then do exercises go down to a single set. Returns the kept rows and the dropped line indexes.
    """
    rows, removed = list(rows), []
    while sum(sets * per_set for _, sets, per_set in rows) > budget:
        for floor in (2, 1):
            candidates = [k for k, (_, sets, _) in enumerate(rows) if sets > floor]
            if candidates:
                k = max(candidates, key=lambda k: (rows[k][1], rows[k][2]))
                rows[k] = (rows[k][0], rows[k][1] - 1, rows[k][2])
                break
            if floor == 2 and len(rows) > MIN_EXERCISES_PER_DAY_AFTER_REPAIR:
                removed.append(rows.pop()[0])
                break
        else:
            break  # every exercise is down to one set and the minimum count: as short as it gets
    return rows, removed


def _fit_sessions(plan: str, session_minutes: int) -> tuple[str, list[int]]:
    """Trim days that run over `session_minutes` and write each day's estimated length.

    See `_trim` for how a long day is shortened. Rows the estimate can't read are left alone.
    Returns the updated plan and the numbers of the days that were shortened.
    """
    lines = plan.splitlines()
    starts = [i for i, line in enumerate(lines) if DAY_HEADING_PATTERN.match(line)]
    dropped: set[int] = set()
    inserts: dict[int, str] = {}
    shortened: list[int] = []

    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        end = next((i for i in range(start + 1, end) if _SECTION_HEADING.match(lines[i])), end)
        block = range(start, end)
        fixed = sum(float(m.group(1)) * 60 for i in block if (m := _WARM_COOL.search(plain_text(lines[i]))))
        rows = [(i, *row) for i in block if (row := _exercise_row(lines[i]))]
        budget = session_minutes * 60 - fixed
        work = sum(sets * per_set for _, sets, per_set in rows)

        if rows and work > budget:
            shortened.append(int(DAY_HEADING_PATTERN.match(lines[start]).group(1)))  # type: ignore[union-attr]
            rows, removed = _trim(rows, budget)
            dropped.update(removed)
            for i, sets, _ in rows:
                lines[i] = _with_sets(lines[i], sets)
            work = sum(sets * per_set for _, sets, per_set in rows)

        estimate = (
            f"**Session time:** about {round((fixed + work) / 60)} min (estimated from the sets above)."
        )
        existing = next((i for i in block if _SESSION_LINE.match(lines[i])), None)
        if existing is not None:
            lines[existing] = estimate
        elif rows:
            inserts[max(i for i in block if lines[i].strip())] = estimate

    out: list[str] = []
    for i, line in enumerate(lines):
        if i not in dropped:
            out.append(line)
        if i in inserts:
            out.append(inserts[i])
    return "\n".join(out), shortened


def fit_to_session(plan: str, session_minutes: int) -> str:
    """Make every day fit `session_minutes` and say so when a day had to be shortened."""
    plan, shortened = _fit_sessions(plan, session_minutes)
    if not shortened:
        return plan
    days = ", ".join(f"Day {n}" for n in shortened)
    return f"{plan}\n\n**Note:** {days} shortened (fewer sets) to fit your {session_minutes}-minute session."
