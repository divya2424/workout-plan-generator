"""Session-length arithmetic the model can't be trusted to do.

Live tests showed the model writing the requested length on its "Session time" line whatever
the plan added up to (one "45 min" day was really about 80). So the code estimates each day
itself, trims days that run over, and writes the real figure.
"""

from __future__ import annotations

import re

from backend.core.constants import (
    CIRCUIT_ROUNDS_PATTERN,
    METRES_PER_UNIT,
    MIN_EXERCISES_PER_DAY_AFTER_REPAIR,
    PARENTHETICAL_NOTE_PATTERN,
    RUN_SECONDS_PER_METRE,
    SECONDS_PER_REP,
    STILL_OVER_NOTE,
    WALK_SECONDS_PER_METRE,
    WARM_COOL_LINE,
)
from backend.core.utils import (
    circuit_header,
    day_blocks,
    join_and,
    join_row,
    plain_text,
    rest_row,
    split_row,
    table_name,
    warm_cool_text,
)

_DURATION = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:-\s*\d+(?:\.\d+)?\s*)?(s|sec|secs|seconds?|min|mins|minutes?|h|hrs?|hours?)\b",
    re.IGNORECASE,
)
_CLOCK = re.compile(r"\b(\d{1,2}):([0-5]\d)\b")  # "1:30"
# "m" then "in" is not metres; the unit is looked up without a plural "s" (see METRES_PER_UNIT).
_DISTANCE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(kilomet(?:re|er)s?|km|miles?|mi|met(?:re|er)s?|m)\b", re.IGNORECASE
)
_NUMBER = re.compile(r"\d+")
_PER_SIDE = re.compile(r"each|per side|per leg|per arm|/side|/leg|/arm", re.IGNORECASE)
_SESSION_LINE = re.compile(r"^\s*\*\*Session time", re.IGNORECASE)
# (line index, sets, seconds per set including rest) for one exercise row.
Row = tuple[int, int, float]


def _duration_seconds(text: str) -> float | None:
    """Seconds in '30 s', '45 sec', '1 min', '1-2 min' (lower bound), '1 hour' or '1:30'; None if no time.

    The first time written wins, so "(8 min): 5 min march, then 0:30 arm swings" is 8 minutes.
    """
    text = plain_text(text)
    found = [match for match in (_CLOCK.search(text), _DURATION.search(text)) if match]
    if not found:
        return None
    match = min(found, key=lambda m: m.start())
    if match.re is _CLOCK:
        return int(match.group(1)) * 60 + int(match.group(2))
    unit = match.group(2).lower()
    return float(match.group(1)) * (3600 if unit.startswith("h") else 60 if unit.startswith("m") else 1)


def _distance_seconds(text: str, name: str) -> float | None:
    """Seconds to cover a distance such as '400 m' or '1 km', at a walking pace for a walk."""
    match = _DISTANCE.search(plain_text(text))
    if not match:
        return None
    pace = WALK_SECONDS_PER_METRE if "walk" in name.lower() else RUN_SECONDS_PER_METRE
    unit = match.group(2).lower()
    return float(match.group(1)) * METRES_PER_UNIT[unit.removesuffix("s")] * pace


def _exercise_row(line: str) -> tuple[int, float] | None:
    """(sets, seconds per set including rest) for an exercise row, or None if it can't be read.

    The Reps cell is read as a time, then a distance, then reps; notes in parentheses such as a
    tempo "(3 s down)" are ignored, but "(each side)" still doubles the set.
    """
    if (name := table_name(line)) is None:
        return None
    cells = split_row(line)
    if len(cells) < 4 or not (sets := _NUMBER.search(cells[1])):
        return None
    amount = PARENTHETICAL_NOTE_PATTERN.sub("", cells[2])
    work = _duration_seconds(amount)
    if work is None:
        work = _distance_seconds(amount, name)
    if work is None:
        if not (reps := _NUMBER.search(amount)):
            return None
        work = int(reps.group()) * SECONDS_PER_REP
    if _PER_SIDE.search(cells[2]):
        work *= 2
    return int(sets.group()), work + (_duration_seconds(cells[3]) or 0)


def _with_sets(line: str, sets: int) -> str:
    cells = split_row(line)
    cells[1] = str(sets)
    return join_row(cells)


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


def _rounds(header: str) -> int:
    """The number of rounds a circuit header asks for ("repeat 4x", "3 rounds"); 1 if it doesn't say."""
    match = CIRCUIT_ROUNDS_PATTERN.search(header)
    return max(1, int(match.group(1) or match.group(2))) if match else 1


def _read_day(lines: list[str], block: range) -> tuple[float, list[Row], int]:
    """One day's fixed seconds, trimmable exercise rows, and number of exercise rows that can't be read.

    Fixed seconds are never trimmed: the warm-up and cool-down, rest rows ("Rest between rounds
    | 3 | - | 30 s"), and circuits, whose rows count once per round plus the rest between rounds
    (the first time on the header row).
    """
    fixed, rows, unreadable, rounds = 0.0, [], 0, 0
    for i in block:
        line = lines[i]
        if not line.lstrip().startswith("|"):
            rounds = 0  # a circuit runs to the end of its table
            if match := WARM_COOL_LINE.match(plain_text(line)):
                fixed += _duration_seconds(warm_cool_text(match)) or 0
        elif circuit_header(line):
            rounds = _rounds(line)
            fixed += rounds * (_duration_seconds(line) or 0)
        elif rest_row(line):
            cells = [*split_row(line), "", "", ""]
            sets = int(n.group()) if (n := _NUMBER.search(cells[1])) else 1
            fixed += sets * ((_duration_seconds(cells[2]) or 0) + (_duration_seconds(cells[3]) or 0))
        elif table_name(line) is not None:
            if (row := _exercise_row(line)) is None:
                unreadable += 1
            elif rounds:
                fixed += rounds * row[0] * row[1]
            else:
                rows.append((i, *row))
    return fixed, rows, unreadable


def estimate_day_minutes(plan: str) -> list[float]:
    """Estimated length of each day in minutes, in plan order (unreadable rows count as 0)."""
    lines = plan.splitlines()
    days = [_read_day(lines, block) for _, block in day_blocks(lines)]
    return [(fixed + sum(sets * per_set for _, sets, per_set in rows)) / 60 for fixed, rows, _ in days]


def _fit_sessions(plan: str, session_minutes: int) -> tuple[str, list[str]]:
    """Trim days that run over `session_minutes` and write each day's estimated length.

    See `_trim` for how a long day is shortened; unreadable rows and circuits are left alone,
    and a day with unreadable rows says "at least". Returns the updated plan and the notes to
    add: the days shortened to fit and how, every exercise removed, and the days still over.
    """
    lines = plan.splitlines()
    dropped: set[int] = set()
    inserts: dict[int, list[str]] = {}
    shortened: list[int] = []
    still_over: list[str] = []
    fewer_sets, cut, notes = False, [], []

    for day, block in day_blocks(lines):
        fixed, rows, unreadable = _read_day(lines, block)
        budget = session_minutes * 60 - fixed
        work = sum(sets * per_set for _, sets, per_set in rows)

        reduced, removed_names = False, []
        if bool(rows) and work > budget:
            before = {i: sets for i, sets, _ in rows}
            rows, removed = _trim(rows, budget)
            dropped.update(removed)
            removed_names = [name for i in sorted(removed) if (name := table_name(lines[i]))]
            reduced = any(sets < before[i] for i, sets, _ in rows)
            for i, sets, _ in rows:
                lines[i] = _with_sets(lines[i], sets)
            work = sum(sets * per_set for _, sets, per_set in rows)

        minutes = round((fixed + work) / 60)
        amount = f"at least {minutes} min" if unreadable else f"about {minutes} min"
        if minutes > session_minutes:
            still_over.append(f"Day {day} ({amount})")
            if removed_names:  # its own note: the still-over note is rewritten on every run
                notes.append(f"**Note:** Removed {join_and(removed_names)} to shorten Day {day}.")
        elif reduced or removed_names:
            shortened.append(day)
            fewer_sets = fewer_sets or reduced
            cut.extend(removed_names)

        basis = (
            "estimated from the sets above; some rows have no fixed time"
            if unreadable
            else "estimated from the sets above"
        )
        estimate = f"**Session time:** {amount} ({basis})."
        existing = next((i for i in block if _SESSION_LINE.match(lines[i])), None)
        if existing is not None:
            lines[existing] = estimate
        elif any(table_name(lines[i]) for i in block):
            last = max(i for i in block if lines[i].strip())
            # A line straight after a table row would render as part of the table.
            inserts[last] = ["", estimate] if lines[last].lstrip().startswith("|") else [estimate]

    out: list[str] = []
    for i, line in enumerate(lines):
        if i not in dropped:
            out.append(line)
        out.extend(inserts.get(i, []))
    if shortened:
        how = "; ".join(
            [*(["fewer sets"] if fewer_sets else []), *([f"removed {join_and(cut)}"] if cut else [])]
        )
        days = join_and([f"Day {n}" for n in shortened])
        notes.insert(0, f"**Note:** {days} shortened ({how}) to fit your {session_minutes}-minute session.")
    if still_over:
        verb = "is" if len(still_over) == 1 else "are"
        notes.append(
            f"**Note:** {join_and(still_over)} {verb} still longer than your {session_minutes}-minute "
            "session; shorten the warm-up or skip the last set if you're short of time."
        )
    return "\n".join(out), notes


def fit_to_session(plan: str, session_minutes: int) -> str:
    """Make every day fit `session_minutes` where it can, and say which days were shortened or still run over.

    A "shortened" note already in the plan (e.g. from an earlier run, before a swap) is not added
    again; an old "still longer" note is replaced, because the day's length may have changed.
    """
    plan, notes = _fit_sessions(STILL_OVER_NOTE.sub("", plan), session_minutes)
    for note in notes:
        if note not in plan:
            plan = f"{plan}\n\n{note}"
    return plan
