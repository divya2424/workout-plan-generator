"""Pure helpers used by the planner: input validation, prompt text, response checks, table parsing.

None of these call Groq or Streamlit, so they are cheap to test on their own.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Collection, Iterator

from backend.core.constants import (
    BENCH_PATTERN,
    BENCH_SUBSTITUTE_PATTERN,
    BULLET_LINE_PATTERN,
    CIRCUIT_HEADER_PATTERN,
    COOL_DOWN_SAFE_LINE,
    DAY_HEADING_PATTERN,
    DEFAULT_MODEL,
    DISCLAIMER_MARKER,
    EQUIPMENT_BANNED,
    EQUIPMENT_DUMBBELLS,
    EQUIPMENT_GYM,
    EQUIPMENT_NONE,
    EQUIPMENT_OPTIONS,
    EXPERIENCE_LEVELS,
    FALLBACK_DISCLAIMER,
    GOALS,
    LEADING_DASH_PATTERN,
    LIMITATION_GUARDS,
    MAX_DAYS,
    MAX_LIMITATIONS_CHARS,
    MAX_SESSION_MINUTES,
    MIN_DAYS,
    MIN_EXERCISES_PER_DAY_AFTER_REPAIR,
    MIN_SESSION_MINUTES,
    MINUTES_PATTERN,
    NEGATED_ITEM_PATTERN,
    NOTES_CELL,
    PARENTHETICAL_NOTE_PATTERN,
    REST_ROW_PATTERN,
    SECTION_HEADING_PATTERN,
    STRETCH_PATTERN,
    SWAP_ROW_CELLS,
    TABLE_SEPARATOR_PATTERN,
    TEXT_ITEM_SPLIT,
    WARM_COOL_LINE,
    WARM_UP_SAFE_LINE,
    LimitationGuard,
)
from backend.core.errors import PlanGenerationError, RetryableError
from backend.core.models import PlanRequest

logger = logging.getLogger(__name__)


def validate_inputs(request: PlanRequest) -> list[str]:
    """Return a list of friendly problems with the request (empty if all good)."""
    problems: list[str] = []
    if request.goal not in GOALS:
        problems.append("Please choose a fitness goal.")
    if request.experience not in EXPERIENCE_LEVELS:
        problems.append("Please choose your experience level.")
    if not _is_int(request.days_per_week) or not MIN_DAYS <= request.days_per_week <= MAX_DAYS:
        problems.append(f"Pick between {MIN_DAYS} and {MAX_DAYS} training days per week.")
    if not request.equipment:
        problems.append(f"Select at least one equipment option (choose '{EQUIPMENT_NONE}' if you have none).")
    elif any(item not in EQUIPMENT_OPTIONS for item in request.equipment):
        problems.append("One of the selected equipment options isn't recognised.")
    if not _is_int(request.session_minutes) or not (
        MIN_SESSION_MINUTES <= request.session_minutes <= MAX_SESSION_MINUTES
    ):
        problems.append(
            f"Session length should be between {MIN_SESSION_MINUTES} and {MAX_SESSION_MINUTES} minutes."
        )
    if len(request.limitations) > MAX_LIMITATIONS_CHARS:
        problems.append(f"Please keep injuries/limitations under {MAX_LIMITATIONS_CHARS} characters.")
    return problems


def _is_int(value: object) -> bool:
    """A whole number, but not a bool (True would otherwise count as 1)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _equipment_tier(equipment: Collection[str]) -> str:
    """The best option selected: a gym has everything dumbbells give, and dumbbells beat none."""
    for tier in (EQUIPMENT_GYM, EQUIPMENT_DUMBBELLS):
        if tier in equipment:
            return tier
    return EQUIPMENT_NONE


def describe_equipment(equipment: Collection[str]) -> str:
    """Spell out the equipment choices so the model can't assume anything extra."""
    tier = _equipment_tier(equipment)
    if tier == EQUIPMENT_GYM:
        return (
            "Full commercial gym (barbells, dumbbells, benches, racks, cable stations, "
            "machines, pull-up bar, cardio machines)"
        )
    if tier == EQUIPMENT_DUMBBELLS:
        return (
            "A pair of adjustable dumbbells, bodyweight only otherwise "
            "(no bench, no pull-up bar, no barbell, no machines)"
        )
    return "Bodyweight only (no weights, no bench, no pull-up bar, no machines)"


def check_plan(
    plan: str, days_per_week: int, limitations: str = "", *, equipment: Collection[str] = ()
) -> str:
    """Validate the model's output and return a cleaned-up plan.

    Raises a retryable error if the response is empty or doesn't contain exactly the requested
    training days. Exercises that clearly conflict with the user's limitations or need
    equipment they didn't select (empty `equipment`: not checked) are dropped, with a note,
    when every day keeps enough exercises; otherwise that is retryable too. Warm-ups,
    cool-downs and Notes cells that break a limitation are repaired, and a day left without a
    real exercise is retryable. Appends the disclaimer if the model forgot it.
    """
    plan = plan.strip()
    if not plan:
        raise RetryableError("The AI returned an empty response. Please try again.")

    day_numbers = [int(n) for n in DAY_HEADING_PATTERN.findall(plan)]
    if day_numbers != list(range(1, days_per_week + 1)):
        raise RetryableError(
            f"The AI returned a plan with {len(day_numbers)} training day(s) instead of "
            f"{days_per_week}. Please click Generate again.",
            model_hint=f'It had {len(day_numbers)} "## Day" sections; exactly {days_per_week} are required.',
        )

    # The order matters: text is repaired only on rows that are kept, and the training-day
    # check runs after the drops, so a drop can't leave a stretch-only day behind.
    plan, dropped_hint = _drop_conflicts(plan, limitations, equipment)
    plan = _repair_session_text(plan, limitations)
    if empty := non_training_days(plan):
        days = join_and([f"Day {n}" for n in empty])
        verb = "has" if len(empty) == 1 else "have"
        hint = (
            f"{days} {verb} no strength or conditioning exercise. Every training day needs real "
            "exercises in the table; make light days easy cardio or core work, and keep stretches "
            "in the cool-down."
        )
        raise RetryableError(
            f"The AI returned a training day without real exercises ({days}). Please try again.",
            model_hint=f"{hint} {dropped_hint}".strip(),
        )

    stretches = [name for name in list_exercises(plan) if is_stretch(name)]
    if stretches and (without := _drop_exercises(plan, stretches)) is not None:
        moved = ", ".join(stretches)
        plan = (
            f"{without}\n\n**Note:** Moved {moved} out of the main exercises (stretches go in the cool-down)."
        )

    plan = _fix_table_separators(plan)
    disclaimer_forms = (f"{DISCLAIMER_MARKER}:", f"{DISCLAIMER_MARKER}**:")
    if limitations and not any(form in plan for form in disclaimer_forms):
        plan = f"{plan}\n\n{FALLBACK_DISCLAIMER}"
    return plan


def _drop_conflicts(plan: str, limitations: str, equipment: Collection[str]) -> tuple[str, str]:
    """Drop the rows that break a limitation guard or need missing equipment, with a note for each reason.

    Returns the plan and a hint naming what was dropped ("" if nothing was). If a day would be
    left too thin, raises a retryable error carrying that hint instead.
    """
    names = list_exercises(plan)
    broken = _broken_guards(names, limitations)
    banned = _conflict_names(broken)
    gear = [name for name in find_equipment_conflicts(names, equipment) if name not in banned]
    if not banned and not gear:
        return plan, ""

    hints = [_limitation_hint(broken)] if broken else []
    if gear:
        hints.append(_equipment_hint(gear, equipment))
    hint = " ".join(hints)
    repaired = _drop_exercises(plan, [*banned, *gear])
    if repaired is None:
        reason = (
            f"conflicts with your limitations ({banned[0]})"
            if banned
            else f"needs equipment you didn't select ({gear[0]})"
        )
        raise RetryableError(f"The AI included an exercise that {reason}. Please try again.", model_hint=hint)

    logger.info("repaired plan by dropping %d conflicting exercise(s)", len(banned) + len(gear))
    notes = [repaired]
    if banned:
        clash = "it conflicts" if len(banned) == 1 else "they conflict"
        notes.append(f"**Note:** Removed {', '.join(banned)} because {clash} with your limitations.")
    if gear:
        needs = "it needs" if len(gear) == 1 else "they need"
        notes.append(f"**Note:** Removed {', '.join(gear)} because {needs} equipment you didn't select.")
    return "\n\n".join(notes), hint


def _repair_session_text(plan: str, limitations: str) -> str:
    """Replace warm-up and cool-down lines that break a limitation guard, and empty such Notes cells.

    A replaced line keeps its minutes and gets a fixed safe text; one note lists every change.
    An emptied Notes cell keeps its row and gets no note. A circuit header row can't be
    repaired, so a banned movement in one is retryable.
    """
    if not limitations:
        return plan
    lines = plan.splitlines()
    changed: list[str] = []
    for i, part, match in _warm_cool_lines(lines):
        if _text_conflicts(warm_cool_text(match), limitations):
            lines[i] = _safe_warm_cool(match)
            changed.append(part)
    for i, line in enumerate(lines):
        if (header := circuit_header(line)) and (broken := _text_conflicts(header, limitations)):
            raise RetryableError(
                "The AI wrote a circuit that conflicts with your limitations. Please try again.",
                model_hint=_limitation_hint(broken),
            )
        # Every row's Notes cell, circuit headers and rest rows included.
        if _row_name(line) and (cells := split_row(line)) != (
            kept := without_banned_notes(cells, limitations)
        ):
            lines[i] = join_row(kept)
    plan = "\n".join(lines)
    if changed:
        plan += (
            f"\n\n**Note:** Changed the {join_and(changed)} to avoid movements that conflict "
            "with your limitations."
        )
    return plan


def _warm_cool_lines(lines: list[str]) -> Iterator[tuple[int, str, re.Match[str]]]:
    """(line index, e.g. "Day 1 warm-up", label match) for each warm-up and cool-down line of a day."""
    for day, block in day_blocks(lines):
        for i in block:
            if match := WARM_COOL_LINE.match(plain_text(lines[i])):
                kind = "warm-up" if match.group(1).lower() == "warm" else "cool-down"
                yield i, f"Day {day} {kind}", match


def warm_cool_text(match: re.Match[str]) -> str:
    """Everything after "Warm-up" or "Cool-down" on a label line, inside the bold or after it."""
    return f"{match.group(2)} {match.group(3)}"


def _safe_warm_cool(match: re.Match[str]) -> str:
    """The fixed safe warm-up or cool-down line, with the minutes of the line it replaces."""
    warm = match.group(1).lower() == "warm"
    label = "Warm-up" if warm else "Cool-down"
    if minutes := MINUTES_PATTERN.search(match.group(2)) or MINUTES_PATTERN.search(match.group(3)):
        label = f"{label} ({minutes.group(1)} min)"
    return f"**{label}:** {WARM_UP_SAFE_LINE if warm else COOL_DOWN_SAFE_LINE}"


def without_banned_notes(cells: list[str], limitations: str) -> list[str]:
    """`cells` of an exercise row, with the Notes cell emptied if it names a banned movement.

    The exercise is fine (it passed the guard); only a suggested variation such as "harder:
    jump squat" isn't.
    """
    if len(cells) > NOTES_CELL and _text_conflicts(cells[NOTES_CELL], limitations):
        return [*cells[:NOTES_CELL], "", *cells[NOTES_CELL + 1 :]]
    return cells


def _text_conflicts(text: str, limitations: str) -> list[tuple[LimitationGuard, list[str]]]:
    """The guards that free text breaks, checked item by item (see TEXT_ITEM_SPLIT).

    An item that rules a movement out ("No jump, touch opposite toe") is skipped.
    """
    items = TEXT_ITEM_SPLIT.split(plain_text(text))
    return _broken_guards([item for item in items if not NEGATED_ITEM_PATTERN.match(item)], limitations)


def find_session_text_conflicts(plan: str, limitations: str) -> list[str]:
    """The warm-ups and cool-downs (e.g. "Day 1 warm-up") whose text breaks a limitation guard.

    Bullet lines right under the label count too. The repair only handles the one-line form
    the prompt asks for; this is for measuring how often the model writes bullets instead.
    """
    lines = plan.splitlines()
    found: list[str] = []
    for i, part, match in _warm_cool_lines(lines):
        text = [warm_cool_text(match)]
        for line in lines[i + 1 :]:
            if not BULLET_LINE_PATTERN.match(line) or WARM_COOL_LINE.match(line):
                break
            text.append(BULLET_LINE_PATTERN.sub("", line))
        if _text_conflicts(", ".join(text), limitations):
            found.append(part)
    return found


def day_blocks(lines: list[str]) -> list[tuple[int, range]]:
    """(day number, line range) of each "## Day" section, up to the next day or other "## " heading."""
    starts = [(i, int(m.group(1))) for i, line in enumerate(lines) if (m := DAY_HEADING_PATTERN.match(line))]
    blocks = []
    for n, (start, day) in enumerate(starts):
        end = starts[n + 1][0] if n + 1 < len(starts) else len(lines)
        end = next((i for i in range(start + 1, end) if SECTION_HEADING_PATTERN.match(lines[i])), end)
        blocks.append((day, range(start, end)))
    return blocks


def non_training_days(plan: str) -> list[int]:
    """The days whose table has no real exercise: none at all, or only stretches and breathing."""
    lines = plan.splitlines()
    return [
        day
        for day, block in day_blocks(lines)
        if not any((name := table_name(lines[i])) and not is_stretch(name) for i in block)
    ]


def is_stretch(name: str) -> bool:
    """True for stretches and mobility drills, which belong in the warm-up or cool-down."""
    return bool(STRETCH_PATTERN.search(plain_text(name)))


def find_equipment_conflicts(exercise_names: list[str], equipment: Collection[str]) -> list[str]:
    """Exercise names that need equipment the client didn't select (empty `equipment`: none).

    Notes in parentheses are ignored, as for limitations. A bench counts only when nothing in
    the name ("Bench Dip (use a chair)") says what stands in for it.
    """
    if not equipment:
        return []
    banned = EQUIPMENT_BANNED[_equipment_tier(equipment)]
    if banned is None:
        return []
    hits = []
    for name in exercise_names:
        full = plain_text(name)
        movement = PARENTHETICAL_NOTE_PATTERN.sub("", full)
        needs_bench = BENCH_PATTERN.search(movement) and not BENCH_SUBSTITUTE_PATTERN.search(full)
        if banned.search(movement) or needs_bench:
            hits.append(name)
    return hits


def _equipment_hint(names: list[str], equipment: Collection[str]) -> str:
    listed = ", ".join(repr(name) for name in names)
    return (
        f"It included {listed}, which need equipment the client doesn't have. "
        f"Use only: {describe_equipment(equipment)}."
    )


def _drop_exercises(plan: str, names: list[str]) -> str | None:
    """`plan` without the table rows for `names`, or None if that leaves a day too thin."""
    drop = set(names)
    kept: list[str] = []
    per_day: dict[int, int] = {}
    day: int | None = None
    for line in plan.splitlines():
        if heading := DAY_HEADING_PATTERN.match(line):
            day = int(heading.group(1))
            per_day[day] = 0
        name = table_name(line)
        if name in drop:
            continue
        if name and day is not None:
            per_day[day] += 1
        kept.append(line)
    if any(count < MIN_EXERCISES_PER_DAY_AFTER_REPAIR for count in per_day.values()):
        return None
    return "\n".join(kept)


def _fix_table_separators(plan: str) -> str:
    """Rewrite slightly garbled table separator rows (e.g. `|---s|`) so the table still renders."""
    lines = plan.splitlines()
    for i, line in enumerate(lines):
        if TABLE_SEPARATOR_PATTERN.match(line):
            # As many cells as the header above, which a garbled `|---+---+---|` doesn't show.
            above = lines[i - 1] if i else ""
            lines[i] = "|" + "---|" * len(split_row(above if above.lstrip().startswith("|") else line))
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


# The model writes typographic dashes and spaces (e.g. a non-breaking hyphen in "Step-Up").
_PLAIN_TEXT = str.maketrans(
    {
        "\N{NON-BREAKING HYPHEN}": "-",
        "\N{HYPHEN}": "-",
        "\N{EN DASH}": "-",
        "\N{EM DASH}": "-",
        "\N{RIGHT SINGLE QUOTATION MARK}": "'",
        "\N{NARROW NO-BREAK SPACE}": " ",
        "\N{THIN SPACE}": " ",
        "\N{NO-BREAK SPACE}": " ",
    }
)


def plain_text(text: str) -> str:
    """`text` with typographic dashes and spaces replaced by plain ones."""
    return text.translate(_PLAIN_TEXT)


def join_and(items: list[str]) -> str:
    """ "A", "A and B", "A, B and C"."""
    return " and ".join([", ".join(items[:-1]), items[-1]]) if len(items) > 1 else "".join(items)


def split_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def join_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _row_name(line: str) -> str | None:
    """The cleaned-up first cell of a table row; None for other lines and header/separator rows."""
    if not line.lstrip().startswith("|"):
        return None
    name = split_row(line)[0].strip("* ")
    if not name or name.lower() == "exercise" or set(name) <= set("-: "):
        return None
    return LEADING_DASH_PATTERN.sub("", name).strip("* ") or None


def circuit_header(line: str) -> str | None:
    """The text of a circuit header row, e.g. "Circuit (repeat 4x, 60 s rest):", else None.

    The name must look like a circuit and either end in a colon or have no Reps / Time, so
    "Hill Sprint (repeat 6x) | 6 | 20 s" and "Superset A: Push-up + Row | 3 | 10" stay exercises.
    """
    name = _row_name(line)
    if not name or not CIRCUIT_HEADER_PATTERN.search(name):
        return None
    return name if name.endswith(":") or not _has_reps(line) else None


def rest_row(line: str) -> bool:
    """True for a row that is only rest, e.g. "Rest between rounds | 3 | - | 30 s"."""
    name = _row_name(line)
    return bool(name and REST_ROW_PATTERN.match(name))


def _has_reps(line: str) -> bool:
    cells = split_row(line)
    return len(cells) > 2 and bool(cells[2].strip("-\N{EN DASH}\N{EM DASH} "))


def table_name(line: str) -> str | None:
    """The exercise name if `line` is an exercise row of a Markdown table, else None.

    Lines that aren't table rows, the header and separator rows, circuit header rows and rest
    rows give None. A leading "- " (the model's way of nesting rows under a circuit) is dropped.
    """
    name = _row_name(line)
    return None if name is None or circuit_header(line) or rest_row(line) else name


def list_exercises(plan: str) -> list[str]:
    """Unique exercise names from the plan's tables, in order."""
    names = (table_name(line) for line in plan.splitlines())
    return list(dict.fromkeys(name for name in names if name))


def _broken_guards(exercise_names: list[str], limitations: str) -> list[tuple[LimitationGuard, list[str]]]:
    """Each guard the plan breaks, with the offending exercise names.

    Notes in parentheses are ignored when looking for banned movements, because the model
    writes things like "Lateral Raise (no load overhead)", but they are still read for allowed
    variations such as "Side Plank (forearm)".
    """
    broken: list[tuple[LimitationGuard, list[str]]] = []
    if not limitations:
        return broken
    for guard in LIMITATION_GUARDS:
        if not guard.trigger.search(limitations):
            continue
        hits = []
        for name in exercise_names:
            full = plain_text(name)
            movement = PARENTHETICAL_NOTE_PATTERN.sub("", full)
            if guard.banned.search(movement) and not (guard.allowed and guard.allowed.search(full)):
                hits.append(name)
        if hits:
            broken.append((guard, hits))
    return broken


def _conflict_names(broken: list[tuple[LimitationGuard, list[str]]]) -> list[str]:
    return list(dict.fromkeys(name for _, hits in broken for name in hits))


def find_limitation_conflicts(exercise_names: list[str], limitations: str) -> list[str]:
    """Exercise names that clearly clash with a common limitation (see LIMITATION_GUARDS)."""
    return _conflict_names(_broken_guards(exercise_names, limitations))


def _limitation_hint(broken: list[tuple[LimitationGuard, list[str]]]) -> str:
    """Tell the model which movements broke the rules and the whole category to avoid."""
    names = ", ".join(repr(name) for name in _conflict_names(broken))
    categories = "; ".join(guard.description for guard, _ in broken)
    return f"It included {names}, which conflicts with the client's limitations. Leave out {categories}."


def parse_swap_row(
    raw: str, old_name: str, existing: list[str], limitations: str = "", equipment: Collection[str] = ()
) -> str:
    """Extract and validate the single replacement table row from the model's reply.

    The row gets the same checks as a plan's rows: limitations, equipment (empty: not
    checked), no stretches, and a Notes cell naming a banned movement is emptied.
    """
    taken = {name.lower() for name in [*existing, old_name]}
    for line in raw.splitlines():
        name = table_name(line)
        cells = split_row(line)
        if name is None or len(cells) != SWAP_ROW_CELLS:
            continue
        if name.lower() in taken:
            raise RetryableError(
                "The AI suggested an exercise that is already in your plan. Please try again.",
                model_hint=f"'{name}' is already in the plan. Choose a different exercise.",
            )
        if broken := _broken_guards([name], limitations):
            raise RetryableError(
                f"The AI suggested '{name}', which conflicts with your limitations. Please try again.",
                model_hint=_limitation_hint(broken),
            )
        if is_stretch(name):
            raise RetryableError(
                f"The AI suggested a stretch ('{name}') as a main exercise. Please try again.",
                model_hint=(
                    f"'{name}' is a stretch or mobility drill. Choose a strength, cardio or core exercise."
                ),
            )
        if gear := find_equipment_conflicts([name], equipment):
            raise RetryableError(
                f"The AI suggested '{name}', which needs equipment you didn't select. Please try again.",
                model_hint=_equipment_hint(gear, equipment),
            )
        return join_row(without_banned_notes(cells, limitations))
    raise RetryableError("The AI didn't return a usable replacement exercise. Please try again.")
