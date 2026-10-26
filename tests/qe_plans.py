"""Plan builders for the QE tests: well-formed model answers with any number of days."""

import re

# Neutral exercises: none trips a limitation guard or the stretch pattern.
SAFE_ROWS = (
    "| Glute Bridge | 3 | 10 | 60 s | Squeeze at the top |",
    "| Wall Sit | 3 | 30 s | 60 s | Thighs parallel |",
    "| Calf Raise | 3 | 12 | 45 s | Pause at the top |",
    "| Dead Bug | 3 | 10 | 45 s | Low back down |",
)
HEADER = "| Exercise | Sets | Reps / Time | Rest | Notes |\n|---|---|---|---|---|"


def day_block(n: int, rows: tuple[str, ...] | list[str] = SAFE_ROWS, warm: str = "5", cool: str = "5") -> str:
    body = "\n".join(rows)
    return (
        f"## Day {n} - Full Body\n**Warm-up ({warm} min):** Brisk walk in place.\n\n{HEADER}\n{body}\n\n"
        f"**Cool-down ({cool} min):** Easy walk.\n**Session time:** about 45 min.\n"
    )


def build_plan(days: int, rows: tuple[str, ...] | list[str] = SAFE_ROWS, tail: str = "", **kw: str) -> str:
    blocks = "\n".join(day_block(n, rows, **kw) for n in range(1, days + 1))
    return (
        "## Plan Overview\nA simple full-body plan.\n\n**Weekly schedule:** Mon - Day 1.\n\n"
        f"{blocks}\n## Progression\n- Add a rep each week.\n\n## Safety Notes\n- Stop if it hurts.\n{tail}"
    )


SESSION_LINE = re.compile(r"\*\*Session time:\*\* about (\d+) min")


def stated_minutes(plan: str) -> list[int]:
    return [int(m) for m in SESSION_LINE.findall(plan)]


def day_count(plan: str) -> int:
    return len(re.findall(r"^## Day \d+", plan, re.MULTILINE))
