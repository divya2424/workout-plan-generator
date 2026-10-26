"""Live end-to-end latency and quality of generate_workout_plan against real Groq.

    uv run python -m qe.live_bench [--calls 8] [--spacing 65] [--out FILE]

Alternates a short (1 day x 15 min) and a long (7 days x 60 min) profile, waits `spacing`
seconds between calls to stay under 8,000 tokens/minute, records every Groq attempt (so the
retry path shows up), keeps rate-limited calls out of the latency figures, and stops at once
on any 429, saying whether it was the tokens-per-day (TPD) cap. Never prints the API key.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
from pathlib import Path

import groq
from dotenv import load_dotenv

from backend.core.constants import DISCLAIMER_MARKER, STRETCH_PATTERN
from backend.core.errors import PlanGenerationError
from backend.core.models import PlanRequest
from backend.core.timing import estimate_day_minutes
from backend.core.utils import (
    find_equipment_conflicts,
    find_limitation_conflicts,
    find_session_text_conflicts,
    list_exercises,
    non_training_days,
    plain_text,
)
from backend.learning.evaluate import find_issues
from backend.llm import retry
from backend.llm.planner import generate_workout_plan, swap_exercise

PROFILES: list[dict] = [
    {
        "goal": "Lose fat",
        "experience": "Beginner",
        "days_per_week": 1,
        "equipment": ["No equipment"],
        "session_minutes": 15,
        "limitations": "bad knees",
    },
    {
        "goal": "Build muscle",
        "experience": "Intermediate",
        "days_per_week": 7,
        "equipment": ["Home dumbbells"],
        "session_minutes": 60,
        "limitations": "sore wrist",
    },
    {
        "goal": "Improve endurance",
        "experience": "Advanced",
        "days_per_week": 1,
        "equipment": ["Full gym"],
        "session_minutes": 15,
        "limitations": "",
    },
    {
        "goal": "General fitness",
        "experience": "Beginner",
        "days_per_week": 7,
        "equipment": ["No equipment"],
        "session_minutes": 60,
        "limitations": "no overhead pressing",
    },
]


def quality(plan: str, profile: dict) -> dict:
    names = list_exercises(plan)
    days = len(re.findall(r"^## Day \d+", plan, re.M))
    gear = find_equipment_conflicts(names, profile["equipment"])
    return {
        "days_ok": days == profile["days_per_week"],
        "limitation_conflicts": find_limitation_conflicts(names, profile["limitations"]),
        "equipment_violations": gear,
        "non_training_days": non_training_days(plan),
        "warm_cool_conflicts": find_session_text_conflicts(plan, profile["limitations"]),
        "stretches_in_table": [n for n in names if STRETCH_PATTERN.search(plain_text(n))],
        "day_minutes": [round(m, 1) for m in estimate_day_minutes(plan)],
        "over_time": [round(m) for m in estimate_day_minutes(plan) if round(m) > profile["session_minutes"]],
        "disclaimer_ok": (f"{DISCLAIMER_MARKER}:" in plan) == bool(profile["limitations"]),
        "notes": re.findall(r"\*\*Note:\*\*[^\n]*", plan),
    }


def swap_once(last: dict, attempts: list[dict]) -> dict:
    """Swap the first exercise of a live plan through swap_exercise and check the result."""
    profile = last["profile"]
    exercise = list_exercises(last["plan"])[0]
    attempts.clear()
    start = time.perf_counter()
    row: dict = {"call": "swap", "kind": "swap", "exercise": exercise}
    try:
        request = PlanRequest(**{**profile, "equipment": tuple(profile["equipment"])})
        swapped = swap_exercise(last["plan"], exercise, "too hard", request)
        changed = [n for n in list_exercises(swapped) if n not in list_exercises(last["plan"])]
        row.update(status="ok", new=changed, quality=quality(swapped, profile), plan=swapped)
    except PlanGenerationError as exc:
        row.update(status="error", message=str(exc))
    row["seconds"] = time.perf_counter() - start
    row["attempts"] = [{k: v for k, v in a.items() if k != "raw"} for a in attempts]
    print(
        f"swap {exercise!r}: {row['status']} {row['seconds']:.2f}s {row.get('new') or row.get('message')}"
        + (f" quality={row['quality']}" if "quality" in row else ""),
        flush=True,
    )
    return row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=65.0)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--profiles", default="", help="comma-separated PROFILES indexes, in order")
    parser.add_argument(
        "--swap", action="store_true", help="after the last call, swap one exercise in its plan"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv()
    if not os.getenv("GROQ_API_KEY"):
        raise SystemExit("GROQ_API_KEY is not set")
    logging.basicConfig(level=logging.WARNING)

    attempts: list[dict] = []
    real_call = retry.call_groq

    def timed_call(*a: object, **kw: object) -> str:
        start = time.perf_counter()
        try:
            text = real_call(*a, **kw)  # type: ignore[arg-type]
            attempts.append({"ok": True, "s": time.perf_counter() - start, "raw": text})
            return text
        except Exception as exc:
            attempts.append({"ok": False, "s": time.perf_counter() - start, "error": str(exc)})
            raise

    retry.call_groq = timed_call  # type: ignore[assignment]
    results = []
    order = [int(k) for k in args.profiles.split(",")] if args.profiles else list(range(args.calls))
    for i in range(min(args.calls, len(order))):
        profile = PROFILES[order[i] % len(PROFILES)]
        kind = "short (1d x 15m)" if profile["days_per_week"] == 1 else "long (7d x 60m)"
        if i:
            time.sleep(args.spacing)
        attempts.clear()
        start = time.perf_counter()
        row: dict = {"call": i + 1, "kind": kind, "profile": profile}
        try:
            plan = generate_workout_plan(**profile)  # type: ignore[arg-type]
            row.update(status="ok", quality=quality(plan, profile), plan=plan)
        except PlanGenerationError as exc:
            cause = exc.__cause__
            body = json.dumps(getattr(cause, "body", None) or "")
            if isinstance(cause, groq.RateLimitError):
                row.update(status="rate_limited", tpd="TPD" in body or "per day" in body, message=str(exc))
            else:
                row.update(status="error", message=str(exc))
        row["seconds"] = time.perf_counter() - start
        request = PlanRequest(**{**profile, "equipment": tuple(profile["equipment"])})
        row["raw_issues"] = [find_issues(a["raw"], request) for a in attempts if a.get("raw")]
        row["attempts"] = list(attempts)
        results.append(row)
        print(
            f"call {i + 1} {kind}: {row['status']} {row['seconds']:.2f}s attempts="
            f"{[(a['ok'], round(a['s'], 2)) for a in row['attempts']]}"
            + (f" quality={row['quality']}" if "quality" in row else "")
            + (f" message={row.get('message')}" if "message" in row else "")
            + f" raw_issues={row['raw_issues']}",
            flush=True,
        )
        if row["status"] == "rate_limited":
            print(f"429 ({'TPD' if row.get('tpd') else 'per-minute'} limit): stopping live testing.")
            break

    last = next((r for r in reversed(results) if r["status"] == "ok"), None)
    if args.swap and last and results[-1]["status"] != "rate_limited":
        time.sleep(args.spacing)
        results.append(swap_once(last, attempts))

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
