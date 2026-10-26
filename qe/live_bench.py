"""Live end-to-end latency and quality of generate_workout_plan against real Groq.

    uv run python -m qe.live_bench [--calls 8] [--spacing 65] [--out FILE]

Alternates a short (1 day x 15 min) and a long (7 days x 60 min) profile, waits `spacing`
seconds between calls to stay under 8,000 tokens/minute, records every Groq attempt (so the
retry path shows up), keeps rate-limited calls out of the latency figures, and stops at once
on a tokens-per-day (TPD) 429. Never prints the API key.
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
from backend.core.timing import estimate_day_minutes
from backend.core.utils import find_limitation_conflicts, list_exercises, plain_text
from backend.llm import retry
from backend.llm.planner import generate_workout_plan

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
# Equipment a bodyweight or dumbbell-only client can't use (a crude, name-based check).
NOT_AT_HOME = re.compile(
    r"barbell|cable|machine|smith|bench|pull-?up|chin-?up|kettlebell|band|leg press|lat pull", re.I
)


def quality(plan: str, profile: dict) -> dict:
    names = list_exercises(plan)
    days = len(re.findall(r"^## Day \d+", plan, re.M))
    gear = [] if "Full gym" in profile["equipment"] else [n for n in names if NOT_AT_HOME.search(n)]
    return {
        "days_ok": days == profile["days_per_week"],
        "limitation_conflicts": find_limitation_conflicts(names, profile["limitations"]),
        "equipment_violations": gear,
        "stretches_in_table": [n for n in names if STRETCH_PATTERN.search(plain_text(n))],
        "day_minutes": [round(m, 1) for m in estimate_day_minutes(plan)],
        "over_time": [round(m) for m in estimate_day_minutes(plan) if round(m) > profile["session_minutes"]],
        "disclaimer_ok": (f"{DISCLAIMER_MARKER}:" in plan) == bool(profile["limitations"]),
        "notes": re.findall(r"\*\*Note:\*\*[^\n]*", plan),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=int, default=8)
    parser.add_argument("--spacing", type=float, default=65.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
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
            attempts.append({"ok": True, "s": time.perf_counter() - start})
            return text
        except Exception as exc:
            attempts.append({"ok": False, "s": time.perf_counter() - start, "error": str(exc)})
            raise

    retry.call_groq = timed_call  # type: ignore[assignment]
    results = []
    for i in range(args.calls):
        profile = PROFILES[i % len(PROFILES)]
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
        row["attempts"] = list(attempts)
        results.append(row)
        print(
            f"call {i + 1} {kind}: {row['status']} {row['seconds']:.2f}s attempts="
            f"{[(a['ok'], round(a['s'], 2)) for a in row['attempts']]}"
            + (f" quality={row['quality']}" if "quality" in row else "")
            + (f" message={row.get('message')}" if "message" in row else ""),
            flush=True,
        )
        if row.get("tpd"):
            print("Daily token cap (TPD) hit: stopping live testing.")
            break

    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
