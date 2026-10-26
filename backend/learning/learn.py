"""Self-improvement loop: test the trainer, learn rules from its mistakes, keep only rules that help.

    uv run python -m backend.learning.learn [--samples N] [--dry-run]

1. Draft a plan for every test profile with the live model and count its mistakes (evaluate.py).
2. Show the mistakes to the model and ask for at most MAX_NEW_RULES_PER_RUN new rules.
3. Draft the same profiles again with the new rules. Only if there are fewer mistakes are the
   rules written to the learned-rules section of agents/workout-trainer.md.

It runs offline, not inside the app: the hosted app's disk is reset on every deploy, and
learning from users' free text would let anyone write rules into the prompt.
"""

import argparse
import re
import time
from collections.abc import Sequence
from datetime import date

import groq
from dotenv import load_dotenv

from backend.core.constants import (
    MAX_LEARNED_RULES,
    MAX_NEW_RULES_PER_RUN,
    MAX_RULE_CHARS,
    READ_TIMEOUT_SECONDS,
    REFLECTION_MAX_TOKENS,
)
from backend.core.errors import PlanGenerationError
from backend.core.models import PlanRequest
from backend.core.utils import resolve_credentials
from backend.learning.evaluate import PROFILES, describe, find_issues
from backend.llm.agent_file import add_learned_rules, learned_rules, render
from backend.llm.client import Messages, call_groq
from backend.llm.planner import plan_max_tokens
from backend.llm.prompts import build_user_prompt, system_prompt

MAX_RATE_LIMIT_WAIT_SECONDS = 70.0  # the per-minute allowance; a longer wait is the daily cap
RATE_LIMIT_ATTEMPTS = 4
# Mistakes per draft, in PROFILES x samples order; None where the draft failed.
Results = list[list[str] | None]


class LearningStopped(Exception):
    """The round can't finish fairly (e.g. the daily token cap); nothing is written."""


_BULLET = re.compile(r"^\s*[-*]\s+(.+)$")


def _retry_after(error: groq.RateLimitError) -> float:
    """Seconds Groq asks us to wait; the default wait if the header is missing or not a number."""
    try:
        return float(error.response.headers.get("retry-after") or MAX_RATE_LIMIT_WAIT_SECONDS)
    except ValueError:
        return MAX_RATE_LIMIT_WAIT_SECONDS


def _call(messages: Messages, api_key: str, model: str, max_tokens: int) -> str:
    """One model call that waits out short rate limits instead of failing the whole round.

    A long wait (the daily token cap) stops the round, since the drafts still to come would fail.
    """
    for attempt in range(1, RATE_LIMIT_ATTEMPTS + 1):
        try:
            return call_groq(messages, api_key, model, max_tokens, READ_TIMEOUT_SECONDS)
        except PlanGenerationError as exc:
            cause = exc.__cause__
            if not isinstance(cause, groq.RateLimitError) or attempt == RATE_LIMIT_ATTEMPTS:
                raise
            wait = _retry_after(cause)
            if wait > MAX_RATE_LIMIT_WAIT_SECONDS:
                minutes = round(wait / 60)
                raise LearningStopped(
                    f"Groq's token limit is used up for about {minutes} min. Run the loop again later."
                ) from exc
            print(f"    rate limited, waiting {wait:.0f} s")
            time.sleep(wait + 1)
    raise AssertionError("unreachable: the loop returns or raises")


def _draft(request: PlanRequest, system: str, api_key: str, model: str) -> str | None:
    """One raw plan from the model, or None if it failed."""
    messages: Messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": build_user_prompt(request)},
    ]
    try:
        return _call(messages, api_key, model, plan_max_tokens(request.days_per_week))
    except PlanGenerationError as exc:
        print(f"    failed: {exc}")
        return None


def run_suite(system: str, samples: int, api_key: str, model: str) -> Results:
    """The mistakes in `samples` drafts of each test profile."""
    results: Results = []
    for request in PROFILES:
        for _ in range(samples):
            draft = _draft(request, system, api_key, model)
            found = None if draft is None else find_issues(draft, request)
            shown = "failed, not counted" if found is None else "; ".join(found) or "no mistakes"
            print(f"  {describe(request)}: {shown}")
            results.append(found)
    return results


def count_mistakes(baseline: Results, trial: Results) -> tuple[int, int]:
    """Mistakes before and after, counting only drafts that succeeded in both runs (a fair comparison)."""
    pairs = [
        (before, after)
        for before, after in zip(baseline, trial, strict=True)
        if before is not None and after is not None
    ]
    return sum(len(before) for before, _ in pairs), sum(len(after) for _, after in pairs)


def clean_rules(raw: str, existing: Sequence[str]) -> list[str]:
    """The usable new rules in the model's reply: one-line bullets, not too long, not already known."""
    known = {rule.casefold() for rule in existing}
    rules: list[str] = []
    for line in raw.splitlines():
        if not (match := _BULLET.match(line)):
            continue
        rule = " ".join(match.group(1).split())
        if (
            len(rule) <= MAX_RULE_CHARS
            and "<!--" not in rule
            and "-->" not in rule
            and rule.casefold() not in known
        ):
            known.add(rule.casefold())
            rules.append(rule)
    return rules[:MAX_NEW_RULES_PER_RUN]


def propose_rules(issues: list[str], existing: list[str], api_key: str, model: str) -> list[str]:
    """Ask the model for rules that would prevent `issues`."""
    messages: Messages = [
        {"role": "system", "content": render("reflection-system", max_new=MAX_NEW_RULES_PER_RUN)},
        {
            "role": "user",
            "content": render(
                "reflection-request",
                trainer_prompt=system_prompt(existing),
                issues="\n".join(f"- {issue}" for issue in issues),
                learned_rules="\n".join(f"- {rule}" for rule in existing) or "None yet.",
            ),
        },
    ]
    return clean_rules(_call(messages, api_key, model, REFLECTION_MAX_TOKENS), existing)


def learn(samples: int = 1, *, dry_run: bool = False) -> list[str]:
    """Run one learning round and return the rules that were kept (written unless `dry_run`)."""
    api_key, model = resolve_credentials(None, None)
    existing = learned_rules()
    room = MAX_LEARNED_RULES - len(existing)
    if room <= 0:
        print(f"Already {len(existing)} learned rules: merge or delete some in the agent file first.")
        return []

    print(f"Baseline with {len(existing)} learned rules:")
    baseline = run_suite(system_prompt(), samples, api_key, model)
    runs = [request for request in PROFILES for _ in range(samples)]
    mistakes = [
        f"{describe(request)}: {issue}"
        for request, found in zip(runs, baseline, strict=True)
        for issue in found or []
    ]
    if not mistakes:
        print("No mistakes found: nothing to learn.")
        return []

    new = propose_rules(mistakes, existing, api_key, model)[:room]
    if not new:
        print("The model proposed no usable rules.")
        return []
    print("Trying:\n" + "\n".join(f"  - {rule}" for rule in new))

    trial = run_suite(system_prompt([*existing, *new]), samples, api_key, model)
    if trial.count(None) > baseline.count(None):
        print(
            f"{trial.count(None)} drafts failed with the new rules, {baseline.count(None)} before: "
            "rules discarded."
        )
        return []
    before, after = count_mistakes(baseline, trial)
    print(f"Mistakes in the drafts that succeeded both times: {before} before, {after} with the new rules.")
    if after >= before:
        print("Not an improvement: rules discarded.")
        return []
    if dry_run:
        print("Dry run: agent file not changed.")
    else:
        add_learned_rules(new, note=f"learned {date.today()}, mistakes {before} -> {after}")
        print("Rules added to the agent file. Review the diff before committing.")
    return new


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--samples", type=int, default=1, help="drafts per test profile (more = less noise)")
    parser.add_argument("--dry-run", action="store_true", help="report, but don't change the agent file")
    args = parser.parse_args(argv)
    load_dotenv()
    try:
        learn(args.samples, dry_run=args.dry_run)
    except (PlanGenerationError, LearningStopped) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
