"""All constants for the workout plan generator: input options, limits and model settings."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Literal, NamedTuple

# --- Prompts ---------------------------------------------------------------------------
# The prompt text lives in the agent file, not in code (see backend/llm/agent_file.py).
AGENT_FILE = Path(__file__).resolve().parents[2] / "agents" / "workout-trainer.md"

DISCLAIMER_MARKER = "Disclaimer"
# The tags around the client's own text in the prompt; stripped from that text (see prompts.py).
CLIENT_NOTES_TAG = re.compile(r"</?\s*client_notes\s*>", re.IGNORECASE)
# Every section the code reads from the agent file; a file missing one (e.g. a misspelt marker
# that merged two sections) is rejected instead of loading.
PROMPT_SECTIONS: tuple[str, ...] = (
    "system",
    "plan-request",
    "swap-system",
    "swap-request",
    "learned-rules-heading",
    "reflection-system",
    "reflection-request",
    "learned-rules",
)

# --- Input options and limits ----------------------------------------------------------

GOALS: tuple[str, ...] = ("Build muscle", "Lose fat", "General fitness", "Improve endurance")
EXPERIENCE_LEVELS: tuple[str, ...] = ("Beginner", "Intermediate", "Advanced")
EQUIPMENT_NONE = "No equipment"
EQUIPMENT_DUMBBELLS = "Home dumbbells"
EQUIPMENT_GYM = "Full gym"
EQUIPMENT_OPTIONS: tuple[str, ...] = (EQUIPMENT_NONE, EQUIPMENT_DUMBBELLS, EQUIPMENT_GYM)

MIN_DAYS, MAX_DAYS = 1, 7
MIN_SESSION_MINUTES, MAX_SESSION_MINUTES = 15, 120
DEFAULT_DAYS = 3
DEFAULT_SESSION_MINUTES = 45
SESSION_MINUTES_STEP = 5
MAX_LIMITATIONS_CHARS = 300
MAX_SWAP_REASON_CHARS = 300
SWAP_ROW_CELLS = 5  # Exercise | Sets | Reps / Time | Rest | Notes
NOTES_CELL = 4  # index of the Notes cell in an exercise row
# A plan with a few banned exercises is repaired by dropping them, as long as every day still
# has at least this many; otherwise the model is asked again.
MIN_EXERCISES_PER_DAY_AFTER_REPAIR = 3
# Used to estimate how long a set takes when the plan gives reps rather than a time.
SECONDS_PER_REP = 3.0

DEFAULT_MODEL = "openai/gpt-oss-20b"
TEMPERATURE = 0.7  # some variety so "Regenerate" gives a different plan

# Reasoning models think before answering, and those hidden tokens count against max_tokens.
# Low effort is plenty for writing a plan and keeps each request small. Other models don't
# accept the parameter, so it is only sent to these.
REASONING_MODEL_PREFIXES: tuple[str, ...] = ("openai/gpt-oss",)
REASONING_EFFORT: Literal["low"] = "low"

# Output budget per request. Groq reserves prompt + max_tokens against the per-minute token
# allowance (8000/min on the free tier) before answering, so the budget is only as large as
# needed. Live tests: 1000 + 550/day cut off 6 of 8 reasoning-model answers; 3000 + 500/day
# made 7-day requests impossible to admit.
BASE_MAX_TOKENS = 1500
TOKENS_PER_DAY = 400
MAX_TOKENS_CAP = 4500
SWAP_MAX_TOKENS = 800  # one table row, plus room for hidden reasoning

# --- Timeouts and retries -------------------------------------------------
# Responses are not streamed, so the read timeout must cover the whole generation.
CONNECT_TIMEOUT_SECONDS = 5.0
READ_TIMEOUT_SECONDS = 40.0
MAX_ATTEMPTS = 2

# --- Learning loop (backend/learning) ---------------------------------------
MAX_LEARNED_RULES = 12  # past this, consolidate the rules by hand before learning more
MAX_NEW_RULES_PER_RUN = 2
MAX_RULE_CHARS = 300
REFLECTION_MAX_TOKENS = 1500
MIN_SESSION_FILL = 0.85  # a day shorter than this share of the session counts as a mistake

FALLBACK_DISCLAIMER = (
    f"**{DISCLAIMER_MARKER}:** This plan is general fitness guidance, not medical advice. "
    "Because you mentioned an injury or limitation, check with a doctor or physiotherapist "
    "before starting, and stop any exercise that causes pain."
)

DAY_HEADING_PATTERN = re.compile(r"^##\s*Day\s+(\d+)\b", re.IGNORECASE | re.MULTILINE)


def terms(*patterns: str) -> re.Pattern[str]:
    """One case-insensitive pattern for exercise terms, where each space may be a space, a hyphen or nothing.

    So "push ups?\\b" matches "Push-up", "Push Up", "pushups"; a term without a trailing \\b also
    matches its plural.
    """
    return re.compile("|".join(p.replace(" ", "[- ]?") for p in patterns), re.IGNORECASE)


# Code-level backstop for the most common limitations. The prompt already forbids these, but
# live tests showed the model slipping about 1 time in 4 (e.g. a shoulder press for "no overhead
# pressing"), so check_plan drops such exercises, or asks again if a day would be left too thin.
class LimitationGuard(NamedTuple):
    trigger: re.Pattern[str]  # matched against the user's limitation text
    banned: re.Pattern[str]  # exercise names that clash with it
    allowed: re.Pattern[str] | None  # variations that are fine anyway, e.g. "forearm"
    description: str  # the whole category, told to the model when it slips


LIMITATION_GUARDS: tuple[LimitationGuard, ...] = (
    LimitationGuard(
        re.compile(r"\bknee", re.IGNORECASE),
        terms("jump", "lunge", r"step ups?\b", "pistol", "plyo", "burpee", "skater", "split squat"),
        None,
        "anything that jumps or loads one bent knee (jumps, lunges, step-ups, split or pistol squats)",
    ),
    LimitationGuard(
        re.compile(r"overhead|shoulder", re.IGNORECASE),
        terms(
            "overhead",
            "shoulder press",
            "military press",
            "push press",
            "arnold",
            "handstand",
            "snatch",
            "jerk",
            "pike",
        ),
        None,
        "ANY movement with the arms raised overhead (shoulder, Arnold, military or push presses, "
        "overhead triceps extensions, overhead squats, handstands, snatches)",
    ),
    LimitationGuard(
        re.compile(r"wrist", re.IGNORECASE),
        terms(
            r"push ups?\b",
            r"press ups?\b",
            "mountain climber",
            "burpee",
            "handstand",
            "plank",
            "bear crawl",
            r"\bdips?\b",
        ),
        re.compile(r"forearm|elbow", re.IGNORECASE),
        "anything with body weight on the hands (push-ups, hand planks, mountain climbers, dips, "
        "burpees); forearm (elbow) versions are fine",
    ),
    LimitationGuard(
        re.compile(r"lower back|back pain", re.IGNORECASE),
        terms(
            "deadlift",
            r"\brdls?\b",
            "good morning",
            r"sit ups?\b",
            "russian twist",
            "superman",
            r"\bv ups?\b",
            "jack knife",
        ),
        None,
        "spinal loading or flexion under load (deadlifts, good mornings, sit-ups, Russian twists, supermans)",
    ),
)

# Stretches and mobility drills, which belong in the warm-up or cool-down, not the main table.
# Yoga poses and dynamic mobility drills count too.
STRETCH_PATTERN = terms(
    "stretch",
    "forward fold",
    "cat cow",
    "child'?s pose",
    "mobility",
    "foam roll",
    "breathing",
    "arm circle",
    "hip circle",
    "yoga",
    "pigeon",
    "cobra pose",
    "downward dog",
    "upward dog",
    "sun salutation",
    "leg swing",
    "arm swing",
    "hip opener",
    "thoracic",
    "inch worm",
)
# A note in parentheses inside an exercise name, e.g. "(no load overhead)".
PARENTHETICAL_NOTE_PATTERN = re.compile(r"\([^)]*\)")
# A separator row: every cell starts with at least three dashes, possibly followed by junk.
TABLE_SEPARATOR_PATTERN = re.compile(r"^\s*\|(?:\s*:?-{3,}[^|]*\|)+\s*$")
# Any "## " heading; a day's section ends at the next one.
SECTION_HEADING_PATTERN = re.compile(r"^##\s")

# --- Equipment check -------------------------------------------------------------------
# Exercise names that need equipment a tier doesn't have, mirroring what describe_equipment
# tells the model. The tier is the best option selected (gym > dumbbells > none). Notes in
# parentheses are ignored, as for limitations. Cardio machines are full phrases, never a bare
# "bike", which would catch "Bicycle Crunch".
# "Air Bike Crunch" is the bicycle crunch, not the cardio machine.
_NOT_AT_HOME = (
    "bar bell",
    "kettle bell",
    r"\bkbs?\b",
    "cable",
    "machine",
    r"\bband(s|ed)?\b",
    "mini band",
    r"pull ups?\b",
    r"chin ups?\b",
    r"\bbar\b",
    "leg press",
    "lat pull down",
    r"\bsmith\b",
    r"\bsled\b",
    r"\btrx\b",
    "suspension",
    "tread mill",
    r"\browers?\b",
    "row erg",
    "ski erg",
    "(exercise|stationary|assault|spin) bike",
    "air bike(?! crunch)",
    "elliptical",
    "push down",
    "face pull",
    "pec deck",
    "leg extension",
    "hack squat",
    "preacher curl",
    "hip (ab|ad)ductor",
    "battle rope",
    "sand bag",
    "land mine",
    r"\brack\b",
    r"\bplates?\b",
    r"\brings?\b",
    r"\bballs?\b",
    r"(in|de)cline\s+(dumbbell|db)\b",
)
_DUMBBELL = ("dumb b?ell", r"\bdbs?\b", "goblet", "renegade", "skull crusher")
EQUIPMENT_BANNED: dict[str, re.Pattern[str] | None] = {
    EQUIPMENT_GYM: None,
    EQUIPMENT_DUMBBELLS: terms(*_NOT_AT_HOME),
    EQUIPMENT_NONE: terms(*_NOT_AT_HOME, *_DUMBBELL),
}
# Only a full gym has a bench, but a chair, the floor, a sofa or the stairs can stand in for
# one: "Bench Dip (use a chair)" is fine at home, "Bench Step-up" is not.
BENCH_PATTERN = re.compile(r"\bbench(es)?\b", re.IGNORECASE)
BENCH_SUBSTITUTE_PATTERN = re.compile(r"chair|floor|sofa|couch|stairs?", re.IGNORECASE)

# --- Warm-up, cool-down and Notes text ---------------------------------------------------
# "**Warm-up (5 min):** text", "**Warm-up:** 10 min jog", "**Warmup** text", any case, and the
# same for the cool-down. Groups: warm/cool, the rest of the bold label, the text after it.
WARM_COOL_LINE = re.compile(
    r"^\s*(?:[-*]\s+)?\*\*\s*(warm|cool)[- ]?(?:up|down)\b([^*]*)\*\*\s*:?\s*(.*)$", re.IGNORECASE
)
# Splits free text into items, so "Forearm plank or push-ups" is checked item by item (the
# wrist guard's "forearm" would otherwise excuse the whole line). Not inside parentheses.
TEXT_ITEM_SPLIT = re.compile(r"(?:[,;+/&]|\.\s|\s(?:then|and|or|with)\s)(?![^(]*\))", re.IGNORECASE)
# An item that rules a movement out ("No jump, touch opposite toe") doesn't instruct it.
NEGATED_ITEM_PATTERN = re.compile(r"^\s*(?:no|avoid|without|don'?t|do not|never)\b", re.IGNORECASE)
# Minutes as written, e.g. "5" or "5-10" in "(5-10 min)".
MINUTES_PATTERN = re.compile(r"(\d+(?:\.\d+)?(?:\s*-\s*\d+(?:\.\d+)?)?)\s*(?:min|mins|minutes?)\b", re.I)
# Fixed lines that replace a warm-up or cool-down that breaks a limitation. Ramp-up sets of the
# first exercise are safe because that exercise already passed the guard.
WARM_UP_SAFE_LINE = "Easy marching or walking in place, then 1-2 light, slow sets of your first exercise."
COOL_DOWN_SAFE_LINE = (
    "Easy walking, then gentle, pain-free stretches for the muscles you trained (20-30 s each)."
)

# --- Circuits and the session-time estimate ----------------------------------------------
# A table row that heads a circuit rather than naming an exercise, e.g.
# "Circuit (repeat 4x, 60 s rest after each round):". The rows under it count once per round.
# Only a name like this that ends in a colon or has no Reps / Time is a header (see
# utils.circuit_header): "Hill Sprint (repeat 6x) | 6 | 20 s" and "Superset A: Push-up + Row | 3 |
# 10" are exercises.
CIRCUIT_HEADER_PATTERN = re.compile(r"^(?:circuit|superset|rounds?)\b|repeat\s*\d", re.IGNORECASE)
# A row that is only rest, e.g. "Rest between rounds | 3 | - | 30 s" ("Rest-Pause Curl" is an exercise).
REST_ROW_PATTERN = re.compile(r"^rest(?:\s|$)", re.IGNORECASE)
# A note fit_to_session writes about the current length; replaced, not kept, when it runs again.
STILL_OVER_NOTE = re.compile(
    r"\n*^\*\*Note:\*\* [^\n]* still longer than your \d+-minute session[^\n]*$", re.M
)
CIRCUIT_ROUNDS_PATTERN = re.compile(r"repeat\s*(\d+)\s*x|(\d+)\s*rounds?", re.IGNORECASE)
# Distances in the Reps cell: 0.3 s per metre is a 5 min/km run, 0.6 s a 10 min/km walk.
RUN_SECONDS_PER_METRE = 0.3
WALK_SECONDS_PER_METRE = 0.6
# Keyed by the unit without a plural "s".
METRES_PER_UNIT: dict[str, float] = {
    "m": 1,
    "metre": 1,
    "meter": 1,
    "km": 1000,
    "kilometre": 1000,
    "kilometer": 1000,
    "mi": 1609,
    "mile": 1609,
}
# A Markdown bullet line ("- item" or "* item"), and the "- " the model puts before circuit rows.
BULLET_LINE_PATTERN = re.compile(r"^\s*[-*]\s+")
LEADING_DASH_PATTERN = re.compile(r"^-\s+")
