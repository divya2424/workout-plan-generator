"""Shared fixtures and fakes for the backend tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import groq
import httpx

from backend.core.models import PlanRequest

NONE = ["No equipment"]
DUMBBELLS = ["Home dumbbells"]
GYM = ["Full gym"]

VALID = {
    "goal": "Build muscle",
    "experience": "Beginner",
    "days_per_week": 2,
    "equipment": ["Home dumbbells"],
    "session_minutes": 45,
    "limitations": "",
}


def make_request(**overrides: object) -> PlanRequest:
    """A valid PlanRequest, with any field overridden."""
    return PlanRequest(**{**VALID, **overrides})  # type: ignore[arg-type]


GOOD_PLAN = """## Plan Overview
A simple full-body split.

## Day 1 - Full Body A
| Exercise | Sets | Reps / Time | Rest | Notes |
|---|---|---|---|---|
| Goblet Squat | 3 | 10 | 60s | Chest up |
| **Dumbbell Row** | 3 | 10 | 60s | Flat back |

## Day 2 - Full Body B
| Exercise | Sets | Reps / Time | Rest | Notes |
|---|---|---|---|---|
| Push-up | 3 | 8 | 60s | Knees if needed |
| Goblet Squat | 3 | 10 | 60s | |

## Progression
- Add reps.
"""


def fake_response(content: str | None, finish_reason: str = "stop") -> SimpleNamespace:
    choice = SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)
    return SimpleNamespace(choices=[choice], usage=None)


def mock_client(*responses: object) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.side_effect = list(responses)
    return client


def status_error(cls: type[groq.APIStatusError], code: int) -> groq.APIStatusError:
    request = httpx.Request("POST", "https://api.groq.com")
    return cls("boom", response=httpx.Response(code, request=request), body=None)


RICH_PLAN = """## Plan Overview
An upper/lower split.

## Day 1 - Upper
| Exercise | Sets | Reps / Time | Rest | Notes |
|---|---|---|---|---|
| Bench Press | 3 | 8 | 90s | Elbows at 45 |
| Seated Dumbbell Shoulder Press | 3 | 10 | 60s | Brace |
| Cable Row | 3 | 10 | 60s | Squeeze |
| Triceps Pushdown | 3 | 12 | 45s | Elbows pinned |

## Day 2 - Lower
| Exercise | Sets | Reps / Time | Rest | Notes |
|---|---|---|---|---|
| Back Squat | 3 | 8 | 90s | Depth to parallel |
| Romanian Deadlift | 3 | 10 | 90s | Hinge |
| Leg Curl | 3 | 12 | 60s | Slow |
| Calf Raise | 3 | 15 | 45s | Pause at top |

## Progression
- Add load weekly.
"""
