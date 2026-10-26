from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import groq
import httpx
import pytest

from backend.core.constants import (
    DEFAULT_MODEL,
    FALLBACK_DISCLAIMER,
    MAX_ATTEMPTS,
    READ_TIMEOUT_SECONDS,
    SWAP_MAX_TOKENS,
    TOTAL_BUDGET_SECONDS,
)
from backend.core.errors import PlanGenerationError
from backend.core.utils import (
    check_plan,
    describe_equipment,
    find_limitation_conflicts,
    list_exercises,
    resolve_credentials,
    validate_inputs,
)
from backend.llm import client as groq_client
from backend.llm import planner, retry
from backend.llm.planner import generate_workout_plan
from backend.llm.prompts import build_user_prompt
from tests.helpers import GOOD_PLAN, VALID, fake_response, make_request, mock_client, status_error

# --- input validation ---------------------------------------------------------


def test_valid_inputs_have_no_problems() -> None:
    assert validate_inputs(make_request()) == []


@pytest.mark.parametrize(
    "override",
    [
        {"goal": ""},
        {"experience": "Expert"},
        {"days_per_week": 0},
        {"days_per_week": 8},
        {"equipment": []},
        {"equipment": ["Spaceship"]},
        {"session_minutes": 5},
        {"limitations": "x" * 400},
    ],
)
def test_invalid_inputs_are_reported(override: dict) -> None:
    assert validate_inputs(make_request(**override))


def test_invalid_inputs_raise_friendly_error_without_calling_api() -> None:
    with patch.object(groq_client, "Groq") as groq_cls:
        with pytest.raises(PlanGenerationError, match="training days"):
            generate_workout_plan(**{**VALID, "days_per_week": 0}, api_key="k")
        groq_cls.assert_not_called()


def test_missing_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(PlanGenerationError, match="API key"):
        generate_workout_plan(**VALID)


# --- prompt construction ------------------------------------------------------


def test_equipment_is_made_explicit() -> None:
    assert "Bodyweight only" in describe_equipment(["No equipment"])[0]
    assert "dumbbells" in describe_equipment(["No equipment", "Home dumbbells"])[0]
    assert "Full commercial gym" in describe_equipment(["Home dumbbells", "Full gym"])[0]


def test_user_prompt_contains_every_input() -> None:
    prompt = build_user_prompt(
        make_request(
            goal="Lose fat",
            experience="Advanced",
            days_per_week=5,
            equipment=["No equipment"],
            session_minutes=30,
            limitations="bad knees",
        )
    )
    for expected in ("Lose fat", "Advanced", "exactly 5", "30 minutes", "Bodyweight only", "bad knees"):
        assert expected in prompt


# --- response validation ------------------------------------------------------


def test_check_plan_accepts_correct_day_count() -> None:
    assert check_plan(GOOD_PLAN, 2, limitations="") == GOOD_PLAN.strip()


MALFORMED_PLANS = ["", "   ", "Just do some squats.", GOOD_PLAN.replace("## Day 2", "## Day 3")]


@pytest.mark.parametrize("bad", MALFORMED_PLANS)
def test_check_plan_rejects_empty_or_malformed(bad: str) -> None:
    with pytest.raises(PlanGenerationError):
        check_plan(bad, 2, limitations="")


def test_check_plan_rejects_wrong_day_count() -> None:
    with pytest.raises(PlanGenerationError, match="2 training day"):
        check_plan(GOOD_PLAN, 3, limitations="")


def test_disclaimer_added_when_injury_given_and_model_forgot() -> None:
    result = check_plan(GOOD_PLAN, 2, limitations="bad knees")
    assert result.endswith(FALLBACK_DISCLAIMER)
    assert "Disclaimer" not in check_plan(GOOD_PLAN, 2, limitations="")


# --- API call -----------------------------------------------------------------


def test_successful_generation_sends_system_and_user_prompt() -> None:
    client = mock_client(fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        plan = generate_workout_plan(**{**VALID, "limitations": "bad knees"}, api_key="k")
    assert "## Day 2" in plan and "Disclaimer" in plan
    messages = client.chat.completions.create.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "bad knees" in messages[1]["content"]


def test_malformed_response_is_retried_once() -> None:
    client = mock_client(fake_response("Do some squats."), fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        assert "## Day 1" in generate_workout_plan(**VALID, api_key="k")
    assert client.chat.completions.create.call_count == 2


def test_repeated_empty_response_gives_friendly_error() -> None:
    client = mock_client(fake_response(None), fake_response(""))
    with patch.object(groq_client, "Groq", return_value=client):
        with pytest.raises(PlanGenerationError, match="empty"):
            generate_workout_plan(**VALID, api_key="k")


CONNECTION_ERROR = groq.APIConnectionError(request=httpx.Request("POST", "https://api.groq.com"))


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (status_error(groq.AuthenticationError, 401), "API key was rejected"),
        (status_error(groq.RateLimitError, 429), "rate limit"),
        (status_error(groq.NotFoundError, 404), "isn't available"),
        (status_error(groq.BadRequestError, 400), "status 400"),
        (RuntimeError("weird"), "Something went wrong"),
    ],
)
def test_permanent_errors_are_friendly_and_not_retried(error: Exception, message: str) -> None:
    client = mock_client(error, fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        with pytest.raises(PlanGenerationError, match=message):
            generate_workout_plan(**VALID, api_key="k")
    assert client.chat.completions.create.call_count == 1


@pytest.mark.parametrize(
    ("error", "message"),
    [
        (status_error(groq.InternalServerError, 500), "status 500"),
        (status_error(groq.InternalServerError, 503), "status 503"),
        (CONNECTION_ERROR, "internet"),
    ],
)
def test_transient_errors_are_retried_then_reported(error: Exception, message: str) -> None:
    client = mock_client(*[error] * MAX_ATTEMPTS)
    with patch.object(groq_client, "Groq", return_value=client):
        with pytest.raises(PlanGenerationError, match=message):
            generate_workout_plan(**VALID, api_key="k")
    assert client.chat.completions.create.call_count == MAX_ATTEMPTS


def test_transient_error_recovers_on_retry() -> None:
    client = mock_client(CONNECTION_ERROR, fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        assert "## Day 1" in generate_workout_plan(**VALID, api_key="k")


def test_truncated_response_is_rejected_and_retried() -> None:
    cut_off = fake_response(GOOD_PLAN, finish_reason="length")
    client = mock_client(cut_off, fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        assert "## Progression" in generate_workout_plan(**VALID, api_key="k")
    assert client.chat.completions.create.call_count == 2


def test_always_truncated_gives_friendly_error() -> None:
    client = mock_client(*[fake_response(GOOD_PLAN, finish_reason="length")] * MAX_ATTEMPTS)
    with patch.object(groq_client, "Groq", return_value=client):
        with pytest.raises(PlanGenerationError, match="cut off"):
            generate_workout_plan(**VALID, api_key="k")


def test_output_budget_scales_with_days_and_timeouts_are_set() -> None:
    plan7 = "\n".join(f"## Day {n} - x" for n in range(1, 8))
    client = mock_client(fake_response(GOOD_PLAN), fake_response(plan7))
    with patch.object(groq_client, "Groq", return_value=client):
        generate_workout_plan(**VALID, api_key="k")
        generate_workout_plan(**{**VALID, "days_per_week": 7}, api_key="k")
    two_days, seven_days = (c.kwargs for c in client.chat.completions.create.call_args_list)
    assert seven_days["max_tokens"] > two_days["max_tokens"]
    assert two_days["timeout"].read <= READ_TIMEOUT_SECONDS


def test_retry_stops_when_time_budget_is_spent(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [0.0]
    monkeypatch.setattr(retry, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=lambda _s: None))

    def slow_failure(**_kwargs: object) -> None:
        now[0] += TOTAL_BUDGET_SECONDS + 1  # the first attempt eats the whole budget
        raise CONNECTION_ERROR

    client = MagicMock()
    client.chat.completions.create.side_effect = slow_failure
    with patch.object(groq_client, "Groq", return_value=client):
        with pytest.raises(PlanGenerationError, match="internet"):
            generate_workout_plan(**VALID, api_key="k")
    assert client.chat.completions.create.call_count == 1


def test_retry_waits_the_configured_backoff_between_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    slept: list[float] = []
    monkeypatch.setattr(retry, "RETRY_BACKOFF_SECONDS", 2.0)
    monkeypatch.setattr(retry, "time", SimpleNamespace(monotonic=time.monotonic, sleep=slept.append))
    client = mock_client(CONNECTION_ERROR, fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        generate_workout_plan(**VALID, api_key="k")
    assert slept == [2.0]  # one wait, between attempt 1 and 2, and none after success


def test_client_is_created_once_and_reused() -> None:
    client = mock_client(fake_response(GOOD_PLAN), fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client) as groq_cls:
        generate_workout_plan(**VALID, api_key="k")
        generate_workout_plan(**VALID, api_key="k")
    assert groq_cls.call_count == 1
    assert groq_cls.call_args.kwargs["max_retries"] == 0


# --- swap ---------------------------------------------------------------------------

SWAP_REQUEST = make_request(limitations="sore wrist")


def test_swap_replaces_only_matching_rows_and_sends_a_small_request() -> None:
    client = mock_client(fake_response("| Reverse Lunge | 3 | 8 | 60s | Stay upright |"))
    with patch.object(groq_client, "Groq", return_value=client):
        new_plan = planner.swap_exercise(GOOD_PLAN, "Goblet Squat", "hurts", SWAP_REQUEST, api_key="k")
    assert new_plan.count("Reverse Lunge") == 2 and "Goblet Squat" not in new_plan
    # Exactly the two Goblet Squat rows changed; every other line is untouched.
    before, after = GOOD_PLAN.splitlines(), new_plan.splitlines()
    assert len(before) == len(after)
    assert [b for b, a in zip(before, after, strict=True) if b != a] == [
        line for line in before if "Goblet Squat" in line
    ]
    call = client.chat.completions.create.call_args.kwargs
    assert call["max_tokens"] == SWAP_MAX_TOKENS
    assert "## Day 1" not in call["messages"][1]["content"]  # the full plan is not re-sent
    assert "sore wrist" in call["messages"][1]["content"]


@pytest.mark.parametrize(
    "bad_reply",
    [
        "Sure! Try lunges.",
        "| Exercise | Sets | Reps / Time | Rest | Notes |\n|---|---|---|---|---|",
        "| Push-up | 3 | 8 | 60s | already in plan |",
        "| Goblet Squat | 3 | 10 | 60s | same exercise |",
        "| Lunge | 3 |",
    ],
)
def test_swap_rejects_unusable_replies(bad_reply: str) -> None:
    client = mock_client(*[fake_response(bad_reply)] * MAX_ATTEMPTS)
    with patch.object(groq_client, "Groq", return_value=client):
        with pytest.raises(PlanGenerationError):
            planner.swap_exercise(GOOD_PLAN, "Goblet Squat", "", SWAP_REQUEST, api_key="k")
    assert client.chat.completions.create.call_count == MAX_ATTEMPTS


def test_swap_matches_exercise_names_case_insensitively() -> None:
    client = mock_client(fake_response("| Reverse Lunge | 3 | 8 | 60s | Stay upright |"))
    with patch.object(groq_client, "Groq", return_value=client):
        new_plan = planner.swap_exercise(GOOD_PLAN, "  goblet SQUAT ", "", SWAP_REQUEST, api_key="k")
    assert "Goblet Squat" not in new_plan and new_plan.count("Reverse Lunge") == 2


def test_swap_unknown_exercise_does_not_call_api() -> None:
    with patch.object(groq_client, "Groq") as groq_cls:
        with pytest.raises(PlanGenerationError, match="doesn't appear"):
            planner.swap_exercise(GOOD_PLAN, "Deadlift", "", SWAP_REQUEST, api_key="k")
        groq_cls.assert_not_called()


def test_list_exercises_reads_table_rows() -> None:
    assert list_exercises(GOOD_PLAN) == ["Goblet Squat", "Dumbbell Row", "Push-up"]


# --- boundary guarantees ------------------------------------------------------------


def test_unexpected_internal_errors_become_friendly_errors() -> None:
    """A bug anywhere inside the public functions must not leak a raw exception to the UI."""
    with patch.object(planner, "build_user_prompt", side_effect=ValueError("bug")):
        with pytest.raises(PlanGenerationError, match="Something went wrong"):
            generate_workout_plan(**VALID, api_key="k")
    with patch.object(planner, "build_swap_prompt", side_effect=ValueError("bug")):
        with pytest.raises(PlanGenerationError, match="Something went wrong"):
            planner.swap_exercise(GOOD_PLAN, "Goblet Squat", "", SWAP_REQUEST, api_key="k")


def test_non_string_input_does_not_crash() -> None:
    with pytest.raises(PlanGenerationError):
        generate_workout_plan("Build muscle", "Beginner", 3, ["No equipment"], limitations=None, api_key="k")  # type: ignore[arg-type]


def test_blank_model_env_var_falls_back_to_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GROQ_MODEL", "")
    assert resolve_credentials("k", None) == ("k", DEFAULT_MODEL)
    monkeypatch.setenv("GROQ_MODEL", "custom-model")
    assert resolve_credentials("k", None) == ("k", "custom-model")
    assert resolve_credentials("k", "explicit") == ("k", "explicit")


def test_plan_request_normalises_and_compares_by_value() -> None:
    a = make_request(equipment=["Full gym"], limitations="  sore wrist  ")
    assert a.equipment == ("Full gym",) and a.limitations == "sore wrist"
    assert a == make_request(equipment=("Full gym",), limitations="sore wrist")
    assert a != make_request(equipment=["Full gym"], days_per_week=4)


def test_check_plan_repairs_garbled_table_separators() -> None:
    garbled = GOOD_PLAN.replace("|---|---|---|---|---|", "|---|---s|---|---|---|", 1)
    fixed = check_plan(garbled, 2, limitations="")
    assert "---s" not in fixed
    assert fixed.count("|---|---|---|---|---|") == 2


@pytest.mark.parametrize(
    ("model", "expected"),
    [("openai/gpt-oss-20b", "low"), ("qwen/qwen3.8-27b", groq.omit)],
)
def test_reasoning_effort_is_only_sent_to_reasoning_models(model: str, expected: object) -> None:
    client = mock_client(fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        generate_workout_plan(**VALID, api_key="k", model=model)
    assert client.chat.completions.create.call_args.kwargs["reasoning_effort"] == expected


# --- limitation guard -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("limitations", "exercise", "conflict"),
    [
        ("no overhead pressing", "Seated Dumbbell Shoulder Press (light)", True),  # seen live
        ("no overhead pressing", "Overhead Triceps Extension (rope)", True),  # seen live
        ("bad knees", "Step\N{NON-BREAKING HYPHEN}Up (low step)", True),  # seen live, non-breaking hyphen
        ("sore wrist", "Mountain Climbers", True),  # seen live
        ("sore wrist", "Side Plank (forearm)", False),  # forearm variations are fine
        ("no overhead pressing", "Overhead Squat (no weight or light)", True),  # seen live
        ("no overhead pressing", "Seated Dumbbell Lateral Raise (no load overhead)", False),  # seen live
        ("bad knees", "Dumbbell Hip Thrust (feet flat, no jumping)", False),  # note, not movement
        ("lower back pain", "Seated side crunch", False),  # low-load, a judgment call
        ("asthma", "Burpee", False),  # no rule for this limitation: left to the prompt
        ("", "Burpee", False),
    ],
)
def test_limitation_guard(limitations: str, exercise: str, conflict: bool) -> None:
    assert bool(find_limitation_conflicts([exercise], limitations)) is conflict


def test_plan_breaking_a_limitation_is_rejected_then_retried() -> None:
    bad = GOOD_PLAN.replace("Goblet Squat", "Jump Squat")
    client = mock_client(fake_response(bad), fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        plan = generate_workout_plan(**{**VALID, "limitations": "bad knees"}, api_key="k")
    assert "Jump Squat" not in plan
    assert client.chat.completions.create.call_count == 2


def test_swap_rejects_a_replacement_that_breaks_a_limitation() -> None:
    knee_request = make_request(limitations="bad knees")
    replies = [fake_response("| Walking Lunge | 3 | 10 | 60s | x |")] * MAX_ATTEMPTS
    with patch.object(groq_client, "Groq", return_value=mock_client(*replies)):
        with pytest.raises(PlanGenerationError, match="conflicts with your limitations"):
            planner.swap_exercise(GOOD_PLAN, "Push-up", "", knee_request, api_key="k")


def test_retry_tells_the_model_what_was_wrong() -> None:
    bad = GOOD_PLAN.replace("Goblet Squat", "Jump Squat")
    client = mock_client(fake_response(bad), fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        generate_workout_plan(**{**VALID, "limitations": "bad knees"}, api_key="k")
    first, second = (c.kwargs["messages"] for c in client.chat.completions.create.call_args_list)
    assert len(second) == len(first) + 1
    assert "Jump Squat" in second[-1]["content"] and "rejected" in second[-1]["content"]


def test_transport_retries_resend_the_original_request() -> None:
    client = mock_client(CONNECTION_ERROR, fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        generate_workout_plan(**VALID, api_key="k")
    first, second = (c.kwargs["messages"] for c in client.chat.completions.create.call_args_list)
    assert first == second
