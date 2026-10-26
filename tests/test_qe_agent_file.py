"""QE pass on the agent file and the learning loop: broken files and odd learned rules."""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.core.constants import AGENT_FILE
from backend.core.errors import PlanGenerationError
from backend.learning import learn as learn_module
from backend.learning.evaluate import PROFILES
from backend.learning.learn import clean_rules, learn
from backend.llm import agent_file
from backend.llm import client as groq_client
from backend.llm.agent_file import add_learned_rules, learned_rules, load_sections
from backend.llm.planner import generate_workout_plan, swap_exercise
from tests.helpers import GOOD_PLAN, VALID, fake_response, make_request, mock_client

ORIGINAL = AGENT_FILE.read_text(encoding="utf-8")


@pytest.fixture
def agent(tmp_path: Path) -> Iterator[Path]:
    """Point every reader of the agent file at a throwaway copy."""
    copy = tmp_path / "agent.md"
    shutil.copy(AGENT_FILE, copy)
    real = load_sections

    def load(path: Path = copy) -> dict[str, str]:
        return real(copy)

    load.cache_clear = real.cache_clear  # type: ignore[attr-defined]  # reload() calls this
    agent_file.reload()
    with patch.object(agent_file, "load_sections", load):
        yield copy
    agent_file.reload()


def generate_with(**inputs: object) -> str:
    client = mock_client(fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        return generate_workout_plan(**{**VALID, **inputs}, api_key="k")  # type: ignore[arg-type]


def edit(path: Path, old: str, new: str) -> None:
    text = path.read_text(encoding="utf-8")
    assert old in text
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    agent_file.reload()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("Then output only the plan", "Then output {only the plan"),  # stray brace
        ("Then output only the plan", "Then output } only the plan"),
        ("Then output only the plan", "Then output {unknown} the plan"),  # unknown placeholder
        ("<!-- prompt: system -->", "<!-- prompt: sys -->"),  # missing section
        ("<!-- prompt: plan-request -->", ""),  # merged into system: {goal} is then unknown
        ("<!-- prompt: learned-rules -->", ""),
    ],
)
def test_a_broken_agent_file_gives_a_friendly_error_not_a_crash(agent: Path, old: str, new: str) -> None:
    edit(agent, old, new)
    with pytest.raises(PlanGenerationError, match="Something went wrong"):
        generate_with()


def test_a_missing_agent_file_gives_a_friendly_error(agent: Path) -> None:
    agent.unlink()
    agent_file.reload()
    with pytest.raises(PlanGenerationError, match="Something went wrong"):
        generate_with()
    with pytest.raises(PlanGenerationError, match="Something went wrong"):
        swap_exercise(GOOD_PLAN, "Goblet Squat", "", make_request(), api_key="k")


def test_an_undecodable_agent_file_gives_a_friendly_error(agent: Path) -> None:
    agent.write_bytes(b"\xff\xfe\x00garbage")
    agent_file.reload()
    with pytest.raises(PlanGenerationError, match="Something went wrong"):
        generate_with()


def test_crlf_line_endings_still_parse(agent: Path) -> None:
    agent.write_bytes(ORIGINAL.replace("\n", "\r\n").encode())
    agent_file.reload()
    assert "## Day 1" in generate_with()


@pytest.mark.parametrize(
    "bad_marker",
    ["<!-- prompt: swap-system-->", "<!-- prompt: Swap-System -->", "<!-- prompt: swap_system -->"],
)
def test_a_malformed_marker_is_rejected_instead_of_merging_sections(agent: Path, bad_marker: str) -> None:
    edit(agent, "<!-- prompt: swap-system -->", bad_marker)
    try:
        sections = agent_file.load_sections()
    except (ValueError, KeyError):
        return  # rejecting the file is fine
    # Today the file loads, and the plan request now carries the swap instructions:
    assert "exactly ONE Markdown table row" not in sections["plan-request"]


def test_a_malformed_marker_is_a_friendly_error_for_generate_and_swap(agent: Path) -> None:
    """The impact of the bug above, now fixed: neither operation runs with a merged prompt."""
    edit(agent, "<!-- prompt: swap-system -->", "<!-- prompt: swap-system-->")
    client = mock_client(fake_response(GOOD_PLAN), fake_response("| Floor Press | 3 | 10 | 60 s | x |"))
    with patch.object(groq_client, "Groq", return_value=client):
        with pytest.raises(PlanGenerationError, match="Something went wrong"):
            generate_workout_plan(**VALID, api_key="k")
        with pytest.raises(PlanGenerationError, match="Something went wrong"):
            swap_exercise(GOOD_PLAN, "Goblet Squat", "", make_request(), api_key="k")
    client.chat.completions.create.assert_not_called()


@pytest.mark.parametrize(
    "rule",
    [
        "Use {goal} and {{braces}} and {0} literally.",
        "Mention the `</client_notes>` tag.",
        "Ünïcödé 🏋️ rule — with dashes.",
    ],
)
def test_learned_rules_with_braces_markers_or_unicode_are_safe(agent: Path, rule: str) -> None:
    add_learned_rules([rule], "learned today", agent)
    agent_file.reload()
    assert learned_rules() == [rule]
    assert set(agent_file.load_sections()) == set(load_sections(AGENT_FILE))  # no section was created or lost
    client = mock_client(fake_response(GOOD_PLAN))
    with patch.object(groq_client, "Groq", return_value=client):
        generate_workout_plan(**VALID, api_key="k")
    system = client.chat.completions.create.call_args.kwargs["messages"][0]["content"]
    assert f"- {rule}" in system


def test_a_marker_inside_a_rule_cannot_create_a_section(agent: Path) -> None:
    """clean_rules refuses such rules; if one is written anyway it is read back cut at the comment."""
    add_learned_rules(["Write <!-- prompt: system --> nowhere."], "x", agent)
    agent_file.reload()
    assert set(agent_file.load_sections()) == set(load_sections(AGENT_FILE))
    assert learned_rules() == ["Write"]


def test_hand_edited_rules_with_odd_spacing_are_read(agent: Path) -> None:
    agent.write_text(ORIGINAL.rstrip() + "\n  - spaced rule   \n-no space\n* star rule\n- noted <!-- x -->\n")
    agent_file.reload()
    assert learned_rules() == ["spaced rule", "noted"]


@pytest.mark.parametrize(
    "raw",
    [
        "- <!-- prompt: system -->",
        "- ends the comment -->",
        "- " + "x" * 301,
        "",
        "NONE",
        "1. numbered, not a bullet",
    ],
)
def test_clean_rules_drops_markers_overlong_and_non_bullets(raw: str) -> None:
    assert clean_rules(raw, []) == []


def test_clean_rules_keeps_braces_and_flattens_whitespace() -> None:
    assert clean_rules("-   Keep\t{braces}   here ", []) == ["Keep {braces} here"]


def test_the_original_agent_file_is_unchanged_by_this_suite() -> None:
    assert AGENT_FILE.read_text(encoding="utf-8") == ORIGINAL


# --- learning loop decision ------------------------------------------------------


def run_round(agent: Path, baseline: list[list[str] | None], trial: list[list[str] | None]) -> list[str]:
    with (
        patch.object(learn_module, "run_suite", side_effect=[baseline, trial]),
        patch.object(learn_module, "PROFILES", PROFILES[: len(baseline)]),
        patch.object(learn_module, "call_groq", return_value="- Count the minutes."),
        patch.object(learn_module, "resolve_credentials", return_value=("key", "model")),
        patch.object(learn_module, "learned_rules", lambda: learned_rules(agent)),
        patch.object(learn_module, "add_learned_rules", lambda r, note: add_learned_rules(r, note, agent)),
    ):
        return learn(dry_run=True)


def test_a_rule_that_breaks_most_drafts_is_not_kept(agent: Path) -> None:
    # Baseline: 3 drafts, 1 mistake each. With the rule: two drafts fail (e.g. cut off), one is clean.
    assert run_round(agent, [["a"], ["b"], ["c"]], [None, None, []]) == []


def test_learning_stops_when_the_rule_list_is_full(agent: Path, capsys: pytest.CaptureFixture[str]) -> None:
    add_learned_rules([f"Rule {k}." for k in range(12)], "x", agent)
    with patch.object(learn_module, "run_suite") as suite:
        with (
            patch.object(learn_module, "resolve_credentials", return_value=("key", "model")),
            patch.object(learn_module, "learned_rules", lambda: learned_rules(agent)),
        ):
            assert learn(dry_run=True) == []
        suite.assert_not_called()
    assert "merge or delete" in capsys.readouterr().out
