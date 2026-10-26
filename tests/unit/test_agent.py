import ast
import shutil
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch

import groq
import pytest

from backend.core.constants import AGENT_FILE, MAX_NEW_RULES_PER_RUN
from backend.core.errors import PlanGenerationError
from backend.core.utils import validate_inputs
from backend.learning import learn as learn_module
from backend.learning.evaluate import PROFILES, find_issues
from backend.learning.learn import Results, clean_rules, count_mistakes, learn
from backend.llm import agent_file
from backend.llm.agent_file import add_learned_rules, learned_rules, load_sections
from backend.llm.prompts import build_swap_prompt, build_user_prompt, swap_system_prompt, system_prompt
from tests.helpers import GOOD_PLAN, make_request, status_error

PLAN_SECTIONS = {"system", "plan-request", "swap-system", "swap-request", "learned-rules-heading"}


@pytest.fixture
def agent_copy(tmp_path: Path) -> Iterator[Path]:
    """A throwaway copy of the agent file, so tests never edit the real one."""
    copy = tmp_path / "agent.md"
    shutil.copy(AGENT_FILE, copy)
    yield copy
    agent_file.reload()


# --- the agent file -----------------------------------------------------------


def test_every_prompt_section_is_present_and_renders() -> None:
    assert PLAN_SECTIONS | {"reflection-system", "reflection-request", "learned-rules"} <= set(
        load_sections()
    )
    request = make_request(limitations="bad knees")
    for prompt in (
        system_prompt(),
        swap_system_prompt(),
        build_user_prompt(request),
        build_swap_prompt(request, "| A | 3 | 10 | 60 s | x |", ["A"], "bored"),
    ):
        assert prompt.strip()
        assert "{" not in prompt  # no placeholder left unfilled


def test_no_prompt_text_is_left_in_python_code() -> None:
    for path in Path("backend").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert "personal trainer" not in node.value, f"prompt text in {path}"


def test_client_text_is_never_parsed_as_a_template() -> None:
    prompt = build_user_prompt(make_request(limitations="{goal} {0} }{"))
    assert "<client_notes>\n{goal} {0} }{\n</client_notes>" in prompt


def test_learned_rules_are_added_to_the_system_prompt_only() -> None:
    assert "Lessons from earlier plans" not in system_prompt([])
    prompt = system_prompt(["Count the minutes."])
    assert "Lessons from earlier plans" in prompt
    assert prompt.index("- Count the minutes.") < prompt.index("# Output format")
    assert "Count the minutes" not in swap_system_prompt()


def test_added_rules_are_saved_with_a_note_and_read_back_without_it(agent_copy: Path) -> None:
    assert learned_rules(agent_copy) == []
    add_learned_rules(["Rule one.", "Rule two."], "learned 2026-10-26, mistakes 5 -> 2", agent_copy)
    add_learned_rules(["Rule three."], "learned 2026-10-27, mistakes 2 -> 1", agent_copy)
    assert learned_rules(agent_copy) == ["Rule one.", "Rule two.", "Rule three."]
    assert "- Rule one. <!-- learned 2026-10-26, mistakes 5 -> 2 -->" in agent_copy.read_text()
    assert load_sections(agent_copy)["swap-system"] == load_sections()["swap-system"]  # rest untouched


# --- scoring drafts -----------------------------------------------------------


def test_a_draft_is_scored_on_every_rule() -> None:
    request = make_request(days_per_week=3, session_minutes=60, limitations="no overhead pressing")
    draft = GOOD_PLAN.replace("| Push-up |", "| Overhead Press |").replace(
        "Goblet Squat", "Cat-Cow Stretch", 1
    )
    issues = find_issues(draft, request)
    assert "wrote 2 training days instead of 3" in issues
    assert "programmed Overhead Press despite the limitation" in issues
    assert "put stretches in the exercise table (Cat-Cow Stretch)" in issues
    assert any("only about" in issue for issue in issues)  # 2 exercises can't fill an hour
    assert "left out the disclaimer although the client listed a limitation" in issues


def test_test_profiles_are_valid_requests() -> None:
    assert all(validate_inputs(request) == [] for request in PROFILES)


# --- learning -----------------------------------------------------------------


def test_only_short_new_one_line_rules_are_kept() -> None:
    raw = "\n".join(
        [
            "Here are the lessons:",
            "- Add up the minutes before answering.",
            "-   add up the MINUTES before answering.",  # duplicate
            "- Already known.",
            "- " + "x" * 400,  # too long
            "- Sneak <!-- a marker --> in.",
            "* Use 2-3 sets for beginners.",
            "- One more.",
        ]
    )
    rules = clean_rules(raw, ["already known."])
    assert rules == ["Add up the minutes before answering.", "Use 2-3 sets for beginners."]
    assert len(rules) == MAX_NEW_RULES_PER_RUN


def run_round(agent_copy: Path, baseline: Results, trial: Results, **kwargs: bool) -> list[str]:
    """One learning round against `agent_copy`, with the two test-suite results and the model faked."""
    with (
        patch.object(learn_module, "run_suite", side_effect=[baseline, trial]),
        patch.object(learn_module, "PROFILES", PROFILES[: len(baseline)]),
        patch.object(learn_module, "call_groq", return_value="- Count the minutes."),
        patch.object(learn_module, "resolve_credentials", return_value=("key", "model")),
        patch.object(learn_module, "learned_rules", lambda: learned_rules(agent_copy)),
        patch.object(
            learn_module, "add_learned_rules", lambda rules, note: add_learned_rules(rules, note, agent_copy)
        ),
    ):
        return learn(**kwargs)


def test_rules_that_reduce_mistakes_are_written_to_the_agent_file(agent_copy: Path) -> None:
    assert run_round(agent_copy, [["a", "b"], ["c"]], [["a"], []]) == ["Count the minutes."]
    assert learned_rules(agent_copy) == ["Count the minutes."]
    assert "mistakes 3 -> 1 -->" in agent_copy.read_text()


def test_rules_that_dont_help_are_discarded(agent_copy: Path) -> None:
    before = agent_copy.read_text()
    assert run_round(agent_copy, [["a", "b"]], [["a", "b"]]) == []
    assert agent_copy.read_text() == before


def test_a_dry_run_never_changes_the_agent_file(agent_copy: Path) -> None:
    before = agent_copy.read_text()
    assert run_round(agent_copy, [["a", "b"]], [[]], dry_run=True) == ["Count the minutes."]
    assert agent_copy.read_text() == before


def test_nothing_is_proposed_when_the_baseline_is_clean(agent_copy: Path) -> None:
    with patch.object(learn_module, "propose_rules") as propose:
        assert run_round(agent_copy, [[], None], [[], []]) == []
    propose.assert_not_called()


def rate_limited(retry_after: str) -> PlanGenerationError:
    error = PlanGenerationError("rate limited")
    cause = status_error(groq.RateLimitError, 429)
    cause.response.headers["retry-after"] = retry_after
    error.__cause__ = cause
    return error


def test_only_drafts_that_succeeded_both_times_are_compared() -> None:
    # The live run that motivated this: a rate-limited baseline draft counted 1, its retest 11.
    baseline: Results = [["a"], None, ["b", "c"]]
    trial: Results = [[], ["x"] * 11, ["b"]]
    assert count_mistakes(baseline, trial) == (3, 1)


def test_a_short_rate_limit_is_waited_out() -> None:
    with (
        patch.object(learn_module, "call_groq", side_effect=[rate_limited("20"), "- Rule."]),
        patch.object(learn_module.time, "sleep") as sleep,
    ):
        assert learn_module.propose_rules(["a"], [], "key", "model") == ["Rule."]
    sleep.assert_called_once_with(21.0)


def test_the_daily_cap_stops_the_round_instead_of_counting_as_mistakes() -> None:
    with (
        patch.object(learn_module, "call_groq", side_effect=rate_limited("550")),
        pytest.raises(learn_module.LearningStopped, match="about 9 min"),
    ):
        learn_module.run_suite("system", 1, "key", "model")


def test_reflection_sees_the_trainers_instructions_and_lessons() -> None:
    with patch.object(learn_module, "call_groq", return_value="NONE") as call:
        assert learn_module.propose_rules(["Day 1 too long"], ["Old lesson."], "key", "model") == []
    request = call.call_args.args[0][1]["content"]
    assert "# Hard rules" in request
    assert "- Old lesson." in request
    assert "- Day 1 too long" in request


# --- prompt edits P1-P5 and the agent-file loader (QE bug 9) --------------------------


@pytest.mark.parametrize(
    ("section", "phrase"),
    [
        ("system", "stretching-only day"),
        ("system", "Notes column and Progression"),
        ("system", "15 min = 3 min warm-up"),
        ("system", "No header rows inside the table"),
        ("swap-system", "Never a stretch"),
    ],
)
def test_the_prompt_keeps_each_rule_from_the_qe_fixes(section: str, phrase: str) -> None:
    assert phrase in load_sections()[section]


@pytest.mark.parametrize(
    "bad_marker",
    [
        "<!-- prompt: swap-system-->",
        "<!-- prompt: Swap-System -->",
        "<!-- prompt: swap_system -->",
        "  <!-- prompt: swap-system -->",
        "<!--prompt: swap-system -->",
    ],
)
def test_a_malformed_marker_fails_to_load(agent_copy: Path, bad_marker: str) -> None:
    text = agent_copy.read_text().replace("<!-- prompt: swap-system -->", bad_marker, 1)
    agent_copy.write_text(text)
    with pytest.raises(ValueError, match="malformed"):
        load_sections(agent_copy)


def test_a_learned_rule_that_mentions_a_marker_in_its_bullet_still_loads(agent_copy: Path) -> None:
    agent_copy.write_text(agent_copy.read_text() + "- Never write <!-- prompt: x --> in a plan.\n")
    assert set(load_sections(agent_copy)) == set(load_sections())


def test_a_rule_whose_trial_has_more_failed_drafts_is_discarded(agent_copy: Path) -> None:
    assert run_round(agent_copy, [["a", "b"], ["c"]], [None, []]) == []
    assert learned_rules(agent_copy) == []


def test_a_rule_with_equal_failures_and_fewer_mistakes_is_kept(agent_copy: Path) -> None:
    assert run_round(agent_copy, [None, ["a", "b"]], [None, ["a"]]) == ["Count the minutes."]


def test_a_non_numeric_retry_after_uses_the_default_wait() -> None:
    with (
        patch.object(learn_module, "call_groq", side_effect=[rate_limited("soon"), "- Rule."]),
        patch.object(learn_module.time, "sleep") as sleep,
    ):
        assert learn_module.propose_rules(["a"], [], "key", "model") == ["Rule."]
    sleep.assert_called_once_with(learn_module.MAX_RATE_LIMIT_WAIT_SECONDS + 1)


# --- new issue types in the dry run (AC18) ----------------------------------------------


def test_a_draft_needing_missing_equipment_is_reported() -> None:
    draft = GOOD_PLAN.replace("| Push-up |", "| Barbell Row |")
    issues = find_issues(draft, make_request(equipment=["No equipment"]))
    assert (
        "programmed Goblet Squat, Dumbbell Row, Barbell Row, which needs equipment the client doesn't have"
        in issues
    )


def test_a_banned_movement_in_a_warm_up_or_its_bullets_is_reported() -> None:
    draft = GOOD_PLAN.replace(
        "## Day 1 - Full Body A\n",
        "## Day 1 - Full Body A\n**Warm-up (5 min):** Walk.\n- 10 walking lunges\n\n",
    ).replace("## Day 2 - Full Body B\n", "## Day 2 - Full Body B\n**Cool-down:** Jump rope, 2 min.\n\n")
    issues = find_issues(draft, make_request(limitations="bad knees"))
    assert "put movements banned by the limitation in the Day 1 warm-up and Day 2 cool-down" in issues


def test_a_stretch_only_or_rest_day_is_reported() -> None:
    draft = GOOD_PLAN.replace("| Push-up |", "| Hamstring Stretch |").replace(
        "| Goblet Squat | 3 | 10 | 60s | |", "| Quad Stretch | 2 | 30 s | - | |"
    )
    issues = find_issues(draft, make_request())
    assert "Day 2 had no strength or conditioning exercise (rest or stretching only)" in issues
