"""Read and update the agent file (agents/workout-trainer.md) that holds every prompt.

The file is split into sections by `<!-- prompt: NAME -->` markers. Templates are filled with
str.format, so the values (the client's text, learned rules) are never parsed as templates.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from backend.core.constants import AGENT_FILE, PROMPT_SECTIONS

_MARKER = re.compile(r"^<!-- prompt: ([a-z-]+) -->$", re.MULTILINE)
# Any line that looks like a marker; each must be an exact _MARKER, or two sections would merge.
_MARKER_LIKE = re.compile(r"^[ \t]*<!--\s*prompt", re.MULTILINE | re.IGNORECASE)
_RULE_LINE = re.compile(r"^- (.+?)(?:\s*<!--.*-->)?$")  # a learned rule, optionally with a note


@lru_cache(maxsize=4)
def load_sections(path: Path = AGENT_FILE) -> dict[str, str]:
    """Section name -> text. Cached: call `reload()` after the file changes."""
    text = path.read_text(encoding="utf-8")
    if len(_MARKER_LIKE.findall(text)) != len(_MARKER.findall(text)):
        raise ValueError(f"{path} has a malformed <!-- prompt: name --> marker")
    parts = _MARKER.split(text)  # [preamble, name1, body1, name2, body2, ...]
    sections = {name: body.strip() for name, body in zip(parts[1::2], parts[2::2], strict=True)}
    # A marker misspelt beyond what _MARKER_LIKE sees ("<!--- prompt", "promt") loses its section.
    if missing := [name for name in PROMPT_SECTIONS if name not in sections]:
        raise ValueError(f"{path} is missing the prompt section(s) {', '.join(missing)}")
    return sections


def reload() -> None:
    load_sections.cache_clear()


def render(name: str, path: Path = AGENT_FILE, **values: object) -> str:
    """The `name` section with its {placeholders} filled in. KeyError names a missing value."""
    return load_sections(path)[name].format(**values)


def learned_rules(path: Path = AGENT_FILE) -> list[str]:
    """The learned rules, without the notes recorded next to them."""
    lines = load_sections(path)["learned-rules"].splitlines()
    return [match.group(1) for line in lines if (match := _RULE_LINE.match(line.strip()))]


def learned_rules_block(rules: list[str] | None = None, path: Path = AGENT_FILE) -> str:
    """`rules` (default: the learned rules) as a prompt section, or "" when there are none."""
    rules = learned_rules(path) if rules is None else rules
    if not rules:
        return ""
    bullets = "\n".join(f"- {rule}" for rule in rules)
    return f"\n{load_sections(path)['learned-rules-heading']}\n{bullets}\n"


def add_learned_rules(rules: list[str], note: str, path: Path = AGENT_FILE) -> None:
    """Append `rules` to the learned-rules section (the last one in the file), each tagged with `note`."""
    text = path.read_text(encoding="utf-8")
    if list(load_sections(path))[-1] != "learned-rules":
        raise ValueError(f"{path} must end with the learned-rules section")
    lines = "".join(f"- {rule} <!-- {note} -->\n" for rule in rules)
    path.write_text(text.rstrip("\n") + "\n" + lines, encoding="utf-8")
    reload()
