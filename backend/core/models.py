"""Data types shared across the backend."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from backend.core.constants import DEFAULT_SESSION_MINUTES


@dataclass(frozen=True)
class PlanRequest:
    """The client's profile. One immutable value instead of six loose parameters.

    Equality works, so the UI can tell whether the inputs changed since a plan was made.
    Text and equipment are normalised on creation (stripped, stored as a tuple).
    """

    goal: str
    experience: str
    days_per_week: int
    equipment: tuple[str, ...]
    session_minutes: int = DEFAULT_SESSION_MINUTES
    limitations: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "equipment", tuple(self.equipment))
        object.__setattr__(self, "limitations", self.limitations.strip())

    @classmethod
    def from_parts(
        cls,
        goal: str,
        experience: str,
        days_per_week: int,
        equipment: Iterable[str],
        session_minutes: int = DEFAULT_SESSION_MINUTES,
        limitations: str = "",
    ) -> PlanRequest:
        return cls(goal, experience, days_per_week, tuple(equipment), session_minutes, limitations)
