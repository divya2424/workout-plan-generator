"""Streamlit UI for the workout plan generator. Run with: streamlit run frontend/app.py"""

import logging
import os
import sys
from collections.abc import Callable
from dataclasses import asdict
from datetime import date
from pathlib import Path

# Make the project root importable when launched as `streamlit run frontend/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st
from dotenv import load_dotenv

from backend.core.constants import (
    DEFAULT_DAYS,
    DEFAULT_SESSION_MINUTES,
    EQUIPMENT_OPTIONS,
    EXPERIENCE_LEVELS,
    GOALS,
    MAX_DAYS,
    MAX_LIMITATIONS_CHARS,
    MAX_SESSION_MINUTES,
    MIN_DAYS,
    MIN_SESSION_MINUTES,
    SESSION_MINUTES_STEP,
)
from backend.core.errors import PlanGenerationError
from backend.core.models import PlanRequest
from backend.core.utils import list_exercises, validate_inputs
from backend.llm.planner import generate_workout_plan, swap_exercise

load_dotenv()
# No-op after the first call, so safe to run on every Streamlit rerun.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def get_api_key() -> str | None:
    """Read the key from .env / environment, falling back to Streamlit secrets."""
    key = os.getenv("GROQ_API_KEY")
    if key:
        return key
    try:
        secret = st.secrets.get("GROQ_API_KEY")
    except Exception:  # no secrets.toml configured
        return None
    return str(secret) if secret else None


def collect_inputs() -> PlanRequest:
    """Render the input widgets in the sidebar and return their values."""
    with st.sidebar:
        st.header("About you")
        goal = st.selectbox("Fitness goal", GOALS, index=None, placeholder="Choose a goal")
        experience = st.selectbox(
            "Experience level", EXPERIENCE_LEVELS, index=None, placeholder="Choose your level"
        )
        days_per_week = st.slider("Training days per week", MIN_DAYS, MAX_DAYS, DEFAULT_DAYS)
        session_minutes = st.slider(
            "Session length (minutes)",
            MIN_SESSION_MINUTES,
            MAX_SESSION_MINUTES,
            DEFAULT_SESSION_MINUTES,
            step=SESSION_MINUTES_STEP,
        )
        equipment = st.multiselect("Equipment access", EQUIPMENT_OPTIONS, placeholder="Select what you have")
        limitations = st.text_area(
            "Injuries or limitations (optional)",
            placeholder='e.g. "bad knees", "no overhead pressing"',
            max_chars=MAX_LIMITATIONS_CHARS,
        )
    return PlanRequest(
        goal=goal or "",
        experience=experience or "",
        days_per_week=int(days_per_week),
        equipment=tuple(equipment),
        session_minutes=int(session_minutes),
        limitations=limitations or "",
    )


def run_with_spinner(spinner_text: str, action: Callable[[], None]) -> bool:
    """Run `action` with a spinner and show any friendly error. True on success.

    `action` must store its result in session state itself. It runs inside the spinner so
    the result is saved even if Streamlit interrupts the script when the spinner closes.
    """
    with st.spinner(spinner_text):
        try:
            action()
        except PlanGenerationError as exc:
            st.error(str(exc))
            return False
    return True


def run_generation(request: PlanRequest) -> bool:
    """Validate the request, then generate a plan and store it in session state."""
    problems = validate_inputs(request)
    if problems:
        for problem in problems:
            st.warning(problem)
        return False

    def action() -> None:
        st.session_state.plan = generate_workout_plan(**asdict(request), api_key=get_api_key())
        st.session_state.plan_inputs = request

    return run_with_spinner("Building your plan...", action)


def render_swap(plan: str, request: PlanRequest) -> None:
    """Let the user replace a single exercise in the current plan."""
    exercises = list_exercises(plan)
    if not exercises:
        return
    with st.expander("Swap an exercise"):
        exercise = st.selectbox("Exercise to replace", exercises)
        reason = st.text_input("Why? (optional)", placeholder="e.g. hurts my wrist, too easy")
        if exercise and st.button("Swap this exercise"):
            target = exercise

            def action() -> None:
                st.session_state.plan = swap_exercise(plan, target, reason, request, api_key=get_api_key())

            if run_with_spinner(f"Finding an alternative to {target}...", action):
                st.rerun()


def render_plan(plan: str, request: PlanRequest) -> None:
    """Show the plan with its Regenerate, Download and Swap controls."""
    with st.container(border=True):
        st.markdown(plan)

    col_regenerate, col_download, _ = st.columns([1, 1, 3])
    # Regenerate reuses the request that produced the current plan, for a fresh variation.
    if col_regenerate.button("Regenerate", width="stretch") and run_generation(request):
        st.rerun()
    col_download.download_button(
        "Download (.md)",
        data=plan,
        file_name=f"workout-plan-{date.today().isoformat()}.md",
        mime="text/markdown",
        width="stretch",
    )
    render_swap(plan, request)


def main() -> None:
    st.set_page_config(page_title="Workout Plan Generator", page_icon="🏋️", layout="wide")
    st.title("🏋️ Workout Plan Generator")
    st.caption(
        "Tell us about yourself in the sidebar and get a weekly plan built around "
        "your goal, schedule, equipment and limitations."
    )

    request = collect_inputs()
    st.session_state.setdefault("plan", None)

    if st.button("Generate Plan", type="primary"):
        run_generation(request)

    plan = st.session_state.plan
    if plan is None:
        st.info("Fill in your details in the sidebar, then click **Generate Plan**.")
        return

    plan_request = st.session_state.plan_inputs
    if plan_request != request:
        st.caption("Your inputs have changed since this plan was made. Click Generate Plan to update it.")
    render_plan(plan, plan_request)


if __name__ == "__main__":
    main()
