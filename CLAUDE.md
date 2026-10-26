# CLAUDE.md

Workout Plan Generator: a single-page Streamlit app (Codebasics assignment) that turns a
user's goal, experience, schedule, equipment and injuries into a weekly workout plan via
Groq. Live at https://workout-plan-gen.streamlit.app/ (deploys from `main`).

## Commands

Use `uv` for everything; there is no requirements.txt.

```bash
uv sync                                   # install locked dependencies
uv run streamlit run frontend/app.py      # run the app (needs GROQ_API_KEY in .env)
uv run pytest                             # all tests, Groq mocked (live tests skipped)
uv run pytest -m live                     # tests against real Groq
uv run ruff check . && uv run ruff format --check .
uv run mypy backend frontend              # strict
uv run python -m backend.learning.learn --dry-run   # score live drafts, propose rules
```

A change is done only when ruff, mypy and pytest all pass.

## Layout

```
agents/workout-trainer.md   every prompt the model sees + learned rules
frontend/app.py             Streamlit UI only
backend/core/               no network: constants, models, errors, utils, timing
backend/llm/                planner (public API), prompts + agent_file, retry, client, error_mapping
backend/learning/           offline self-improvement loop (evaluate, learn)
tests/                      pytest; fakes in tests/helpers.py
.claude/agents/             product-manager, software-engineer, qe subagents
```

## Rules

- **Prompts live in `agents/workout-trainer.md`, never in `.py` files** (a test enforces
  this). Sections start at `<!-- prompt: NAME -->`; `{placeholders}` are filled with
  `str.format` by `backend/llm/prompts.py`. Keep markers and placeholders intact.
- **All constants go in `backend/core/constants.py`**; pure helpers in `backend/core/utils.py`.
- `backend/` never imports Streamlit; the frontend never calls Groq directly.
- Public operations (`generate_workout_plan`, `swap_exercise`) raise only
  `PlanGenerationError` with a user-safe message. The app must never crash on any input.
- The model is not trusted: code checks the day count, guarded limitations, stretches in
  the main table and session length after every answer. Keep those checks when changing
  the prompt.
- Keep it simple. The user has asked more than once to remove over-engineering: no new
  layers, frameworks or abstractions unless a task needs them.
- Every change comes with tests: unit tests for pure functions, integration tests through
  the public operations or `streamlit.testing.v1.AppTest`, Groq mocked. Live tests go
  behind `@pytest.mark.live`.

## Groq limits and secrets

- Default model `openai/gpt-oss-20b`, sent `reasoning_effort="low"`. The free tier allows
  8,000 tokens/minute and **200,000 tokens/day**, shared with the live demo. One plan uses
  about 4,000 tokens and a learning round about 50,000, so keep live runs small and say how
  many calls were made.
- Never print, log or copy the value of `GROQ_API_KEY`; only check that it is set. `.env`
  is gitignored. Check diffs for `gsk_` before committing.

## Git

- Commit or push only when asked, and confirm before each push; one approval doesn't
  carry over to the next change.
- Keep the README in step with behaviour changes (structure, checks, the live-iteration
  table).

## The agents and the /feature workflow

New requirements go through the team workflow: `/feature <requirement>`
(`.claude/skills/feature/SKILL.md`). The main session runs it; the agents hand work over
through numbered files in `work/<slug>/`. It runs end to end without the user, who hears
back only when the work is done (or if it would need a commit, push or deploy):

1. **product-manager** grills the requirement and writes `01-brief.md`, deciding open
   questions itself and recording each decision.
2. **software-engineer** challenges it in `02-engineering-review.md` (no code yet).
3. **product-manager** answers that review and writes `03-spec.md`: acceptance criteria and
   tasks.
4. **software-engineer** builds it with unit and integration tests: `04-implementation.md`.
5. **product-manager** checks every criterion (`05-product-review.md`), and at the same time
   **qe** tries to break it (`06-qe-report.md`, bugs as `xfail` tests).
6. **software-engineer** fixes the failures, bugs and small improvements; stage 5 re-checks.
   At most 3 fix rounds, then `07-summary.md`: results, the decisions made for the user to
   confirm, and a backlog for the next run. Nothing is committed.
