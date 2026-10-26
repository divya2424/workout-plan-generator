---
name: feature
description: Take a requirement through the team workflow. The product-manager grills it, agrees the tasks with the software-engineer, the engineer builds it, then product and QE review and test it, and the engineer fixes what they find. Use when the user brings a new requirement, feature, or change to existing behaviour, or types /feature.
argument-hint: <requirement in plain words>
---

# Feature workflow: product -> engineering -> product + QE -> fixes

You (the main session) run this workflow. The agents can't talk to each other directly:
each starts with no memory, so they hand work over through files in one folder. Your jobs
are to start each agent with the right files, pass questions between agents, and stop the
loop at the limits below.

**Run it end to end without the user.** The user wants to hear back only when the work is
done. Don't ask them questions or for approval at any stage, and don't send progress
updates. Questions go to the agent that can answer them, and decisions are recorded in the
files so the user can review them at the end. The only reasons to stop early are in
"Stopping early" below.

Requirement: $ARGUMENTS

## Setup

Make a short kebab-case slug for the requirement and create `work/<slug>/`. Write the
requirement, word for word, to `work/<slug>/00-requirement.md`. Every agent prompt below
starts with: "You are in the /feature workflow. Read every file in work/<slug>/ in
number order before you start."

## Stage 1: product grills the requirement

Start **product-manager**. Ask it to write `01-brief.md`:
- The user problem, who has it, and why it matters for them.
- What it would change in the plans users get, with an example.
- **Decisions**: every open question, the options, the choice it made, and why. Choose
  the safest option that still meets the user need. The user reviews these at the end.
- Draft acceptance criteria.
- Out of scope.

## Stage 2: engineering challenges the brief

Start **software-engineer** in review mode (no code changes). Ask it to write
`02-engineering-review.md`:
- How it would build it: files touched, the simplest approach, what it reuses.
- Which acceptance criteria code can check, and which only the prompt can influence.
- Risks: token budget, rate limits, safety checks it could weaken, effect on tests.
- A simpler alternative, if there is one.
- Questions or objections for product, marked **blocking** or **non-blocking**.

## Stage 3: product writes the spec and assigns the tasks

Start **product-manager**. Ask it to answer every point in `02-engineering-review.md` and
write `03-spec.md`:
- Final acceptance criteria, numbered AC1, AC2, ... Each one must be testable.
- Tasks for the software-engineer, numbered T1, T2, ... Each task gives what to change,
  the ACs it satisfies, and the tests it needs.
- What was decided about each engineering objection, and why.

If the engineer raised a blocking objection that the spec doesn't resolve, repeat stages
2 and 3 once. If it is still open after that, product decides, choosing the simpler and
safer option, and records it under "Decisions" for the user to review at the end.

## Stage 4: engineering builds it

Start **software-engineer**. Ask it to do every task in `03-spec.md` and write
`04-implementation.md`: for each task, what changed (file links), the tests added, and
anything it deviated from and why. It runs the full gate (ruff, mypy, pytest) and pastes
the real output.

## Stage 5: product reviews and QE tests (start both at once)

Start both agents in the same turn:
- **product-manager**: check each AC against real behaviour (generated plans where
  needed, within the token budget). Write `05-product-review.md`: each AC as PASS or FAIL
  with evidence, then any gaps, then improvements marked **now** (small, inside this
  scope) or **later** (backlog).
- **qe**: test the change and the code around it. Write `06-qe-report.md` in its report
  format. Every bug comes with a failing test marked `xfail(strict=True, reason="BUG: ...")`.

## Stage 6: fix round

If `05` has a FAIL or a **now** improvement, or `06` has a bug, start
**software-engineer**. Ask it to fix them, turn each `xfail` bug test into a normal passing
test, and add a "## Fix round N" section to `04-implementation.md` with the gate output.

Then run stage 5 again, limited to what changed. Product re-checks the failed ACs and QE
re-tests the fixed bugs and their surroundings. Each writes a "## Round N" section in its
own file.

**Stop after 3 fix rounds**, even if items are still open. They go in the summary.

## Done

The feature is done when every AC passes, no crash, unsafe-plan or wrong-plan bug is open,
and the gate is green. Then:
1. Write `07-summary.md`: what was built, the ACs, every decision the agents made on the
   user's behalf, the bugs found and fixed, what's still open, and the **later** backlog.
2. Now report to the user, for the first time since the run started. Say what was built,
   the AC results, and the gate output. List the decisions for them to confirm or change.
   Name anything still open, and offer the backlog as the next /feature run.
3. Don't commit or push: the changes stay in the working tree for the user to review
   (see CLAUDE.md).

## Stopping early

Come back to the user before the end only if:
- Going on would mean an action that leaves this machine or is hard to undo: a commit,
  a push, a deploy, deleting files the workflow didn't create, or a paid or quota-heavy
  service beyond the budgets below.
- Groq's daily token cap is reached and the remaining stages need live calls. Finish
  everything that doesn't need them first.
- The requirement turns out to contradict a safety or scope rule (for example nutrition
  advice), so no version of it can be built safely.

Even then, finish what you can and write `07-summary.md` before reporting.

## Rules for running the workflow

- Each agent writes only its own files: product writes 01, 03 and 05; engineering writes
  02 and 04 and the code; QE writes 06 plus `tests/` and `qe/`.
- Pass the slug and the stage in every agent prompt. Never paste a whole file into a
  prompt; the agent reads it from the folder.
- Live Groq calls share 200,000 tokens a day with the live demo. Give each stage a budget
  (product review: 6 plans at most; QE: 8 calls at most), and stop live work on a
  "tokens per day" 429.
- Run agents in the background and wait for each to finish before the next stage (product
  and QE in stage 5 run together). Between stages, check that the expected file was
  written. If an agent didn't write it, start that agent again once with the reason.
