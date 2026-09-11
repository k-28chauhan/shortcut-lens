# KICKOFF — how to build shortcut-lens with Claude Code

This package is the complete spec. Claude Code writes the code; you review, run the GPU jobs, and learn.

## What's in the package

| File | Purpose |
|---|---|
| `CLAUDE.md` | Rules Claude Code reads automatically every session |
| `docs/PRD.md` | What we're building, requirements, metrics, scope |
| `docs/ARCHITECTURE.md` | Layout, the label firewall, interfaces, data contracts |
| `docs/PLAN.md` | Milestones M0–M8 with tasks, tests and exit gates |
| `docs/EXPERIMENTS.md` | Pre-registered experiments E1–E8, figures, compute budget |
| `docs/TESTING.md` | Test strategy and required tests |
| `docs/DECISIONS.md` | Decisions already made, and why |
| `docs/ABSTRACT.md` | The claim, method and what would falsify it |
| `docs/STATUS.md` | Progress log (Claude Code keeps it current) |
| `docs/CODE_TOUR.md` | Reading order through the code (filled in as modules land) |
| `docs/LEARNING_PATH.md` | Your study plan, exercises and interview prep |
| `docs/RUNBOOK_GPU.md` | Your checklist for Kaggle/Colab GPU sessions |
| `.claude/commands/` | Slash commands: `/next-milestone`, `/gate`, `/decision`, `/explain`, `/handoff` |
| `vocab/` | Frozen phrase lists for naming and evaluation (do not edit after results exist) |

## Setup

1. Create an empty GitHub repo named `shortcut-lens` and clone it.
2. Copy everything from this package into it, including the hidden `.claude/` folder. Commit:
   `git add . && git commit -m "docs: project spec"`.
3. Install Claude Code (see the official docs) and run `claude` inside the repo folder.

## First prompt

```
Read CLAUDE.md, then every file in docs/. Summarise the project back to me in 10 lines,
list anything in the spec that is ambiguous or contradictory, then run /next-milestone to
propose the plan for M0. Do not write code until I approve the plan.
```

## The loop for each milestone

1. `/next-milestone` → Claude Code proposes a plan. Read it; ask questions; approve or adjust.
2. Let it work. Review diffs as it goes; interrupt if it drifts outside the milestone.
3. When it says the milestone is done: `/gate Mx`. Read the gate report.
4. Read the files listed in the milestone summary in `docs/STATUS.md`, in order. Use
   `/explain <file>` for anything unclear. Do the matching section of `docs/LEARNING_PATH.md`.
5. Approve moving on. Start a fresh session (`/clear`) for the next milestone so context stays clean —
   STATUS.md carries the memory between sessions.

## GPU handoffs (M3, M5, M6)

When Claude Code says a GPU job is needed: `/handoff <name>`, then follow `docs/RUNBOOK_GPU.md`.
Tell Claude Code when it's finished so it can pull and validate the artefacts.

## When things go wrong

- A gate fails → ask Claude Code for a diagnosis first, not a fix. Gate thresholds do not move.
- A result looks surprising → ask it to check for leakage, id misalignment and wrong splits before
  believing it.
- It wants to change a frozen file (vocab, thresholds, metric definitions) → it must write a
  DECISIONS entry and explain why; you decide.

## Rough schedule

Week 1: M0–M2 · Week 2: M3–M4 (+ GPU H1, H2) · Week 3: M5–M7 (+ GPU H3) · Week 4: M8 · Week 5: buffer/stretch.
