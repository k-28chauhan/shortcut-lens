---
description: Plan the next unfinished milestone from docs/PLAN.md
argument-hint: [optional milestone id, e.g. M3]
---

Read `docs/STATUS.md` and `docs/PLAN.md`. Identify the milestone to work on: $ARGUMENTS if given,
otherwise the first milestone whose exit gate has not passed.

Confirm the previous milestone's gate passed and the human approved it (check the gate log in
STATUS.md). If not, stop and say so.

Then produce a plan for this milestone only:
1. The goal in one sentence.
2. Each task from PLAN.md, with the files you will create or change.
3. The tests you will write for each task (see docs/TESTING.md).
4. External APIs you must verify by running code, and how.
5. Any GPU handoff this milestone needs.
6. Ambiguities or risks, as questions for the human.

Do not write code. Wait for approval.
