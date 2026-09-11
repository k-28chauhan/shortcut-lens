---
description: Run and report the exit gate for a milestone
argument-hint: [milestone id, e.g. M4]
---

Run the exit gate for milestone $ARGUMENTS exactly as written in `docs/PLAN.md`. Gate thresholds
are frozen; do not reinterpret them.

1. Run `make check`. Report the result.
2. Run every check listed under the gate. For thresholds on results, show the measured value,
   the threshold, and the artefact/manifest the value came from.
3. Confirm every task checkbox for the milestone is ticked, or list what is missing.
4. Confirm STATUS.md, CODE_TOUR.md and DECISIONS.md are updated for this milestone.

Report PASS or FAIL per item and overall. If anything fails, diagnose the likely cause, but do not
change code or thresholds until the human decides. Add the result to the gate log in STATUS.md.
