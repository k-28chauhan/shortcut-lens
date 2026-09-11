---
description: Prepare a GPU job for the human to run on Kaggle or Colab
argument-hint: [handoff name, e.g. h1_erm]
---

Prepare GPU handoff $ARGUMENTS following `docs/RUNBOOK_GPU.md`.

1. Make sure all code needed is committed and `make check` and `make smoke` pass. Push.
2. Write `jobs/$ARGUMENTS.yaml` listing each stage command in order.
3. Give the human, in one block: the commit hash, the jobs file path, the expected run ids,
   the expected output files per run, and a time estimate (from measured timings if available;
   otherwise label it an estimate).
4. If the estimate exceeds 1 GPU-hour, state it prominently and ask for approval.
5. Add a row to the GPU handoff log in STATUS.md with status "prepared".

After the human reports completion: pull artefacts, run `slens validate-run` on each run, record
measured timings, and update the handoff log.
