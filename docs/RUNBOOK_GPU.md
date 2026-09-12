# RUNBOOK — GPU sessions (Kaggle or Colab)

Claude Code prepares every GPU job; the human runs it. This runbook is the human's checklist.
`notebooks/gpu_runner.ipynb`'s cells are real as of M3 (see below for exactly what each does).

## One-time setup

1. Create a Hugging Face account and a **write** access token.
2. Create a private dataset repo for artefacts, e.g. `<user>/shortcut-lens-artifacts`.
3. Store the token as a secret named `HF_TOKEN`:
   - Kaggle: Notebook → Add-ons → Secrets.
   - Colab: the key icon (Secrets) in the left sidebar; enable notebook access.
4. Kaggle: enable Internet in notebook settings and choose a GPU accelerator (T4). Check your
   remaining weekly GPU quota before long jobs. Colab: Runtime → Change runtime type → T4 GPU.
   Quotas and session limits change — check current values on each platform.

## Every handoff

1. In Claude Code, run `/handoff <name>`. It gives you: the commit hash, the jobs file
   (e.g. `jobs/h1_erm.yaml` -- see ARCHITECTURE §6 for the exact YAML shape: a flat, already-
   expanded list of `{stage, config, seed, space}` steps, one `slens <stage> ...` call each), the
   expected run ids, the expected outputs, and a time estimate.
2. Make sure that commit is pushed to GitHub.
3. Open `notebooks/gpu_runner.ipynb` on Kaggle/Colab. Set the four variables in its first code
   cell: `COMMIT`, `JOBS`, `REPO_URL`, `HF_ARTIFACT_REPO`.
4. Run the cells in order. They:
   1. clone the repo and check out `COMMIT`;
   2. install `uv` and `uv sync --extra cu126` (the CUDA-wheel extra, D-020);
   3. print `nvidia-smi` and a `torch.cuda.is_available()` check;
   4. log in to Hugging Face using the `HF_TOKEN` secret (never pasted into a cell);
   5. run `uv run slens run-jobs $JOBS` (resumes automatically if this cell is re-run after a
      disconnect -- each stage's own config-hash/manifest check, or `train`'s epoch checkpoint,
      decides what is already done; nothing about progress is tracked by the notebook itself);
   6. glob `artifacts/runs/*/manifest.json` for every run the jobs produced and
      `HFHubStore(repo_id=HF_ARTIFACT_REPO).push_run(run_id)` each one to the Hub (whole run
      folder, checkpoints included -- `slens gradcam`, M8, needs them back later).
5. If the session disconnects: reopen, run the cells again from the top (cloning again is
   cheap; step 5 above is what actually resumes).
6. Back on your laptop: tell Claude Code the handoff finished. It runs
   `slens pull-artifacts --run-ids <ids> --repo-id <repo>` and `slens validate-run <id>` for each
   run, records measured timings in STATUS, and continues.
7. Optionally, before spending GPU quota: `uv run slens run-jobs jobs/<file>.yaml --dry-run`
   (locally, no GPU needed) prints the exact `slens <stage> ...` command for every step in the
   jobs file, so you can sanity-check the plan first.

## Rules

- Never paste tokens into notebook cells or commit them.
- Never edit code inside the GPU notebook. If something fails, bring the error back to Claude Code,
  fix it in the repo, push, and re-run from the new commit.
- Keep the browser tab active on Colab during long runs.
