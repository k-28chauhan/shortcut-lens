# RUNBOOK — GPU sessions (Kaggle or Colab)

Claude Code prepares every GPU job; the human runs it. This runbook is the human's checklist.
Claude Code refines the exact notebook cells in M3.

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
   (e.g. `jobs/h1_erm.yaml`), the expected run ids, the expected outputs, and a time estimate.
2. Make sure that commit is pushed to GitHub.
3. Open `notebooks/gpu_runner.ipynb` on Kaggle/Colab. Set the two variables at the top:
   `COMMIT = "<hash>"` and `JOBS = "jobs/<file>.yaml"`.
4. Run the cells in order. They:
   1. clone the repo and check out the commit;
   2. install `uv` and sync the locked environment;
   3. print `nvidia-smi` and the torch CUDA check;
   4. log in to Hugging Face with `HF_TOKEN`;
   5. run `slens run-jobs $JOBS` (resumes automatically if re-run after a disconnect);
   6. upload finished run folders to the artefact repo.
5. If the session disconnects: reopen, run the cells again. Completed stages are skipped and
   training resumes from the last epoch checkpoint.
6. Back on your laptop: tell Claude Code the handoff finished. It runs
   `slens pull-artifacts --run-ids …` and `slens validate-run <id>` for each run, records measured
   timings in STATUS, and continues.

## Rules

- Never paste tokens into notebook cells or commit them.
- Never edit code inside the GPU notebook. If something fails, bring the error back to Claude Code,
  fix it in the repo, push, and re-run from the new commit.
- Keep the browser tab active on Colab during long runs.
