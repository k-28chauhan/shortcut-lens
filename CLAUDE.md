# CLAUDE.md — shortcut-lens

You are building **shortcut-lens**: a tool that automatically finds the hidden failure groups
("slices") of an image classifier, names them in plain English, checks whether the named cause
is real using counterfactual image edits, and fixes the failures with last-layer retraining.
Everything is validated against shortcuts we plant ourselves, so we always know the right answer.

The human owner (GS) is building this as their first serious ML project and **will read every
line of this codebase afterwards to learn it**. Optimise for, in this order:
**correctness → honesty of results → readability → speed**. Clever code is a bug.

---

## 1. Read these at the start of every session

1. `docs/STATUS.md` — where we are, what is next, open questions. Always read first.
2. `docs/PLAN.md` — milestones, tasks, exit gates. Work only on the current milestone.
3. `docs/PRD.md` — what we are building and why. Source of truth for scope.
4. `docs/ARCHITECTURE.md` — repo layout, zones, interfaces, data contracts.
5. `docs/EXPERIMENTS.md` and `docs/TESTING.md` — when working on experiments or tests.
6. `docs/DECISIONS.md` — prior decisions. Do not silently contradict one.

If these documents conflict, PRD wins on scope, ARCHITECTURE wins on structure, and you
ask the human before proceeding.

---

## 2. Workflow

- **One milestone at a time.** Never start milestone N+1 until milestone N's exit gate in
  `docs/PLAN.md` passes and the human has approved moving on.
- **Plan before coding.** At the start of a milestone (or any task touching more than ~3 files),
  present a short plan: files to create or change, tests to write, open questions. Wait for approval.
- **Tests with the code.** Write tests alongside every module (see `docs/TESTING.md`). For pure
  logic (metrics, statistics, EM, planting) write the test first.
- **Small commits.** One logical change per commit, conventional messages
  (`feat(discovery): add error-aware EM`, `test(stats): bootstrap coverage`).
- **Definition of done** for any task: code + tests + `make check` green + docs updated
  (STATUS, CODE_TOUR, DECISIONS when relevant). Tick the checkbox in `docs/PLAN.md`.
- **End of milestone ritual:** run the gate (`/gate Mx`), write a milestone summary in
  `docs/STATUS.md` (what was built, files to read in order, how to run it, gate results,
  decisions made, open questions), then **stop and wait for the human.**
- **Ask instead of guessing** on anything touching evaluation integrity, dataset construction,
  metric definitions, or scope. Guess freely on naming and small style choices.

---

## 3. Integrity rules (non-negotiable)

These exist so every number we publish survives scrutiny. Breaking one is worse than being slow.

1. **Group-label firewall.** Ground-truth group labels (background, patch presence, etc.) may only
   be read inside the *oracle zone*: `build/`, `oracle/`, `evaluation/`, `verification/`,
   `mitigation/oracle_ref/`, `report/`. The *label-free zone* (`data/`, `models/`, `training/`,
   `embeddings/`, `discovery/`, `naming/`, `mitigation/label_free/`, `mitigation/selection.py`)
   must never import from the oracle zone, directly or indirectly. This is enforced by
   import-linter contracts in `pyproject.toml`. **Never weaken, delete or bypass a contract.**
   Datasets are injected into label-free code by the CLI (the composition root) and expose only
   `{"image", "y", "example_id"}`.
2. **No group labels for any choice in the label-free pipeline.** That includes checkpoint
   selection, hyperparameter tuning, number of clusters, thresholds, and early stopping. ERM
   checkpoints are selected by average validation accuracy. Oracle-tuned variants are allowed only
   as clearly labelled reference rows, implemented in `mitigation/oracle_ref/`.
3. **The test split is for final reporting only.** Nothing is selected or tuned on it. Final
   evaluations go through `slens evaluate --final`, which appends to `results/final_eval_log.csv`.
   Re-running a final evaluation for the same config requires `--rerun-reason "..."`.
4. **Frozen artefacts stay frozen.** `vocab/*.yaml`, gate thresholds in `docs/PLAN.md`, metric
   definitions in `docs/PRD.md` §12, and experiment specs in `docs/EXPERIMENTS.md` are fixed before
   results exist. Changing one requires a `docs/DECISIONS.md` entry *and* telling the human
   explicitly. Never adjust any of them to make a result look better.
5. **No fabricated numbers.** Never write a result, benchmark figure or "expected value" into code,
   docs or README unless it was produced by a committed config and recorded in a manifest. Use
   `[TBD]` placeholders. Literature numbers may appear only as labelled references.
6. **Surprising results get investigated, not tuned away.** If a result looks too good, too bad or
   odd, stop, check for bugs (leakage, misalignment of example ids, wrong split), and report
   what you found to the human.
7. **Every reported number is traceable.** Each run writes `manifest.json` (git commit, dirty flag,
   config, config hash, seed, package versions, hardware, input artifact ids). Tables are generated
   from artifacts by `slens report`, never typed by hand.

---

## 4. Compute rules

- Assume the development machine has **no GPU**. Locally, run only CPU smoke configs
  (`configs/**/smoke*.yaml`, SyntheticShapes, tiny subsets).
- GPU work is **handed off** to the human, who runs it on Kaggle or Colab using
  `notebooks/gpu_runner.ipynb` and `docs/RUNBOOK_GPU.md`. Use `/handoff` to prepare a job:
  a jobs YAML under `jobs/`, the exact commit to run, expected outputs, and a time estimate.
- Before proposing any sweep, **measure one real run**, extrapolate, and ask before anything
  estimated above 1 GPU-hour.
- After a handoff, pull artifacts (`slens pull-artifacts`) and validate them (`slens validate-run`)
  before using them. Mismatched commit or config hash means the run cannot be used.
- Never commit secrets. Tokens come from environment variables (`HF_TOKEN`).

---

## 5. Code standards

- Python 3.11, managed with `uv`. Dependencies pinned in `uv.lock`. Adding a dependency requires
  a one-line DECISIONS entry (why, alternatives).
- Type hints everywhere; `mypy` strict on `src/shortcut_lens`. `ruff` for lint and format.
- Small, pure functions where possible. Separate I/O from logic so logic is testable on CPU.
- Prefer explicit, boring code: plain loops over clever vectorised one-liners when the clever
  version is hard to read, unless it matters for speed (then comment why).
- No metaprogramming, no plugin registries built with decorators and import side effects,
  no deep inheritance. A simple dict registry in one file is fine.
- Fail loudly. No `except Exception: pass`. Validate inputs at module boundaries
  (shapes, id alignment, split names) with clear error messages.
- Seeds are explicit parameters, never hidden globals. Use `numpy.random.Generator`, not the
  legacy global RNG.
- Configs are Pydantic models loaded from YAML. No hardcoded paths or magic constants in logic.
- Notebooks are allowed only as thin launchers (`notebooks/`). All logic lives in `src/`.
- **Verify external APIs by running code**, not from memory: torchvision weights enums and the
  Oxford-IIIT Pet target types, open_clip model and pretrained tags, WILDS / Waterbirds metadata
  columns, BLIP model ids, pytorch-grad-cam API. Record what you verified in DECISIONS.

---

## 6. Learning-friendly requirements (the human will study this code)

- Every module starts with a docstring of 5–15 lines: what the module does, **the ML concept
  behind it in plain English**, and where it sits in the pipeline.
- Comments explain **why**, not what. Non-obvious maths gets a comment with the formula and a
  paper reference (e.g. EM updates, BH procedure, AFR weights).
- Keep `docs/CODE_TOUR.md` current: a recommended reading order with one paragraph per module
  and its key functions. Update it whenever a module lands.
- When you make a non-obvious choice, add a DECISIONS entry. These become interview answers.
- Prefer names that teach: `worst_group_accuracy`, not `wga_fn`; `val_a`, `val_b`, not `v1`, `v2`.

---

## 7. Commands

```bash
make setup          # uv sync, install pre-commit hooks
make check          # ruff + mypy + import-linter + pytest (fast tests); must be green before "done"
make test           # fast tests only
make test-slow      # slow tests (real CLIP, bootstrap coverage simulations, network)
make smoke          # full pipeline end-to-end on SyntheticShapes on CPU (< 3 minutes)
make reproduce-cpu  # regenerate all tables and figures from released artifacts
make app            # run the Gradio slice explorer locally

slens --help        # CLI: build, train, embed, reliance, discover, confirm, name, verify,
                    # mitigate, evaluate, report, gradcam, export-demo, run-jobs,
                    # pull-artifacts, validate-run
```

Project slash commands live in `.claude/commands/`: `/next-milestone`, `/gate`, `/decision`,
`/explain`, `/handoff`.

---

## 8. Do not

- Do not start work outside the current milestone, or add features not in the PRD.
- Do not touch the test split for anything except `evaluate --final`.
- Do not edit `vocab/*.yaml` or gate thresholds without the DECISIONS + human process above.
- Do not download datasets or model weights not listed in `docs/PRD.md` §7 and
  `docs/ARCHITECTURE.md` without asking.
- Do not launch long-running jobs locally or leave background processes running.
- Do not write marketing language in README or docs. State what was measured.
