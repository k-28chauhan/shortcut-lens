# DECISIONS — shortcut-lens

One entry per non-obvious choice. Format: id, date, decision, why, alternatives considered, revisit if.
These entries are the source of interview answers — keep the "why" concrete.

---

**D-001 · 2026-09-11 · Oxford-IIIT Pet (cat vs dog) as the planted-shortcut base dataset.**
Why: natural images at real resolution, so CLIP and BLIP can see the patch; binary task gives four
groups that mirror Waterbirds exactly; licence reported as CC BY-SA 4.0, which permits a public demo
(verify before publishing); trimap masks exist for possible background edits later.
Alternatives: CIFAR-10 cat vs dog (stronger shortcut learning, but 32×32 images make naming weak);
Imagenette (no natural binary pair). Revisit if: gate G3 fails (see D-002).

**D-002 · 2026-09-11 · Fallback order if the model does not learn the planted patch.**
Why: discovery can only be evaluated if the model actually relies on the shortcut.
Order: (1) check for bugs; (2) use ρ=0.99 for the headline config; (3) switch the base to CIFAR-10
cat vs dog with the same planting code. Gate thresholds never change.

**D-003 · 2026-09-11 · Group-label firewall via dependency injection + import-linter.**
Why: leaking group labels into label-free code is the easiest way to produce fake results. Making it
mechanically impossible is better than being careful. Datasets are built in the oracle zone and
injected by the CLI; they only expose image, class label and id.
Alternatives: code review only (too easy to miss); materialising rendered images to disk (large for sweeps).

**D-004 · 2026-09-11 · ERM checkpoint selection by average validation accuracy.**
Why: selecting by worst-group accuracy uses group labels and would contaminate the "label-free" claim.
Alternatives: last epoch (also fine; noted as a sensitivity check if time allows).

**D-005 · 2026-09-11 · Validation split into val_a (discover, retrain) and val_b (confirm, select).**
Why: discovering and confirming on the same data lets random clusters look like real failures.
Alternatives: cross-fitting (more data-efficient, more complex). Revisit if val_b is too small in realistic mode.

**D-006 · 2026-09-11 · Report both balanced and realistic validation modes.**
Why: Waterbirds' validation set is group-balanced within class, which makes discovery and last-layer
retraining look easier than in practice. Realistic mode shows the practical picture.

**D-007 · 2026-09-11 · Sensitivity map x-axis is measured reliance (R_net), not correlation strength ρ.**
Why: at low ρ, discovery may fail simply because the model did not use the shortcut. Plotting against
measured reliance separates "the model didn't learn it" from "the tool didn't find it".

**D-008 · 2026-09-11 · Null-patch control for interventions.**
Why: adding any square occludes part of the image. A grey square of the same size isolates the effect
of the shortcut's colour from the effect of occlusion.

**D-009 · 2026-09-11 · No random crops for planted data; horizontal flip only for all datasets.**
Why: random crops could cut the patch out, silently lowering the effective ρ. Using one augmentation
policy for all datasets keeps comparisons clean.

**D-010 · 2026-09-11 · Last-layer mitigation on frozen embeddings, on CPU.**
Why: DFR showed the final layer is enough for this class of problem; it makes mitigation cheap enough
to run many seeds and variants. Full JTT retraining is stretch S3.

**D-011 · 2026-09-11 · Implement the Domino-style mixture from scratch (EM), tested against sklearn.**
Why: sklearn's GMM cannot include a weighted error-likelihood term; writing EM is also the best
learning exercise in the project. With the error weight set to zero it must match sklearn.
Note: this is a per-class simplification of Domino, and the write-up must say so.

**D-012 · 2026-09-11 · Frozen vocabulary and evaluation keyword files, hashed.**
Why: if the phrase list is edited after seeing results, the naming numbers mean nothing. Freezing
before any naming run makes the result honest. The `no_artifacts` variant measures how much the
result depends on including artefact phrases at all.

**D-013 · 2026-09-11 · Typer + Pydantic + YAML for configs instead of Hydra.**
Why: the owner will read every line; explicit Pydantic models are easier to learn than Hydra's
composition magic. Sweeps are handled by a small grid runner.

**D-014 · 2026-09-11 · uv for environment management; pinned lockfile.**
Why: fast, reproducible installs on laptops, CI and Kaggle/Colab.

**D-015 · 2026-09-11 · Grad-CAM is presented as illustration only.**
Why: saliency maps can look convincing while being wrong (Adebayo et al., 2018). Evidence comes from
counterfactual interventions.

**D-016 · 2026-09-11 · CSV logging by default; Weights & Biases optional.**
Why: fewer accounts and secrets; results live in the repo's artefacts either way.

**D-017 · 2026-09-11 · Hugging Face Hub as the artefact store for GPU sessions.**
Why: works identically from Kaggle and Colab, versioned, free. Token only via `HF_TOKEN`.
Alternative: Google Drive (Colab only).

**D-018 · 2026-09-11 · Demo bundle uses PlantedPets images by default.**
Why: Waterbirds combines CUB and Places images with research-oriented terms; publishing them in a
public app needs a licence check first. Revisit after checking.

**D-019 · 2026-09-11 · Mitigation headline results use realistic val mode; balanced mode is reported for comparability with the literature.**
Why: in balanced mode, retraining on group-balanced data does most of the work regardless of method (see H7a).

**D-020 · 2026-09-11 · uv PyTorch extras (`cpu` / `cu126`) with explicit indexes.**
Why: the dev laptop is CPU-only but GPU handoffs run on Kaggle/Colab T4s; a single pinned torch
build cannot serve both without either downloading unneeded CUDA wheels locally or manually
reinstalling on GPU sessions (breaking the "same lockfile everywhere" reproducibility rule).
Decision: declare optional extras `cpu` and `cu126` in `pyproject.toml`, both pinning identical
`torch==2.14.0` + `torchvision==0.29.0`, marked conflicting in `[tool.uv]`. `[[tool.uv.index]]`
entries for the PyTorch CPU (`download.pytorch.org/whl/cpu`) and CUDA 12.6
(`download.pytorch.org/whl/cu126`) wheel indexes, both `explicit = true`; `[tool.uv.sources]`
routes torch and torchvision to the matching index per extra, with the `cu126` source additionally
marked `sys_platform == 'linux'` (verified by running `uv lock`: the cu126 index only publishes
linux/windows wheels, so without the marker uv's cross-platform resolution fails trying to solve
`cu126` for macOS even though that extra is never installed there). The `pytorch-cpu` index does
carry macOS arm64 wheels, so the `cpu` extra needs no platform marker. All other ML dependencies
(open_clip_torch, transformers, grad-cam, ...) stay as ordinary main dependencies, unaffected by
which extra is active.
Usage: laptop and CI run `uv sync --extra cpu` (`make setup` uses this). The GPU notebook runs
`uv sync --extra cu126`.
Verification (M0, actually run): `uv run python -c "import torch; print(torch.__version__,
torch.version.cuda, torch.cuda.is_available())"` after `--extra cpu` printed `2.14.0 None False` --
a genuine CPU build. `torch==2.14.0`/`torchvision==0.29.0` were also confirmed (via the PyPI JSON
API) to be the matching compatible pair, and both `+cu126` wheels to exist for `cp311`/linux.
`cu126` was chosen over `cu121`/`cu124` (retired for this torch release) and newer `cu128`+
(chosen `cu126` is the more conservative, widely available choice). The exact-CUDA-version check
against the Kaggle/Colab driver and `'sm_75'` in `torch.cuda.get_arch_list()` cannot be run from
this CPU-only machine; both checks are deferred to the first cell of `notebooks/gpu_runner.ipynb`
in M3, to run for real on the GPU session before it is trusted.
Note: `numpy` and `scipy` in the main dependency list (`numpy==2.4.6`, `scipy==1.17.1`) are one
minor version behind the latest PyPI release at the time of writing (`numpy==2.5.3` requires
Python >=3.12, which conflicts with the `requires-python = ">=3.11,<3.12"` pin in D-021; `scipy`
followed for consistency) -- discovered by letting `uv lock` resolve unpinned versions first, then
hard-pinning what it actually chose, rather than guessing compatibility from memory.
Alternatives: a single CPU-only pin with manual `pip install` of CUDA wheels in the GPU notebook
(rejected: not reproducible from the lockfile, easy to silently drift versions between laptop and
GPU session).

**D-021 · 2026-09-11 · Exact Python 3.11 patch pin via `.python-version`; CI on ubuntu-24.04 with astral-sh/setup-uv.**
Why: reproducibility requires the same interpreter locally, in CI and on GPU notebooks, not just "3.11".
Decision: pin the latest available 3.11.x patch release in `.python-version`;
`requires-python = ">=3.11,<3.12"` in `pyproject.toml`. CI (`ci.yml`) runs on `ubuntu-24.04`, uses
`astral-sh/setup-uv` with dependency caching, and calls `uv python install` so the pinned
interpreter (not the runner's system Python) is used. `notebooks/gpu_runner.ipynb` likewise installs
and uses the uv-managed interpreter from `.python-version`, not the platform's default Python.
Alternatives: floating on "3.11" latest per environment (rejected: silent interpreter drift between
laptop, CI and GPU session is exactly the kind of bug this project's reproducibility rules exist to prevent).

**D-022 · 2026-09-11 · Pre-registered ResNet-18 fallback for the E3 sensitivity sweep.**
Why: the 45-run sweep's GPU cost is only an estimate until H1 timings are measured; deciding the
fallback rule now, before seeing the estimate, keeps the choice honest (see EXPERIMENTS.md's
pre-registration rule).
Decision: after H1 measured timings are in, if the extrapolated cost of the full 45-run sweep
(training + embeddings + reliance) exceeds 6 GPU-hours, run the sweep on ResNet-18 instead of
ResNet-50. Headline experiments (E1, E2, E4, E6, E7) remain ResNet-50 regardless. If the ResNet-18
sweep is used, also run ResNet-18 bridge runs at the headline config (ρ=0.95, patch size 32, seeds
0–2) so the two architectures are comparable at one point.
Alternatives: deciding ad hoc after seeing the estimate (rejected: pre-registration is the point).

**D-023 · 2026-09-11 · torchvision pretrained weights: `IMAGENET1K_V1` for both resnet18 and resnet50.**
Why: comparability with the Waterbirds/DFR literature, which reports results using the V1 (original)
ImageNet weights, not the newer V2 training recipes.
Decision: `models/backbones.py` loads `ResNet18_Weights.IMAGENET1K_V1` /
`ResNet50_Weights.IMAGENET1K_V1`. Exact enum names verified by running code at M3 (CLAUDE.md §5's
"verify external APIs" rule), not assumed from memory.
Alternatives: `IMAGENET1K_V2` (higher-accuracy backbones, but the shift in training recipe —
different augmentation, LR schedule — would make published WGA-gap comparisons to prior work
less clean).

**D-024 · 2026-09-11 · Label-free rule for selecting which (method, space) combination feeds naming, verification and `dfr_discovered`.**
Why: gate G4 (best of all method × space combinations against ground truth) is a legitimate sanity
check that the tool *can* find the shortcut, but it is an oracle-style selection and must never be
used to choose what downstream stages actually consume — that would leak group-label information
into a label-free pipeline.
Decision: add FR-C4 — per run, select the (method, space) combination whose top confirmed slice has
the largest error-rate lift over the rest of its class on `val_b`, tie-break by smallest q-value.
This selection uses only `val_b` error rates and q-values (no group labels) and is implemented in
the label-free zone (`discovery/select_combo.py`). The selected combination is what feeds naming
(M5), verification (M6) and `dfr_discovered` (M7). All (method, space) combinations still appear in
an appendix table (T2) for transparency; G4 keeps using the oracle-best-of-all framing, but only as
a sanity gate, never as the selection rule for downstream stages.
Alternatives: a fixed method/space (simpler, but arbitrary and often not the best-performing
combination); selecting via oracle metrics against `val_b` groups (rejected: exactly the group-label
leakage this project's firewall exists to prevent).

**D-025 · 2026-09-11 · Device policy: `auto` picks MPS > CUDA > CPU; correctness-sensitive paths force CPU.**
Why: the dev machine turned out to have an Apple Silicon GPU (MPS, MacBook Air M4, 24 GB unified
memory), not a CUDA GPU and not CPU-only as originally assumed when D-020/D-021 were written --
local training/embedding should use it by default. But MPS's floating-point reductions are not
bit-stable run to run the way CPU is, so anything checking exact reproducibility must not trust it.
Decision: `device: auto` resolves to `mps` if `torch.backends.mps.is_available()`, else `cuda`,
else `cpu`. Three branches in `training/erm.py` / `embeddings/model_space.py`: CUDA uses fp16
autocast + GradScaler (unchanged); MPS uses float32, optionally `torch.autocast("mps")`, and
explicitly never GradScaler (its MPS support is immature); CPU uses float32. Smoke configs,
correctness gates and determinism tests (e.g. the exact-resume test) set `device: cpu` explicitly
in their YAML, never `auto`. `PYTORCH_ENABLE_MPS_FALLBACK=1` is exported whenever `slens
train`/`slens embed` run locally with `device=mps`, so an op with no MPS kernel falls back to CPU
instead of raising; it is not set for CPU-only paths (`make check`/`test`/`smoke`), where it would
be dead weight (see the Makefile). DataLoader `num_workers=0` for smoke/test/CI configs
(multiprocessing fork issues on macOS); a small configurable number for larger local runs,
defaulting low on macOS specifically.
Alternatives: always forcing CPU locally, as originally decided under the (incorrect) assumption
that the dev machine was CPU-only -- superseded now that the real hardware has a usable GPU;
treating MPS as bit-stable enough for correctness tests (rejected: known MPS reduction-order
nondeterminism would make exact-resume and coverage-simulation tests flaky for reasons unrelated
to real bugs, defeating their purpose).
Revisit if: a specific op has no MPS kernel and no CPU fallback (needs a per-op workaround), or
MPS numerics turn out unstable enough to affect even non-exact-reproducibility results.

**D-026 · 2026-09-11 · macOS torch install: the `cpu` extra already resolves the MPS-capable wheel.**
Why: unlike Linux/Windows, PyTorch does not publish a separate MPS-specific wheel or index -- the
standard macOS arm64 wheel (served by both plain PyPI and `download.pytorch.org/whl/cpu`, already
confirmed identical in D-020) has MPS support built in. No third extra or macOS-specific index
routing is needed alongside the `cpu`/`cu126` extras already declared.
Decision: `make setup` and any local `uv sync --extra cpu` on this machine installs the
MPS-capable build automatically; `torch.backends.mps.is_available()` is what `device: auto` checks
(D-025), not a different install path. The `cu126` extra and the exact-CUDA-version / `sm_75`
checks in `notebooks/gpu_runner.ipynb` (D-020) remain relevant only for cloud GPU handoffs
(Kaggle/Colab T4), never for local runs.
Verification (M0, actually run): `uv run python -c "import torch;
print(torch.backends.mps.is_available(), torch.backends.mps.is_built())"` printed `True True` on
this machine after `uv sync --extra cpu` -- the `cpu` extra's wheel is confirmed MPS-capable
without any further install changes.
Alternatives: none -- this confirms the existing D-020 extras setup already covers this machine
correctly, now that the dev machine's real hardware is known.

**D-027 · 2026-09-12 · Waterbirds: direct CodaLab tarball, not the `wilds` PyPI package.**
Why: `wilds` (latest PyPI release 2.0.0) pulls in `ogb` (Open Graph Benchmark) and `outdated` (an
unmaintained package, no release in years) as transitive dependencies just to load one dataset out
of WILDS' many -- disproportionate for this project's minimal-dependency style (CLAUDE.md §5), and
WILDS' own `WaterbirdsDataset` publishes no checksum for its download (`compressed_size: None`),
so a client-side integrity check would be needed either way.
Decision: `build/waterbirds.py` downloads
`https://worksheets.codalab.org/rest/bundles/0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/`
directly (the exact URL WILDS' own `waterbirds_dataset.py` source uses) and verifies its SHA-256
against a value pinned in `configs/datasets/waterbirds.yaml`, computed and verified by this project
since WILDS publishes none. `download_and_verify()` skips the download if a verified copy is
already cached (outside the repo, `~/.cache/shortcut-lens/downloads/`), and deletes + re-downloads
on a checksum mismatch rather than silently accepting a wrong file. `metadata.csv`'s columns
(`y`, `place`, `img_filename`, `split`) and split encoding (`train=0, val=1, test=2`) were read
directly from WILDS' `waterbirds_dataset.py` and `wilds_dataset.py` source, not assumed. Group
counts per split are asserted against the frozen values in docs/PRD.md §7; a mismatch stops the
build and reports the discrepancy rather than adapting to it (CLAUDE.md §3 rule 6).
Status: superseded by D-029 -- the CodaLab download never became practical; the actual data source
changed to a verified Hugging Face mirror. This entry is kept for why `wilds` itself was rejected
(still true) and as the record of what was tried first.
Alternatives: `wilds` PyPI package (rejected, see Why); a third-party Hugging Face mirror of
Waterbirds (rejected at the time: not the canonical source PRD §7 names, and provenance of a
re-upload from an unverified account cannot be checked the way a checksum against the original
tarball can -- revisited in D-029 once a concrete mirror could actually be checked against known
ground truth, not just trusted).

**D-028 · 2026-09-12 · synthetic_shapes' `background` variant has no naming evaluation, by design.**
Why: `vocab/eval_keywords.yaml` is frozen (D-012) and only defines keywords for the `dot` variant's
groups (`square|dot`, `circle|no_dot`); adding `background`-variant keywords now, before any
naming result exists, would itself require the frozen-file change process (DECISIONS + human
approval, see `vocab/README.md`) for a variant that is not part of the pre-registered CI
integration gate (docs/TESTING.md's gate uses `dot` only) or any pre-registered experiment in
docs/EXPERIMENTS.md.
Decision: build both `dot` and `background` variants in M1 as planned (PLAN.md's M1 task list),
but `background`'s groups (`circle|tint`/`square|no_tint`-style names) are deliberately left out
of `vocab/eval_keywords.yaml`. It remains fully usable for discovery/confirmation (which need no
vocabulary), just not for naming hit@3 scoring. A discovery-only integration test for the
`background` variant (top confirmed slice precision@25 >= 0.8 for the tinted minority group,
analogous to the frozen `dot`-variant CI gate) is added to docs/PLAN.md's M4 task list rather than
written now, since `discovery/` does not exist until M4 -- writing a test against code that does
not exist yet would just be dead code.
Alternatives: extend `vocab/eval_keywords.yaml` now to cover `background` too (rejected: no
naming experiment currently plans to use it, and touching a frozen file preemptively, without a
concrete use, works against the point of freezing it early).

**D-029 · 2026-09-12 · Waterbirds: switched to a verified Hugging Face parquet mirror (`grodino/waterbirds`).**
Why: the CodaLab tarball download (D-027) proved impractically slow on this connection --
~30-40KB/s, multiple hours for ~470MB, confirmed over two attempts (one silently truncated at 21MB
because `curl | tail` masked a real failure exit code; a corrected retry still crawled). The human
found `grodino/waterbirds` on Hugging Face and asked whether it would work. D-027 had already
rejected third-party mirrors on the grounds that their provenance is unverifiable -- what changes
here is that this specific mirror *was* verified, not just trusted: its three splits (train=4795,
validation=1199, test=5794 rows) match WILDS' official split sizes exactly, and -- the real test --
every one of the 12 (class x background) group-count cells across all three splits matches
docs/PRD.md §7's frozen values exactly (e.g. val: landbird|land=467, landbird|water=466,
waterbird|land=133, waterbird|water=133, down to the last image). That level of agreement across
12 independent cells is very strong evidence this is a faithful repackaging of the identical
official data (the mirror's own dataset card description is generic/inaccurate -- it describes a
different, 80/20 train/val split scheme that does not match what the data actually contains --
which is exactly why the counts were checked directly against known ground truth instead of taken
on faith).
Decision: `build/waterbirds.py` now downloads three checksummed parquet files (train/validation/test,
Hugging Face's own split names) from `grodino/waterbirds` instead of one CodaLab tarball. Each
file's URL and SHA-256, plus the mirror's git revision (`e9856c710d0da2e4029d116cdd9d5fce7cc2bc80`)
for reference, are pinned in `configs/datasets/waterbirds.yaml` -- the SHA-256 checks, not the
revision, are what actually gate the build against future drift. Images are embedded as JPEG bytes
in the parquet `image` column; `build_waterbirds()` decodes and caches them to
`builds/waterbirds/<hash>/images/` exactly like the other two datasets, so `RenderedImageDataset`
needs no waterbirds-specific handling. Verified end-to-end: `build_waterbirds()` run against the
real downloaded parquet files reproduces all 12 frozen group-count cells exactly (see
`tests/unit/test_waterbirds.py`, network-marked) and the CLI (`slens build` / `slens data report`)
completed in ~11 seconds total (vs. hours for the CodaLab path).
Alternatives: keep waiting for / manually retrying the CodaLab download (rejected: no reason to
believe the connection would improve, and the mirror is now independently verified); ask the human
to download the CodaLab tarball from a different network (still an option if this mirror ever
becomes unavailable, but unnecessary now).
Revisit if: `grodino/waterbirds` is taken down or the pinned parquet files' checksums ever stop
resolving -- fall back to the CodaLab tarball path preserved in D-027 / git history.

**D-030 · 2026-09-12 · `metrics.py` accuracy-like values are fractions in [0, 1], not percentages.**
Why: PRD §12 and the frozen gate thresholds (docs/PLAN.md) describe some quantities in "points"
(e.g. "WGA gap >= 10 points", recovery's "2-point denominator"), which is ambiguous without a
stated convention -- it could mean percentage points of a 0-100 scale or of a 0-1 scale times 100.
Decision: every function in `metrics.py` (`accuracy`, `group_accuracy`, `worst_group_accuracy`,
`wga_gap`, etc.) returns a fraction in [0, 1]. "N points" anywhere in docs or gate thresholds means
a fraction difference of N/100 on this scale (e.g. `recovery`'s NaN guard fires below a denominator
of 0.02, i.e. "2 points"). Keeps every metric consistent with `sklearn`'s own convention
(`accuracy_score` etc. return fractions), so no metric needs a silent x100/÷100 at its boundary.
Alternatives: percentages (0-100 scale) throughout -- rejected, since it would diverge from
`sklearn`/`scipy` return conventions and require unit conversions at every call site instead of
just at the point where a human-readable "points" figure is reported.

**D-031 · 2026-09-12 · `evaluation/core.py` (M2) takes plain DataFrames, not a `Predictions` dataclass.**
Why: docs/ARCHITECTURE.md §3 sketches a `Predictions` dataclass, but nothing produces a real one
until `training/erm.py` lands in M3 -- adding it to `types.py` now would be a speculative type with
no producer, and the exact fields to include are easier to get right once M3 actually needs to
build one (CLAUDE.md: "don't design for hypothetical future requirements").
Decision: `join_predictions_with_groups`/`group_metrics_table` operate on plain `pandas.DataFrame`s
matching the `predictions/{split}.parquet` schema (`example_id`, `y`, `y_hat`, ...) and the
`oracle.parquet` schema (`example_id`, `group`, `group_name`, ...). The join requires the two
frames to cover exactly the same `example_id` set (raises on any mismatch) rather than silently
keeping the intersection, since a mismatch here means the caller forgot to filter one input to the
right split -- exactly the kind of silent-drop bug docs/TESTING.md warns produces a
plausible-looking wrong number. `Predictions`/`EmbeddingTable` are added to `types.py` in M3 when
`training/erm.py` first produces one; if M3's real needs diverge from the ARCHITECTURE §3 sketch,
that gets its own DECISIONS entry rather than silently drifting from the documented sketch (human
instruction, this session).
Alternatives: add the dataclasses to `types.py` now, matching the sketch exactly, and have
`evaluation/core.py` take one -- rejected per the human's explicit call in this session.

**D-032 · 2026-09-12 · ERM checkpoint selection's "average validation accuracy" means val_a+val_b combined.**
Why: D-004 says selection uses "average validation accuracy," but the val split is itself divided
into `val_a` (discovery) and `val_b` (confirmation) for the discover/confirm firewall (D-005) --
neither PRD nor ARCHITECTURE says which of {val_a, val_b, val_a+val_b} "average validation
accuracy" refers to, and model selection is a decision orthogonal to that firewall's purpose (it
happens before discovery/confirmation exist at all, and doesn't compare a slice's own data against
itself).
Decision: `training/erm.py` computes `val_avg_acc` as accuracy over `val_a` and `val_b` examples
combined (a single `ConcatDataset`) every epoch. Using only one half would make selection depend on
an arbitrary stratified split rather than the whole held-out validation set.
Alternatives: `val_a` only (rejected: throws away half the validation signal for no firewall
benefit, since selection isn't discovery or confirmation); last-epoch selection instead of
best-by-accuracy (noted in docs/PLAN.md M3 as a possible sensitivity check, not the default).

**D-033 · 2026-09-12 · `amp=true` on non-CUDA: warn, don't silently no-op, don't raise; record `precision_mode` in the manifest.**
Why: a shared `TrainConfig` (`amp: bool`) is meant to describe one experiment across both a local
MPS run and a Kaggle/Colab CUDA run without two separate config files -- but AMP is a CUDA-specific
optimisation (D-025: MPS branch is plain float32, no `GradScaler`). Silently ignoring `amp=true` on
MPS/CPU would be an invisible behaviour change; raising would force a config fork per device. The
human's call (this session): log a one-time warning *and* make the manifest itself state which
precision mode actually ran, so a report is never wrong about what happened even if nobody read the
log.
Decision: `manifest.py`'s `Manifest.precision_mode: Literal["fp16_amp", "fp32", "bf16_autocast"] |
None` records what actually ran; `training.erm.resolve_precision_mode(amp, device)` returns
`"fp16_amp"` only when `amp and device == "cuda"`, else `"fp32"` (with a `logging.warning` if `amp`
was requested but had no effect). `bf16_autocast` is reserved in the type for a possible future
MPS/CPU autocast path; nothing currently produces it.
Alternatives: silently no-op (rejected, human: invisible behaviour change); raise on a mismatched
device (rejected, human: forces config duplication per device).

**D-034 · 2026-09-12 · `Predictions`/`EmbeddingTable` dataclasses were not needed in M3 after all.**
Why: D-031 (M2) deferred adding these ARCHITECTURE §3-sketched dataclasses to `types.py` until M3,
on the expectation that `training/erm.py` and `embeddings/*.py` -- the modules that actually
produce predictions and embeddings -- would need them. In practice, every producer
(`training.erm._predict`, `embeddings.model_space.embed_model_space`,
`embeddings.clip_space.embed_clip_space`) and every consumer
(`evaluation.core.group_metrics_table`, `verification.reliance.compute_reliance`,
`embeddings.store`) works entirely on plain `pandas.DataFrame`s (predictions) or
`tuple[np.ndarray, list[str]]` (embeddings) -- matching the on-disk parquet/npy schemas directly,
with no intermediate object that added clarity or caught a bug the schema checks
(`embeddings.store.assert_aligned`, `evaluation.core`'s id-set check) didn't already catch.
Decision: do not add `Predictions`/`EmbeddingTable` to `types.py`. This is reported as a finding,
not silently dropped (CLAUDE.md §3 rule 6: investigate and report a plan that didn't pan out,
rather than forcing it through) -- flagged to the human in the M3 summary.
Revisit if: a later milestone (discovery/naming, M4-M5) finds itself repeatedly threading the same
5+ arrays together and would clearly benefit from a named bundle -- at that point, add the
dataclass where it is first actually needed, not preemptively.

**D-035 · 2026-09-12 · Exact training resume via a fresh per-epoch `DataLoader`, not RNG-state snapshotting.**
Why: `torch.utils.data.RandomSampler`'s shuffle order depends on its `Generator`'s *evolving*
state across successive `__iter__()` calls, not just its initial seed -- reusing one `DataLoader`
(and one `Generator`) across all epochs of a training run means epoch k's shuffle order depends on
every draw made in epochs `0..k-1`, so resuming from a checkpoint would require snapshotting and
restoring that generator's exact internal state, not just the model/optimizer.
Decision: `training/erm.py` builds a brand-new `DataLoader` every epoch, with a fresh
`torch.Generator` seeded by `seeding.make_rng(seed, "erm_shuffle", epoch)` -- deterministic in
`(seed, epoch)` alone. Resuming needs to restore only the model and optimizer state dicts (already
required for any resume); epoch k's shuffle order is reproduced automatically, with nothing RNG-
related to save. Verified in `tests/unit/test_erm.py::test_resume_from_checkpoint_matches_uninterrupted_run`.
Alternatives: snapshot/restore the `Generator`'s state in the checkpoint (rejected: more state to
get right, and diverges from `seeding.py`'s "explicit generator per purpose" convention rather than
extending it).

**D-036 · 2026-09-12 · `reliance.parquet`'s row layout: 4 per-condition rows + 1 `R_net` summary row.**
Why: ARCHITECTURE §4 names `reliance.parquet`'s columns (`direction`, `intervention`, `n`,
`break_rate`, `lo`, `hi`, "plus `R_net` summary row") but not the exact row scheme -- how many rows,
what values `direction`/`intervention` take for the summary row.
Decision: one row per `(direction, intervention)` in `{add, remove} x {real, null}` (4 rows, each
with its own single-condition bootstrap CI on that condition's break/no-break indicator), plus one
`direction="net", intervention="r_net"` row whose `break_rate` is `R_net` itself, with a CI from a
bootstrap that resamples the add-direction and remove-direction example sets independently each
draw (not four independent per-condition bootstraps combined post hoc, which would ignore that
`add_real`/`add_null` share the same underlying dogs and `remove_real`/`remove_null` share the same
cats).
Alternatives: report only the 4 condition rows and compute `R_net` downstream in `report/` --
rejected, since `R_net` (with its own CI) is the headline number `slens reliance`'s CLI output and
gate G3 need directly, not something every caller should have to re-derive.
