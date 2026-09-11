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
