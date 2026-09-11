"""`run-jobs`: a sequential stage runner for GPU sessions, resumable after disconnection.

Concept: a GPU handoff runs a fixed list of stage commands (e.g. train then embed then reliance
for every seed of a config). Kaggle/Colab sessions disconnect; `run-jobs` must skip stages whose
manifest already matches and resume training from the last epoch checkpoint rather than starting
over (docs/RUNBOOK_GPU.md).

Pipeline position: shared. Entry point for `slens run-jobs jobs/<file>.yaml`.

Status: stub -- implemented in M3 (see docs/PLAN.md); extended for the sweep in M6.
"""
