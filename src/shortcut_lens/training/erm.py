"""Empirical risk minimisation (ordinary fine-tuning) with AMP, checkpointing and exact resume.

Concept: ERM just minimises average training loss -- it is expected to learn shortcuts when they
correlate with the label, which is exactly what this project measures. Checkpoint selection uses
average validation accuracy only, never worst-group accuracy, because the latter would require
group labels (D-004, the "no group labels for any choice" integrity rule).

Pipeline position: label-free zone. Entry point for `slens train`.

Status: stub -- implemented in M3 (see docs/PLAN.md).
"""
