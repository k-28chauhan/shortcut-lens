"""Label-free selection of the (method, space) combination fed to naming/verification/mitigation.

Concept: gate G4 deliberately checks the *best* of all method x space combinations against
ground truth as a sanity check -- but using that oracle comparison to choose what downstream
stages consume would leak group labels into a label-free pipeline. This module makes the same
choice using only val_b statistics: the combination whose top confirmed slice has the largest
error-rate lift over the rest of its class, tying-broken by smallest q-value (FR-C4, D-024).

Pipeline position: label-free zone. Runs after `discovery/confirm.py`, before naming (M5),
verification (M6) and `mitigation/label_free/dfr_discovered.py` (M7).

Status: stub -- implemented in M4 (see docs/PLAN.md).
"""
