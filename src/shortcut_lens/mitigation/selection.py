"""Label-free hyperparameter selection for mitigation methods.

Concept: selects by worst *pseudo*-group accuracy on val_b (class x confirmed-slice membership),
tie-break by average accuracy -- never true worst-group accuracy, which would require group
labels (FR-M5). A runtime assertion checks the data it receives has no group columns at all.

Pipeline position: label-free zone (imports nothing from the oracle zone).

Status: stub -- implemented in M7 (see docs/PLAN.md).
"""
