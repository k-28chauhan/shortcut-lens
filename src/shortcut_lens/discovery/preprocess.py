"""L2 normalisation and PCA, fitted on val_a only.

Concept: fitting preprocessing on val_a (never val_b or test) keeps confirmation and final
evaluation honest -- nothing about the projection is chosen using held-out data.

Pipeline position: label-free zone. Shared by every discovery method.

Status: stub -- implemented in M4 (see docs/PLAN.md).
"""
