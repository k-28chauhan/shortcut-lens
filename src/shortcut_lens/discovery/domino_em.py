"""`domino_em`: error-aware diagonal Gaussian mixture, fitted from scratch with EM.

Concept: a per-class simplification of Domino (Eyuboglu et al. 2022) -- each mixture component
also models the probability of being misclassified via a Bernoulli term with weight `w`, so
clustering is pulled toward groups that are both representationally coherent *and* error-prone.
E-step/M-step formulas are in docs/ARCHITECTURE.md §8. At `w=0` this must match sklearn's diagonal
GMM exactly (the test that proves the from-scratch implementation is correct).

Pipeline position: label-free zone (FR-S3).

Status: stub -- implemented in M4 (see docs/PLAN.md).
"""
