"""`ll_balanced`: class-balanced last-layer logistic regression. The mitigation control.

Concept: balances only by class, not by any notion of group -- if this alone recovers most of the
worst-group accuracy gain (as expected in balanced validation mode, H7a), that tells us the
balanced data is doing the work, not the sophistication of the method.

Pipeline position: label-free zone (FR-M1).

Status: stub -- implemented in M7 (see docs/PLAN.md).
"""
