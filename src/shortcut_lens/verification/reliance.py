"""`R_net`: how much the model's predictions actually change when the shortcut is added/removed.

Concept: correlation strength (rho) is a property of the *data*; reliance is a property of the
*model* -- a model can fail to learn even a strong shortcut. Plotting discovery quality against
measured reliance (D-007), not rho, is what separates "the model didn't learn it" from "the tool
didn't find it" (FR-R1).

Pipeline position: oracle zone. Entry point for `slens reliance`.

Status: stub -- implemented in M3 (see docs/PLAN.md).
"""
