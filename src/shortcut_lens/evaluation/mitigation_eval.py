"""Mitigation metrics: average, WGA, recovery, cost of labels (PRD §12 definitions).

Concept: `recovery = (WGA_method - WGA_ERM) / (WGA_dfr_oracle - WGA_ERM)`, NaN if the denominator
is under 2 points (a guard against reporting a meaningless ratio when ERM was already fine).

Pipeline position: oracle zone.

Status: stub -- implemented in M7 (see docs/PLAN.md).
"""
