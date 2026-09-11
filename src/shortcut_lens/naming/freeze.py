"""Verifies `vocab/FROZEN.sha256` before any naming run; refuses on mismatch.

Concept: if the phrase list could be edited after seeing which phrases "work", naming results
would mean nothing (D-012). This check makes that mistake impossible by default; overriding it
requires `--unfreeze` plus a DECISIONS entry.

Pipeline position: label-free zone. Runs before every `slens name` invocation.

Status: stub -- implemented in M5 (see docs/PLAN.md).
"""
