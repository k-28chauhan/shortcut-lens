"""Reproducible randomness: `set_all_seeds` for global state, `make_rng` for local generators.

Concept: two kinds of randomness need to be pinned down. `set_all_seeds` seeds the global RNGs
that third-party libraries (torch, sklearn, Python's own `random`) reach for implicitly, so a
whole run is repeatable. `make_rng` gives *our* code an explicit, local `numpy.random.Generator`
derived deterministically from a seed plus arbitrary context keys (e.g. an example id) -- this
project's own code never reads the legacy global numpy RNG or holds a seed in a hidden global
(CLAUDE.md §5). The same `(seed, *keys)` always produces the same generator state, on any machine.

Pipeline position: shared, pure. `set_all_seeds` is called once per run by `cli.py`; `make_rng` is
called wherever a stage needs its own generator (e.g. `build/planting.py` derives one per image
from `(build_seed, example_id)`, matching docs/ARCHITECTURE.md §8).
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """Seed Python's `random`, numpy's legacy global RNG, and torch (CPU and CUDA if available).

    Call once per run, before any model, dataloader or third-party code that might draw from
    these global RNGs. Does not affect this project's own `numpy.random.Generator` instances,
    which are seeded explicitly via `make_rng`.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_rng(seed: int, *keys: int | str) -> np.random.Generator:
    """Return a `numpy.random.Generator` deterministic in `(seed, *keys)`.

    `keys` let one base seed fan out into many independent, reproducible streams -- e.g. a
    different stream per example id, per split, or per discovery method -- without ever reusing
    the same stream for two different purposes. Uses a cryptographic hash (not Python's built-in
    `hash()`, which is salted per-process and would break reproducibility across runs).
    """
    material = ":".join([str(seed), *(str(key) for key in keys)]).encode("utf-8")
    digest = hashlib.sha256(material).digest()
    seed_int = int.from_bytes(digest[:8], byteorder="big")
    return np.random.default_rng(seed_int)
