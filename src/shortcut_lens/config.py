"""Config loading and hashing: YAML composition, Pydantic validation, canonical config hash.

Concept: every stage's config is a Pydantic model loaded from YAML. Experiment configs compose
smaller ones via a `base:` key (see docs/ARCHITECTURE.md §6). The config hash is the sha256 of the
canonical JSON of the fully resolved config (sorted keys) -- stable regardless of key order or
dict insertion order, so two runs with logically identical config always produce the same hash.
Idempotency (skip a stage if its output already exists) and run ids both depend on that.

Pipeline position: shared, pure. Used by `cli.py` before every stage.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, TypeVar

import yaml
from pydantic import BaseModel, ConfigDict

ModelT = TypeVar("ModelT", bound=BaseModel)


class StrictBaseModel(BaseModel):
    """Base class for every config model: rejects unknown fields so typos in YAML fail loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


def load_yaml_composed(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, recursively merging any `base:` file it points to.

    The file's own keys win over its base's keys, applied recursively for nested dicts (a
    `train: {lr: ...}` override does not wipe out the rest of `train` from the base). `base` may
    itself point to another `base`; composition applies the same way at every level.
    """
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: top-level YAML must be a mapping, got {type(raw).__name__}")

    base_ref = raw.pop("base", None)
    if base_ref is None:
        return raw

    base_path = (path.parent / base_ref).resolve()
    base_config = load_yaml_composed(base_path)
    return _deep_merge(base_config, raw)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge `override` onto `base`.

    Nested dicts merge key-by-key (an override does not wipe out sibling keys); any other value
    type simply replaces the base's value.
    """
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def canonical_json(data: dict[str, Any]) -> str:
    """Serialise `data` to JSON with sorted keys and no extra whitespace -- the hash input."""
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(data: dict[str, Any]) -> str:
    """Full sha256 hex digest of the canonical JSON of `data`. Stable across key order and runs."""
    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def short_hash(data: dict[str, Any], length: int = 8) -> str:
    """First `length` hex characters of `config_hash`, used in run ids (docs/ARCHITECTURE.md §5)."""
    return config_hash(data)[:length]


def load_config(path: str | Path, model: type[ModelT]) -> tuple[ModelT, str]:
    """Load, compose and validate a YAML config against `model`; return it with its config hash.

    The hash is computed from the *resolved* dict (after composition, before Pydantic fills in
    defaults), so the hash does not silently change if `model`'s defaults change later, and two
    configs that resolve to the same dict always hash the same way regardless of key order.
    """
    data = load_yaml_composed(path)
    validated = model.model_validate(data)
    return validated, config_hash(data)
