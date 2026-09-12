"""`R_net`: how much the model's predictions actually change when the shortcut is added/removed.

Concept: correlation strength (rho) is a property of the *data*; reliance is a property of the
*model* -- a model can fail to learn even a strong shortcut. Plotting discovery quality against
measured reliance (D-007), not rho, is what separates "the model didn't learn it" from "the tool
didn't find it" (FR-R1).

FR-R1's two directions and their null controls isolate reliance on the patch's *colour* from mere
reliance on *something being painted there* (the occlusion confound, D-008):
- **add**: dogs without a patch, correctly classified -> add the real patch (`break_add_real`) vs
  add the null/grey patch (`break_add_null`). `net_add = break_add_real - break_add_null`.
- **remove**: cats with a patch, correctly classified -> remove to the clean base
  (`break_remove_real`) vs swap the real patch for the null/grey one at the same position
  (`break_remove_null`). `net_remove = break_remove_real - break_remove_null`.
`R_net` is the mean of the two net rates. Every cached base image is already the clean,
unpatched file (`build/datasets.py`'s convention) -- both directions load the same clean file and
apply `build/planting`'s functions directly, since this module is oracle zone.

Pipeline position: oracle zone. Entry point for `slens reliance`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, cast

import numpy as np
import pandas as pd
import torch
from PIL import Image

from shortcut_lens.build.planting import PatchSpec, add_patch, null_patch
from shortcut_lens.data.transforms import eval_transform
from shortcut_lens.models.backbones import Backbone
from shortcut_lens.seeding import make_rng
from shortcut_lens.stats import percentile_bootstrap

_Mode = Literal["clean", "add_real", "add_null"]


def _render(
    image_root: Path,
    image_ref: str,
    example_id: str,
    build_seed: int,
    spec: PatchSpec,
    mode: _Mode,
) -> torch.Tensor:
    """Load the clean cached image and apply one counterfactual rendering, then eval-transform."""
    image = Image.open(image_root / image_ref).convert("RGB")
    if mode == "add_real":
        image = add_patch(image, spec, example_id, build_seed)
    elif mode == "add_null":
        image = null_patch(image, spec, example_id, build_seed)
    return eval_transform()(image)  # type: ignore[no-any-return]


def _predict_correct(
    model: Backbone, device: str, images: torch.Tensor, y: np.ndarray, batch_size: int
) -> np.ndarray:
    """Boolean array: whether each example is correctly classified under `model`."""
    model.eval()
    y_hats = []
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = images[start : start + batch_size].to(device)
            logits = model(batch).cpu()
            y_hats.append(logits.argmax(dim=1))
    y_hat = torch.cat(y_hats).numpy()
    return cast(np.ndarray, y_hat == y)


def _render_condition(
    subset: pd.DataFrame, image_root: Path, spec: PatchSpec, build_seed: int, mode: _Mode
) -> torch.Tensor:
    images = [
        _render(image_root, str(image_ref), str(example_id), build_seed, spec, mode)
        for image_ref, example_id in zip(subset["image_ref"], subset["example_id"], strict=True)
    ]
    return torch.stack(images)


def compute_reliance(
    model: Backbone,
    device: str,
    joined_table: pd.DataFrame,
    image_root: Path,
    patch_spec: PatchSpec,
    build_seed: int,
    seed: int,
    batch_size: int = 32,
    n_resamples: int = 1000,
) -> pd.DataFrame:
    """`reliance.parquet`: `direction`, `intervention`, `n`, `break_rate`, `lo`, `hi`, plus one
    `direction="net", intervention="r_net"` summary row (ARCHITECTURE §4's schema names the
    columns but not their exact row layout; this row scheme is D-036).

    `joined_table` must have one row per example with columns `example_id`, `image_ref`, `y`,
    `class_name`, `has_patch`, and `correct` (its *baseline* -- undisturbed -- prediction
    correctness, e.g. from `predictions/{split}.parquet`).
    """
    dogs_no_patch = joined_table[
        (joined_table["class_name"] == "dog") & ~joined_table["has_patch"] & joined_table["correct"]
    ]
    cats_with_patch = joined_table[
        (joined_table["class_name"] == "cat") & joined_table["has_patch"] & joined_table["correct"]
    ]
    groups = (("dogs without a patch", dogs_no_patch), ("cats with a patch", cats_with_patch))
    for name, subset in groups:
        if len(subset) == 0:
            raise ValueError(
                f"no correctly-classified {name} in joined_table -- cannot measure R_net"
            )

    add_real = _predict_correct(
        model,
        device,
        _render_condition(dogs_no_patch, image_root, patch_spec, build_seed, "add_real"),
        dogs_no_patch["y"].to_numpy(),
        batch_size,
    )
    add_null = _predict_correct(
        model,
        device,
        _render_condition(dogs_no_patch, image_root, patch_spec, build_seed, "add_null"),
        dogs_no_patch["y"].to_numpy(),
        batch_size,
    )
    remove_real = _predict_correct(
        model,
        device,
        _render_condition(cats_with_patch, image_root, patch_spec, build_seed, "clean"),
        cats_with_patch["y"].to_numpy(),
        batch_size,
    )
    remove_null = _predict_correct(
        model,
        device,
        _render_condition(cats_with_patch, image_root, patch_spec, build_seed, "add_null"),
        cats_with_patch["y"].to_numpy(),
        batch_size,
    )

    rows: list[dict[str, object]] = []
    conditions = (
        ("add", "real", add_real, make_rng(seed, "reliance", "add", "real")),
        ("add", "null", add_null, make_rng(seed, "reliance", "add", "null")),
        ("remove", "real", remove_real, make_rng(seed, "reliance", "remove", "real")),
        ("remove", "null", remove_null, make_rng(seed, "reliance", "remove", "null")),
    )
    for direction, intervention, still_correct, rng in conditions:
        broke = (~still_correct).astype(float)
        point, lo, hi = percentile_bootstrap(broke, rng, n_resamples=n_resamples)
        rows.append(
            {
                "direction": direction,
                "intervention": intervention,
                "n": len(broke),
                "break_rate": point,
                "lo": lo,
                "hi": hi,
            }
        )

    n_add, n_remove = len(dogs_no_patch), len(cats_with_patch)
    rng = make_rng(seed, "reliance", "r_net")
    resamples = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx_add = rng.integers(0, n_add, size=n_add)
        idx_remove = rng.integers(0, n_remove, size=n_remove)
        net_add = float((~add_real[idx_add]).mean() - (~add_null[idx_add]).mean())
        net_remove = float((~remove_real[idx_remove]).mean() - (~remove_null[idx_remove]).mean())
        resamples[i] = (net_add + net_remove) / 2

    net_add_point = (~add_real).mean() - (~add_null).mean()
    net_remove_point = (~remove_real).mean() - (~remove_null).mean()
    r_net_point = (net_add_point + net_remove_point) / 2
    r_net_lo, r_net_hi = np.quantile(resamples, [0.025, 0.975])
    rows.append(
        {
            "direction": "net",
            "intervention": "r_net",
            "n": n_add + n_remove,
            "break_rate": float(r_net_point),
            "lo": float(r_net_lo),
            "hi": float(r_net_hi),
        }
    )
    return pd.DataFrame.from_records(rows)
