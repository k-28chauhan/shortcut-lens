"""`RenderedImageDataset`: the only `ImageDataset` implementation, images rendered on the fly.

Concept: images are rendered from a cached base file at `__getitem__` time, optionally with a
patch painted on top, rather than baking every (patch on/off) variant into separate files -- this
keeps a full ablation sweep over patch size/rho cheap, since the same base JPEG cache serves every
value. Every instance returns exactly `{"image", "y", "example_id"}`, regardless of which real
dataset it wraps, which is what makes it safe to hand to label-free code (D-003). Whether a given
example has the patch (`patch_flags`) is plain data computed once, in the oracle zone, from
`oracle.parquet`'s `has_patch` column -- this class itself never reads `oracle.parquet` or imports
`oracle.groups`; the composition root (`cli.py`) is the one place that connects the two.

Pipeline position: the seam between the oracle zone and the label-free zone. Built in `build/`,
injected into label-free stages by `cli.py`.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

from shortcut_lens.build.planting import PatchSpec, add_patch


class RenderedImageDataset:
    """Renders `{"image", "y", "example_id"}` from a public table + an image cache directory.

    If `patch_spec` is given, examples whose `example_id` is `True` in `patch_flags` have the
    patch painted on at read time; all others are served straight from `image_root`. Passing
    `patch_spec=None` (the synthetic_shapes and waterbirds case) makes this a pure file reader.
    """

    def __init__(
        self,
        public_table: pd.DataFrame,
        image_root: str | Path,
        *,
        patch_spec: PatchSpec | None = None,
        patch_flags: dict[str, bool] | None = None,
        build_seed: int = 0,
        transform: Callable[[Image.Image], Any] | None = None,
    ) -> None:
        if patch_spec is not None and patch_flags is None:
            raise ValueError("patch_flags is required when patch_spec is given")
        self._rows = public_table.reset_index(drop=True)
        self._image_root = Path(image_root)
        self._patch_spec = patch_spec
        self._patch_flags = patch_flags or {}
        self._build_seed = build_seed
        self._transform = transform

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self._rows.iloc[index]
        example_id = str(row["example_id"])
        image = Image.open(self._image_root / row["image_ref"]).convert("RGB")

        if self._patch_spec is not None and self._patch_flags.get(example_id, False):
            image = add_patch(image, self._patch_spec, example_id, self._build_seed)

        rendered: Any = self._transform(image) if self._transform is not None else image
        return {"image": rendered, "y": int(row["y"]), "example_id": example_id}
