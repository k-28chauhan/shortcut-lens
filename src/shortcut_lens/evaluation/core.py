"""Joins predictions with oracle groups; tidy group-metrics table with bootstrap CIs.

Concept: the shared join-and-score step underneath every other evaluation module -- getting the
join right (matching ids, not silently dropping rows) is exactly the kind of thing that produces
a plausible-looking wrong number if it is wrong. Works over plain DataFrames rather than the
`Predictions` dataclass sketched in docs/ARCHITECTURE.md §3, because nothing produces a real
`Predictions` until `training/erm.py` lands in M3 -- the dataclass is added to `types.py` then.

Pipeline position: oracle zone (FR-X1).
"""

from __future__ import annotations

from typing import cast

import pandas as pd

from shortcut_lens.seeding import make_rng
from shortcut_lens.stats import percentile_bootstrap

REQUIRED_PREDICTION_COLUMNS = {"example_id", "y", "y_hat"}
REQUIRED_ORACLE_COLUMNS = {"example_id", "group", "group_name"}


def join_predictions_with_groups(
    predictions: pd.DataFrame, oracle_groups: pd.DataFrame
) -> pd.DataFrame:
    """Inner-join `predictions` and `oracle_groups` on `example_id`.

    Requires the two frames to cover exactly the same `example_id`s -- `oracle_groups` must already
    be filtered to the same split as `predictions` by the caller. A mismatch (rather than silently
    keeping only the intersection) means a bug upstream, e.g. forgetting to filter by split, so it
    is raised rather than masked.
    """
    missing_predictions = REQUIRED_PREDICTION_COLUMNS - set(predictions.columns)
    if missing_predictions:
        raise ValueError(f"predictions missing required columns: {sorted(missing_predictions)}")
    missing_oracle = REQUIRED_ORACLE_COLUMNS - set(oracle_groups.columns)
    if missing_oracle:
        raise ValueError(f"oracle_groups missing required columns: {sorted(missing_oracle)}")

    prediction_ids = set(predictions["example_id"])
    oracle_ids = set(oracle_groups["example_id"])
    if prediction_ids != oracle_ids:
        only_in_predictions = prediction_ids - oracle_ids
        only_in_oracle = oracle_ids - prediction_ids
        raise ValueError(
            "predictions and oracle_groups do not cover the same example_ids: "
            f"{len(only_in_predictions)} only in predictions, "
            f"{len(only_in_oracle)} only in oracle_groups"
        )

    joined = predictions.merge(oracle_groups, on="example_id", how="inner", validate="one_to_one")
    return joined


def group_metrics_table(
    predictions: pd.DataFrame,
    oracle_groups: pd.DataFrame,
    seed: int,
    n_resamples: int = 1000,
) -> pd.DataFrame:
    """Tidy per-group accuracy table with 95% percentile bootstrap CIs.

    Columns: `group`, `group_name`, `n`, `accuracy`, `ci_lo`, `ci_hi`. Each group's bootstrap uses
    its own RNG stream (`make_rng(seed, "group_metrics_table", group_id)`) so groups' resamples
    never share a stream (`seeding.py`'s convention, CLAUDE.md §5).
    """
    joined = join_predictions_with_groups(predictions, oracle_groups)
    correct = (joined["y"] == joined["y_hat"]).to_numpy(dtype=float)
    joined = joined.assign(correct=correct)

    rows: list[dict[str, object]] = []
    for group_key, group_df in joined.groupby("group"):
        group_id = cast(int, group_key)
        rng = make_rng(seed, "group_metrics_table", group_id)
        point, ci_lo, ci_hi = percentile_bootstrap(
            group_df["correct"].to_numpy(), rng, n_resamples=n_resamples
        )
        rows.append(
            {
                "group": group_id,
                "group_name": group_df["group_name"].iloc[0],
                "n": len(group_df),
                "accuracy": point,
                "ci_lo": ci_lo,
                "ci_hi": ci_hi,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values("group").reset_index(drop=True)
