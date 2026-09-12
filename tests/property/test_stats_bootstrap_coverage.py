"""Slow test: percentile bootstrap CI coverage over many simulated datasets.

See docs/PLAN.md M2 and docs/TESTING.md: coverage of a 95% CI should be close to 95% when checked
against many datasets simulated from a known true value.
"""

from __future__ import annotations

import pytest

from shortcut_lens.seeding import make_rng
from shortcut_lens.stats import percentile_bootstrap

N_SIMULATIONS = 500
SAMPLE_SIZE = 40
TRUE_P = 0.7


@pytest.mark.slow
def test_percentile_bootstrap_coverage_is_close_to_95_percent() -> None:
    data_rng = make_rng(0, "coverage_simulation_data")
    covered = 0

    for simulation_index in range(N_SIMULATIONS):
        sample = (data_rng.uniform(size=SAMPLE_SIZE) < TRUE_P).astype(float)
        bootstrap_rng = make_rng(0, "coverage_simulation_bootstrap", simulation_index)
        _point, ci_lo, ci_hi = percentile_bootstrap(sample, bootstrap_rng, n_resamples=500)
        if ci_lo <= TRUE_P <= ci_hi:
            covered += 1

    coverage = covered / N_SIMULATIONS
    assert 0.92 <= coverage <= 0.98, f"coverage {coverage:.3f} outside 95% +/- 3%"
