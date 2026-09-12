.PHONY: setup check test test-slow smoke reproduce-cpu app

# PYTORCH_ENABLE_MPS_FALLBACK=1 must be exported whenever `slens train`/`slens embed` run locally
# with device=mps (D-025) -- an op with no MPS kernel then falls back to CPU instead of raising.
# None of the targets below need it: `check`/`test` never train or embed, and `smoke` forces
# device=cpu by design (docs/PLAN.md M3) since MPS is not bit-stable enough for its exact-resume
# check. Local training/embedding is invoked directly (`uv run slens train --config ...`), not
# through a Make target -- set the variable there once M3 lands, not here.

setup:
	uv sync --extra cpu
	uv run pre-commit install

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/shortcut_lens
	uv run lint-imports
	uv run pytest -m "not slow and not gpu and not network"

test:
	uv run pytest -m "not slow and not gpu and not network"

test-slow:
	uv run pytest -m "slow or network"

smoke:
	uv run pytest tests/integration/test_smoke.py -v

reproduce-cpu:
	@echo "'make reproduce-cpu' is not implemented yet -- lands in M8 (see docs/PLAN.md)." >&2
	@exit 1

app:
	@echo "'make app' is not implemented yet -- lands in M8 (see docs/PLAN.md)." >&2
	@exit 1
