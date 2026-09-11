.PHONY: setup check test test-slow smoke reproduce-cpu app

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
	@echo "'make test-slow' is not implemented yet -- lands starting M1 (see docs/PLAN.md)." >&2
	@exit 1

smoke:
	@echo "'make smoke' is not implemented yet -- lands in M3 (see docs/PLAN.md)." >&2
	@exit 1

reproduce-cpu:
	@echo "'make reproduce-cpu' is not implemented yet -- lands in M8 (see docs/PLAN.md)." >&2
	@exit 1

app:
	@echo "'make app' is not implemented yet -- lands in M8 (see docs/PLAN.md)." >&2
	@exit 1
