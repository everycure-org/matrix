precommit: precommit-hooks
	uv sync --group dev
	git fetch origin
	uv run pre-commit run --from-ref origin/main --to-ref HEAD

precommit-hooks:
	uv run pre-commit install --install-hooks

format:
	uv run ruff check . --fix