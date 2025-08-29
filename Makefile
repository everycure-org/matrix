PREREQUISITES := docker gcloud python3 java uv kubectl

setup: prerequisites precommit-hooks

prerequisites:
	$(info Checking if prerequisites are installed...)
	$(foreach exec,$(PREREQUISITES),\
        $(if $(shell command -v $(exec) 2>/dev/null),,$(error "$(exec) is not installed.")))
	$(info All prerequisites are installed.)
	$(info Checking gcloud auth...)
	@if ! gcloud auth print-access-token >/dev/null 2>&1; then \
		echo "Not logged into gcloud. Please run 'gcloud auth login' first."; \
		exit 1; \
	fi


precommit: precommit-hooks
	uv sync --group dev
	git fetch origin
	uv run pre-commit run --from-ref origin/main --to-ref HEAD

precommit-hooks:
	uv run pre-commit install --install-hooks

format:
	uv run ruff check . --fix