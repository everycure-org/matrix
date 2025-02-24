export UV_PYTHON = .venv/bin/python

# Makefile that builds or serves the docs.
.PHONY: build serve

.ONESHELL:
venv:
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
	uv run python -m ensurepip --upgrade || true

build: install
	uv run mkdocs build -s # build in strict mode, all links must function

serve: 
	uv run mkdocs serve

install:
	uv pip install -r requirements.txt

test: install
	uv run pytest

deploy: build
	gcloud app deploy

lock:
	uv pip compile requirements.in > requirements.txt