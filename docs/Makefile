# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	mkdocs build

serve: install
	.venv/bin/mkdocs serve

install:
	uv venv
	uv pip install -r requirements.txt