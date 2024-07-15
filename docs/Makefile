# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	.venv/bin/mkdocs build -s # build in strict mode, all links must function

serve: install
	.venv/bin/mkdocs serve

install:
	uv venv
	uv pip install -r requirements.txt