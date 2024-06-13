# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	mkdocs build

serve: install
	.venv/bin/mkdocs serve

install:
	uv venv
	uv pip install -r requirements.txt

# FUTURE move away from gh pages based deployment as it creates public content. But acceptable for now
deploy: install
	.venv/bin/mkdocs gh-deploy	