# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	mkdocs build

serve:
	mkdocs serve

install:
	pip install -r requirements.txt