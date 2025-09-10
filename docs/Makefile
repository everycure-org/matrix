# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	.venv/bin/mkdocs build -s # build in strict mode, all links must function

serve: 
	.venv/bin/mkdocs serve

install:
	deactivate 2> /dev/null || echo 'no venv active, continuing'
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
	uv pip install -r requirements.txt

test: install
	uv run pytest

deploy: build
	gcloud app deploy