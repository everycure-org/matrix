# Makefile that builds or serves the docs.
KEEP_VERSIONS := 10

.PHONY: build serve

venv:
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
	uv run python -m ensurepip --upgrade || true

build: install
	.venv/bin/mkdocs build -s # build in strict mode, all links must function

serve: 
	.venv/bin/mkdocs serve

install:
	uv sync

test: install
	uv run pytest

deploy: build
	gcloud app deploy

clean_versions:
	@VERSIONS=$$(gcloud app versions list --service=default --format="value(version.id)" --sort-by="~version.createTime"); \
	DELETE_VERSIONS=$$(($$(echo $$VERSIONS | wc -w) - $(KEEP_VERSIONS))); \
	if [ "$$DELETE_VERSIONS" -le 0 ]; then \
		echo "No versions to delete"; \
	else \
		OLD_VERSIONS=$$(echo $$VERSIONS | cut -d' ' -f$$(($(KEEP_VERSIONS)+1))-); \
		gcloud app versions delete $$OLD_VERSIONS --service=default --quiet; \
	fi
