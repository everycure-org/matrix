# Makefile that builds or serves the docs.
KEEP_VERSIONS := 100

.PHONY: build serve

venv:
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
	uv run python -m ensurepip --upgrade || true

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

clean_versions:
	mapfile -t VERSIONS < <(gcloud app versions list --service=default --format="value(version.id)" --sort-by="~version.createTime")
	KEEP_VERSIONS=100
	DELETE_VERSIONS=$((${#VERSIONS[@]} - KEEP_VERSIONS))
	if [ "$DELETE_VERSIONS" -le 0 ]; then
		echo "No versions to delete"
	else
		OLD_VERSIONS=("${VERSIONS[@]:$KEEP_VERSIONS}")
		echo "Deleting ${#OLD_VERSIONS[@]} old versions"
		# gcloud app versions delete "${OLD_VERSIONS[@]}" --service=default --quiet
	fi

delete-old-versions:
	@VERSIONS=$$(gcloud app versions list --service=default --format="value(version.id)" --sort-by="~version.createTime"); \
	DELETE_VERSIONS=$$(($$(echo $$VERSIONS | wc -w) - $(KEEP_VERSIONS))); \
	if [ "$$DELETE_VERSIONS" -le 0 ]; then \
		echo "No versions to delete"; \
	else \
		OLD_VERSIONS=$$(echo $$VERSIONS | cut -d' ' -f$(KEEP_VERSIONS)-); \
		echo "Deleting versions: $$OLD_VERSIONS"; \
		echo "Deleting $$(echo $$OLD_VERSIONS | wc -w) old versions"; \
	fi
