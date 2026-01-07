.PHONY: lock install precommit precommit_hooks check_release_folder_is_empty

DRUG_LIST_RELEASE_VERSION ?= v1.1.0
DISEASE_LIST_RELEASE_VERSION ?= v0.2.4

PREREQUISITES := python3 uv

prerequisites:
	$(info Checking if prerequisites are installed...)
	$(foreach exec,$(PREREQUISITES),\
        $(if $(shell command -v $(exec) 2>/dev/null),,$(error "$(exec) is not installed.")))
	$(info All prerequisites are installed.)

clean:
	uv venv --clear

install: precommit_hooks
	uv sync

precommit: precommit_hooks
	git fetch origin
	uv run pre-commit run --from-ref origin/main --to-ref HEAD

precommit_hooks:
	uv run pre-commit install --install-hooks

integration_test:
	uv run kedro run -e test

test:
	# activate venv to ensure spark doesn't have python driver mismatches
	uv run pytest -v tests/

test_llm:
	uv run kedro run -e llm_test --pipeline=disease_categories
	uv run kedro run -e llm_test --pipeline=disease_prevalence
	uv run kedro run -e llm_test --pipeline=disease_umn
	uv run kedro run -e llm_test --pipeline=disease_labels

check_release_folder_is_empty:
	@echo "Checking if release folder is empty..."
	@if gsutil ls gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/$(PIPELINE_NAME)/$(RELEASE_VERSION) >/dev/null 2>&1; then \
		echo "❌ Error: Release folder gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/$(PIPELINE_NAME)/$(RELEASE_VERSION) is not empty"; \
		echo "Please empty the release folder first"; \
		exit 1; \
	fi
	@echo "✅ Release folder is empty, proceeding with release..."

release_checks:
	@if [ -n "$$(git status --porcelain)" ]; then \
		echo "❌ Error: You have uncommitted changes. Please commit or stash them first."; \
		exit 1; \
	fi
	@if [ "$$(git branch --show-current)" != "main" ]; then \
		echo "❌ Error: You must be on the main branch to publish a release."; \
		exit 1; \
	fi

release_drug_list: release_checks
	make check_release_folder_is_empty PIPELINE_NAME=drug_list RELEASE_VERSION=$(DRUG_LIST_RELEASE_VERSION)
	git tag $(DRUG_LIST_RELEASE_VERSION)-drug_list
	git push origin $(DRUG_LIST_RELEASE_VERSION)-drug_list
	PIPELINE_NAME=drug_list RELEASE_VERSION=$(DRUG_LIST_RELEASE_VERSION) uv run kedro run -e cloud --pipeline=drug_list
	gh workflow run core_entities_create_release_pr.yaml -f release_version="$(DRUG_LIST_RELEASE_VERSION)"  -f pipeline="drug_list"

publish_drug_list_release:
	gsutil cp gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/$(DRUG_LIST_RELEASE_VERSION)/03_primary/release/ec-drug-list.tsv gs://data.dev.everycure.org/data/01_RAW/drug_list/$(DRUG_LIST_RELEASE_VERSION)/ec-drug-list.tsv
	gsutil cp gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/$(DRUG_LIST_RELEASE_VERSION)/03_primary/release/ec-drug-list.parquet gs://data.dev.everycure.org/data/01_RAW/drug_list/$(DRUG_LIST_RELEASE_VERSION)/ec-drug-list.parquet
	uv run python scripts/create_bq_table.py --project-id ec-orchard-prod --dataset drug_list --table-name ec_drug_list_$(DRUG_LIST_RELEASE_VERSION_UNDERSCORE) --gs-uri gs://data.dev.everycure.org/data/01_RAW/drug_list/$(DRUG_LIST_RELEASE_VERSION)/ec-drug-list.parquet --format PARQUET

release_disease_list: release_checks
	make check_release_folder_is_empty PIPELINE_NAME=disease_list RELEASE_VERSION=$(DISEASE_LIST_RELEASE_VERSION)
	git tag $(DISEASE_LIST_RELEASE_VERSION)-disease_list
	git push origin $(DISEASE_LIST_RELEASE_VERSION)-disease_list
	PIPELINE_NAME=disease_list RELEASE_VERSION=$(DISEASE_LIST_RELEASE_VERSION) uv run kedro run -e cloud --pipeline=disease_list
	gh workflow run core_entities_create_release_pr.yaml -f release_version="$(DISEASE_LIST_RELEASE_VERSION)"  -f pipeline="disease_list"

publish_disease_list_release:
	gsutil cp gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/$(DISEASE_LIST_RELEASE_VERSION)/03_primary/release/ec-disease-list.tsv gs://data.dev.everycure.org/data/01_RAW/disease_list/$(DISEASE_LIST_RELEASE_VERSION)/ec-disease-list.tsv
	gsutil cp gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/$(DISEASE_LIST_RELEASE_VERSION)/03_primary/release/ec-disease-list.parquet gs://data.dev.everycure.org/data/01_RAW/disease_list/$(DISEASE_LIST_RELEASE_VERSION)/ec-disease-list.parquet
	uv run python scripts/create_bq_table.py --project-id ec-orchard-prod --dataset disease_list --table-name ec_disease_list_$(DISEASE_LIST_RELEASE_VERSION_UNDERSCORE) --gs-uri gs://data.dev.everycure.org/data/01_RAW/disease_list/$(DISEASE_LIST_RELEASE_VERSION)/ec-disease-list.parquet --format PARQUET