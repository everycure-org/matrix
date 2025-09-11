# Change project_id, used for BQ queries
PROJECT_ID ?= $(shell gcloud config get-value project)
export EVIDENCE_VAR__project_id := $(PROJECT_ID)

# Change GCS for deployment path
EC_DATA_DOMAIN ?= data.dev.everycure.org
EVIDENCE_BASE_PATH = gs://$(EC_DATA_DOMAIN)/versions

# Change release version of the KG used in the dashboard

RELEASE_VERSION ?= v0.9.9
BENCHMARK_VERSION ?= v0.4.5


export EVIDENCE_VAR__release_version := $(RELEASE_VERSION)
export EVIDENCE_VAR__bq_release_version := $(subst .,_,$(RELEASE_VERSION))
export VITE_release_version := $(RELEASE_VERSION)

# Build time to show when dashboard was last updated
export VITE_build_time := $(shell date -u '+%B %d, %Y at %H:%M:%S UTC')

# TODO: refactor BQ ingested tables to remove versions
# Source KG versions, extracted from globals.yml
KEDRO_BASE_GLOBALS_PATH = "../../pipelines/matrix/conf/base/globals.yml"
ROBOKOP_VERSION = $(shell grep -A1 "robokop:" $(KEDRO_BASE_GLOBALS_PATH) | grep "version:" | sed 's/.*version: //')
RTX_KG2_VERSION = $(shell grep -A1 "rtx_kg2:" $(KEDRO_BASE_GLOBALS_PATH) | grep "version:" | sed 's/.*version: //' | sed 's/^&_rtx_kg_version //' | sed -E 's/\./_/g')

export EVIDENCE_VAR__robokop_version := $(ROBOKOP_VERSION)
export EVIDENCE_VAR__rtx_kg2_version := $(RTX_KG2_VERSION)
export EVIDENCE_VAR__benchmark_version := $(BENCHMARK_VERSION)
export VITE_robokop_version := $(ROBOKOP_VERSION)
export VITE_rtx_kg2_version := $(RTX_KG2_VERSION)
export VITE_benchmark_version := $(BENCHMARK_VERSION)

MAX_CORE_ENTITIES_NORMALIZATION_ERRORS = 1000
export EVIDENCE_VAR__max_core_entities_normalization_errors := $(MAX_CORE_ENTITIES_NORMALIZATION_ERRORS)
export VITE_max_core_entities_normalization_errors := $(MAX_CORE_ENTITIES_NORMALIZATION_ERRORS)

MAX_EDGE_FAILED_NORMALIZATION_BY_NORMALIZATION_SET_PREFIX = 100
export EVIDENCE_VAR__max_edge_failed_normalization_by_normalization_set_prefix := $(MAX_EDGE_FAILED_NORMALIZATION_BY_NORMALIZATION_SET_PREFIX)

.ONESHELL:

.PHONY: clean install bump_basepath run-dev

bump_basepath:
	node ./scripts/update-basepath.cjs $(RELEASE_VERSION)

build: bump_basepath 
	@echo "Building with EVIDENCE_VAR__release_version=$(EVIDENCE_VAR__release_version)"
	npm run build:strict

deploy: 
	# both rsyncs are set up to delete extra files. We use gcloud storage over gsutil because gsutil does not use github action authentication
	gcloud storage rsync --recursive --delete-unmatched-destination-objects build/ $(EVIDENCE_BASE_PATH)/$(RELEASE_VERSION)/evidence
	gcloud storage rsync --recursive --delete-unmatched-destination-objects $(EVIDENCE_BASE_PATH)/$(RELEASE_VERSION)/evidence $(EVIDENCE_BASE_PATH)/latest/evidence

install:
	npm install

populate-infores-source:
	curl -sL "https://github.com/biolink/information-resource-registry/releases/latest/download/infores_catalog.jsonl" -o scripts/temp_infores_catalog.jsonl
	node ./scripts/load-infores.cjs
	rm -f scripts/temp_infores_catalog.jsonl

populate-valid-edge-types-source:
	curl -sL "https://github.com/everycure-org/matrix-schema/releases/latest/download/valid_biolink_edge_types.tsv" -o scripts/temp_valid_edge_types.tsv
	node ./scripts/load-valid-edge-types.cjs
	rm -f scripts/temp_valid_edge_types.tsv

run-sources: populate-infores-source populate-valid-edge-types-source
	npm run sources

run-dev:
	echo "BENCHMARK_VERSION=$(BENCHMARK_VERSION)"
	npm run dev

clean:
	rm -rf build
	rm -rf node_modules
	rm -rf .evidence
