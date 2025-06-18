# Define the image name
IMAGE_NAME = kg-dashboard
REPO_NAME = matrix-images
EVIDENCE_BASE_PATH = gs://data.dev.everycure.org/versions


# controls the project id, useful for BQ queries
PROJECT_ID ?= $(shell gcloud config get-value project)
export EVIDENCE_VAR__project_id := $(PROJECT_ID)

# controls the release version of the KG used in the dashboard
RELEASE_VERSION ?= v0.5.3
export EVIDENCE_VAR__release_version := $(RELEASE_VERSION)
export EVIDENCE_VAR__bq_release_version := $(subst .,_,$(RELEASE_VERSION))
export VITE_release_version := $(RELEASE_VERSION)

export VITE_build_time := $(shell date -u '+%B %d, %Y at %H:%M:%S UTC')

export EVIDENCE_VAR__robokop_version=30fd1bfc18cd5ccb
export EVIDENCE_VAR__rtx_kg2_version=v2_10_0_validated

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

run-sources:
	npm run sources

run-dev:
	npm run dev

clean:
	rm -rf build
	rm -rf node_modules
	rm -rf .evidence
