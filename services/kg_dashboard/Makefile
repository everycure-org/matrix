# Define the image name
IMAGE_NAME = kg-dashboard
REPO_NAME = matrix-images
PROJECT_ID = $(shell gcloud config get-value project)
EVIDENCE_BASE_PATH = gs://data.dev.everycure.org/versions

# controls the release version of the KG used in the dashboard
RELEASE_VERSION ?= v0.3.0
export EVIDENCE_VAR__release_version := $(RELEASE_VERSION)
export EVIDENCE_VAR__bq_release_version := $(subst .,_,$(RELEASE_VERSION))
export VITE_release_version := $(RELEASE_VERSION)

.ONESHELL:

.PHONY: clean install bump_basepath run-dev

bump_basepath:
	node ./scripts/update-basepath.cjs $(RELEASE_VERSION)

build: bump_basepath 
	@echo "Building with EVIDENCE_VAR__release_version=$(EVIDENCE_VAR__release_version)"
	npm run build:strict

deploy: 
	# both rsync's are set up to delete extra files
	gsutil -m rsync -dr build/ $(EVIDENCE_BASE_PATH)/$(RELEASE_VERSION)/evidence
	gsutil -m rsync -dr $(EVIDENCE_BASE_PATH)/$(RELEASE_VERSION)/evidence $(EVIDENCE_BASE_PATH)/latest/evidence

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
