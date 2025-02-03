# Define the image name
IMAGE_NAME = kg-dashboard
REPO_NAME = matrix-images
PROJECT_ID = $(shell gcloud config get-value project)
BUCKET_NAME = gs://data.dev.everycure.org

# controls the release version of the KG used in the dashboard
RELEASE_VERSION ?= v0.3.0
export EVIDENCE_VAR__release_version := $(subst .,_,$(RELEASE_VERSION))

.PHONY: clean install bump_basepath run-dev

bump_basepath:
	node ./scripts/update-basepath.cjs $(RELEASE_VERSION)

build: bump_basepath
	@echo "Building with EVIDENCE_VAR__release_version=$(EVIDENCE_VAR__release_version)"
	npm run build:strict

deploy: build
	gsutil -m rsync -r build/ $(BUCKET_NAME)/evidence/$(RELEASE_VERSION)
	# gsutil rm -r $(BUCKET_NAME)/evidence/latest
	# gsutil -m rsync -r $(BUCKET_NAME)/evidence/$(RELEASE_VERSION) $(BUCKET_NAME)/evidence/latest

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

# docker centric commands
# Build the Docker image
docker-build:
	DOCKER_BUILDKIT=1 docker build --secret id=gcloud,src=$(HOME)/.config/gcloud/application_default_credentials.json .

# Run the Docker container
docker-run: docker-build
	docker run -d -p 80:80 $(IMAGE_NAME)

# Command to push the Docker container to gcloud artifact registry
docker-push: docker-build
	docker tag $(IMAGE_NAME) gcr.io/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME)
	docker push gcr.io/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME)

# Clean up Docker resources
docker-clean:
	docker rm -f $(shell docker ps -aq --filter "ancestor=$(IMAGE_NAME)")
	docker rmi -f $(IMAGE_NAME)

