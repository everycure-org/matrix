.PHONY: data run
TARGET_PLATFORM ?= linux/amd64
dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/moa_visualizer
GIT_SHA ?= $(shell git rev-parse --short HEAD)
TAG ?= $(shell cat version)

DATA_INPUT_PATH = gs://mtrx-us-central1-hub-dev-storage/kedro/data/releases/v0.2.5-rtx-only/runs/feature-moa-extraction-4472893d/datasets/moa_extraction/reporting
MOA_DB_PATH=data/moa_extraction.db
GCP_PROJECT=project-silc

data:
	DATA_INPUT_PATH=$(DATA_INPUT_PATH) GCP_PROJECT=$(GCP_PROJECT) IMAGE_VERSION=$(TAG) MOA_DB_PATH=$(MOA_DB_PATH) python src/init.py

run:
	DATA_INPUT_PATH=$(DATA_INPUT_PATH) GCP_PROJECT=$(GCP_PROJECT) IMAGE_VERSION=$(TAG) MOA_DB_PATH=$(MOA_DB_PATH) streamlit run src/streamlit_app.py

docker_build:
	docker buildx build --progress=plain --build-arg GIT_SHA=${GIT_SHA} --platform $(TARGET_PLATFORM) -t $(dev_docker_image) --load ./
	docker tag $(dev_docker_image) $(dev_docker_image):${TAG}

docker_push: docker_build
	docker push $(dev_docker_image):${TAG}