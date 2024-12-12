TARGET_PLATFORM ?= linux/amd64
dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/moa_visualizer
GIT_SHA ?= $(shell git rev-parse --short HEAD)
TAG ?= $(shell cat version)

docker_build:
	docker buildx build --progress=plain --build-arg GIT_SHA=${GIT_SHA} --platform $(TARGET_PLATFORM) -t $(dev_docker_image) --load ./
	docker tag $(dev_docker_image) $(dev_docker_image):${TAG}

docker_push: docker_build
	docker push $(dev_docker_image):${TAG}

