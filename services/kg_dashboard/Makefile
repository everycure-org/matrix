# Define the image name
IMAGE_NAME = kg-dashboard
REPO_NAME = matrix-images
PROJECT_ID = $(shell gcloud config get-value project)


.PHONY: build run clean

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run: build
	docker run -d -p 80:80 $(IMAGE_NAME)

# Command to push the Docker container to gcloud artifact registry
push:
	docker tag $(IMAGE_NAME) gcr.io/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME)
	docker push gcr.io/$(PROJECT_ID)/$(REPO_NAME)/$(IMAGE_NAME)

# Clean up Docker resources
clean:
	docker rm -f $(shell docker ps -aq --filter "ancestor=$(IMAGE_NAME)")
	docker rmi -f $(IMAGE_NAME)

install-dev:
	npm install

run-sources-dev: install-dev
	npm run sources

run-dev: run-sources-dev
	npm run dev
