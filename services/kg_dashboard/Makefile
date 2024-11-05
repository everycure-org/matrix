# Define the image name
IMAGE_NAME = matrix-kg-dashboard

.PHONY: build run clean

# Build the Docker image
build:
	docker build -t $(IMAGE_NAME) .

# Run the Docker container
run:
	docker run -d -p 80:80 $(IMAGE_NAME)

# Clean up Docker resources
clean:
	docker rm -f $(shell docker ps -aq --filter "ancestor=$(IMAGE_NAME)")
	docker rmi -f $(IMAGE_NAME)
