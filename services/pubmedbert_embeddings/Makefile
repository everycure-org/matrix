dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/pubmedbert_embedding_api
GIT_SHA ?= $(shell git rev-parse --short HEAD)
TAG ?= ${USER}

# Define variables, mostly for testing
PORT := 8000
HOST := localhost
ENDPOINT := v1/embeddings
NUM_REQUESTS := 100

lock:
	uv pip compile requirements.in > requirements.txt

serve: docker_build
	docker run --rm $(dev_docker_image)

docker_build:
	docker buildx build --build-arg GIT_SHA=${GIT_SHA} --platform linux/amd64 -t $(dev_docker_image) ./
	docker tag $(dev_docker_image) $(dev_docker_image):${TAG}

docker_push: docker_build
	docker push $(dev_docker_image):${TAG}


# Test call target (single request)
test_call:
	@echo "Testing embeddings API..."
	@curl -X POST "http://$(HOST):$(PORT)/$(ENDPOINT)" \
		-H "Content-Type: application/json" \
		-d '{"input": "The patient presents with fever and cough", "model": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"}' \
		| json_pp


# Define variables
LOCUST_USERS := 100
LOCUST_SPAWN_RATE := 10
LOCUST_RUN_TIME := 20s

# Run Locust load test
locust:
	@echo uv pip install locust
	@echo "Starting Locust load test..."
	.venv/bin/locust --host=http://$(HOST):$(PORT) \
            --users $(LOCUST_USERS) \
            --spawn-rate $(LOCUST_SPAWN_RATE) \
            --run-time $(LOCUST_RUN_TIME) \
            --headless \
            --only-summary


# Run server target (optional, if you want to start the server from the Makefile)
run_server:
	python main.py

.PHONY: test_call load_test run_server
