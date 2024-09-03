dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/pubmedbert_embedding_api

# Define variables, mostly for testing
PORT := 8000
HOST := localhost
ENDPOINT := v1/embeddings
NUM_REQUESTS := 100

lock:
	uv pip compile requirements.in > requirements.txt
build:
	docker buildx build --platform linux/amd64 -t $(dev_docker_image) ./


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
LOCUST_RUN_TIME := 60s

# Run Locust load test
locust:
	@echo uv pip install locust
	@echo "Starting Locust load test..."
	locust --host=http://$(HOST):$(PORT) \
            --users $(LOCUST_USERS) \
            --spawn-rate $(LOCUST_SPAWN_RATE) \
            --run-time $(LOCUST_RUN_TIME) \
            --headless \
            --only-summary


# Run server target (optional, if you want to start the server from the Makefile)
run_server:
	python main.py

.PHONY: test_call load_test run_server
