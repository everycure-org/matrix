dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/pubmedbert_embedding_api
GIT_SHA ?= $(shell git rev-parse --short HEAD)
TAG ?= ${USER}

# Define variables, mostly for testing, you can set these as env variable before to control behaviour
ENDPOINT ?= http://localhost:8000
OPENAI_API_KEY ?= dummy
SLUG ?= /v1/embeddings
EMBEDDING_MODEL ?= microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext
NUM_REQUESTS := 100

lock:
	uv pip compile requirements.in > requirements.txt

venv:
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
install:
	# deactivate any potentially active venv that may be somewhere else
	deactivate || true
	uv pip install -r requirements.txt

docker_serve: docker_build
	docker run --rm $(dev_docker_image)

docker_build:
	docker buildx build --build-arg GIT_SHA=${GIT_SHA} --platform linux/amd64 -t $(dev_docker_image) ./
	docker tag $(dev_docker_image) $(dev_docker_image):${TAG}

docker_push: docker_build
	docker push $(dev_docker_image):${TAG}


# Test call target (single request)
test_call:
	@echo "Testing embeddings API..."
	curl -X POST "$(ENDPOINT)$(SLUG)" \
		-H "Content-Type: application/json" \
		-H "Authorization: Bearer ${OPENAI_API_KEY}" \
		-d '{"input": "The patient presents with fever and cough", "model": "${EMBEDDING_MODEL}"}'
		


# Define variables
LOCUST_USERS := 100
LOCUST_SPAWN_RATE := 10
LOCUST_RUN_TIME := 10s

# Run Locust load test
locust:
	@echo uv pip install locust
	@echo "Starting Locust load test..."
	.venv/bin/locust --host=$(ENDPOINT) \
            --users $(LOCUST_USERS) \
            --spawn-rate $(LOCUST_SPAWN_RATE) \
            --run-time $(LOCUST_RUN_TIME) \
            --headless \
            --only-summary


serve:
	.venv/bin/python app.py