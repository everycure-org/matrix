dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix

.PHONY: venv install precommit test compose-down docker_test

# executes E2E integration test fully in docker. 
docker_test: docker_build
	IMG="$(dev_docker_image)" docker compose -f compose/docker-compose.yml \
		-f compose/docker-compose.ci.yml \
		up \
		--abort-on-container-exit \
		--exit-code-from matrix-pipeline

docker_build:
	docker buildx build --platform linux/amd64 -t $(dev_docker_image) ./
	docker tag $(dev_docker_image) $(dev_docker_image):${USER}

docker_push: docker_build
	docker push $(dev_docker_image):${USER}

compose-up:
	docker compose -f compose/docker-compose.yml up -d --wait

compose-down:
	docker compose -f compose/docker-compose.yml down

install:
	uv pip install -r requirements.txt

lock:
	uv pip compile requirements.in > requirements.txt

test:
	TESTMON_DATAFILE=/tmp/.testmondata .venv/bin/pytest --testmon -v tests/

venv:
	uv venv -p 3.11

precommit-hooks:
	uv pip install pre-commit
	.venv/bin/pre-commit install --install-hooks

precommit: precommit-hooks
	git fetch origin
	.venv/bin/pre-commit run -s origin/main -o HEAD

wipe_neo:
	# assumes you have neo4j running using docker-compose setup from deployments/compose/
	.venv/bin/python scripts/wipe_neo4j.py 'everycure-test'

integration_test: compose-up wipe_neo
	# NOTE: We are running without xgc due to its long runtime. 
	.venv/bin/kedro run --env test -p test --runner ThreadRunner --without-tags xgc,not-shared

fabricate:
	.venv/bin/kedro run -p fabricator --env test

build: lock
	docker build -t matrix .


argo_template:
	python ./src/matrix/argo.py generate-argo-config $(dev_docker_image)

licenses:  build
	trivy image --scanners license --severity UNKNOWN,CRITICAL matrix

licenses_container: build
	docker run \
		-v $(PWD)/trivy.yaml:/trivy.yaml \
		-v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy  \
		image --scanners license --severity UNKNOWN,CRITICAL matrix