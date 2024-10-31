dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix
TAG ?= ${USER}
GIT_SHA ?= $(shell git rev-parse --short HEAD)
export PYSPARK_PYTHON = .venv/bin/python
export UV_PYTHON = .venv/bin/python
export TQDM_DISABLE ?= 1
TARGET_PLATFORM ?= linux/amd64

# ensures all make targets run in one shell (rather than line by line in new shell)
.ONESHELL: 


default: install fast_test compose_down integration_test
	echo "done!"

prerequisites:
	@command -v docker >/dev/null 2>&1 || { echo "Error: docker is not installed." >&2; exit 1; }
	@command -v gcloud >/dev/null 2>&1 || { echo "Error: gcloud is not installed." >&2; exit 1; }
	@command -v python3 >/dev/null 2>&1 || { echo "Error: python3 is not installed." >&2; exit 1; }
	@command -v java >/dev/null 2>&1 || { echo "Error: java is not installed." >&2; exit 1; }
	@command -v uv >/dev/null 2>&1 || { echo "Error: uv is not installed." >&2; exit 1; }
	@echo "All prerequisites are installed."


venv: prerequisites
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
	uv run python -m ensurepip --upgrade || true

install: venv precommit-hooks
	uv pip install -r requirements.txt #force it to install in our venv


precommit: precommit-hooks
	git fetch origin
	uv run pre-commit run -s origin/main -o HEAD

precommit-hooks:
	uv pip install pre-commit 
	uv run pre-commit install --install-hooks

full_test:
	# activate venv to ensure spark doesn't have python driver mismatches
	uv run pytest -v tests/

# fast_test: export PYSPARK_PYTHON = uv run python
fast_test: 
	# activate venv to ensure spark doesn't have python driver mismatches
	TESTMON_DATAFILE=/tmp/.testmondata uv run pytest --testmon -v tests/

format:
	uv run ruff check . --fix

# executes E2E integration test fully in docker. 
docker_test: docker_build
	IMG="$(dev_docker_image)" docker compose -f compose/docker-compose.yml \
		-f compose/docker-compose.ci.yml \
		up \
		--force-recreate \
		--abort-on-container-exit \
		--exit-code-from matrix-pipeline

docker_build:
	docker buildx build --progress=plain --build-arg GIT_SHA=${GIT_SHA} --platform $(TARGET_PLATFORM) -t $(dev_docker_image) --load ./
	docker tag $(dev_docker_image) $(dev_docker_image):${TAG}

docker_push: docker_build
	docker push $(dev_docker_image):${TAG}

compose_up:
	docker compose -f compose/docker-compose.yml up -d --wait --remove-orphans
	
compose_down:
	bash scripts/compose_down_retry.sh
	# docker compose -f compose/docker-compose.yml down

lock:
	uv pip compile requirements.in > requirements.txt

wipe_neo:
	# assumes you have neo4j running using docker-compose from `compose_up`
	uv run python scripts/wipe_neo4j.py 'everycure-test'

integration_test: compose_up wipe_neo
	# NOTE: We are running without xgc due to its long runtime. 
	ulimit -n 10000
	uv run kedro run --env test -p test --runner ThreadRunner --without-tags xgc,not-shared

fabricate:
	uv run kedro run -p fabricator --env test

licenses_container: docker_build
	docker run \
		-v $(PWD)/trivy.yaml:/trivy.yaml \
		-v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy  \
		image --scanners license --severity UNKNOWN,CRITICAL $(dev_docker_image)

clean:
	@echo "cleaning various cache locations to ensure clean installation is possible"
	@echo "this may be necessary e.g. when updating one of our local packages"
	rm -rf .pytest_cache .venv
	rm -rf $(uv cache dir) ~/.cache/pre-commit
	docker volume prune -f
