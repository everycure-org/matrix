dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix
TAG ?= ${USER}
GIT_SHA ?= $(shell git rev-parse --short HEAD)
export PYSPARK_PYTHON = .venv/bin/python
export UV_PYTHON = .venv/bin/python
export TQDM_DISABLE ?= 1
TARGET_PLATFORM ?= linux/amd64
PREREQUISITES := docker gcloud python3 java uv

# ensures all make targets run in one shell (rather than line by line in new shell)
.ONESHELL: 

default: install fast_test compose_down integration_test
	echo "done!"

prerequisites:
	$(info Checking if prerequisites are installed...)
	$(foreach exec,$(PREREQUISITES),\
        $(if $(shell command -v $(exec) 2>/dev/null),,$(error "$(exec) is not installed.")))
	$(info All prerequisites are installed.)

venv: prerequisites
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
	uv run python -m ensurepip --upgrade || true

install: venv precommit-hooks
	uv pip install -r requirements.txt #force it to install in our venv

install_compatibility: venv precommit-hooks
	# expects the file requirements.overrides.txt to be present, allows devs to override
	# versions for local machine limitations (e.g. newer pytorch on macos intel)
	uv pip install -r requirements.txt --overrides requirements.overrides.txt

precommit: precommit-hooks
	git fetch origin
	uv run pre-commit run --from-ref origin/main --to-ref HEAD

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
docker_test: certificate docker_build
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

certificate:
	@echo "Generating test SSL certificate for Neo4J."
	openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout compose/certs/bolt/private.key -out compose/certs/bolt/public.crt -config compose/certs/openssl.cnf > /dev/null 2>&1
	cat compose/certs/bolt/private.key compose/certs/bolt/public.crt > compose/certs/bolt/private_and_public.pem

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
	uv run kedro run --env test -p test --without-tags xgc,not-shared

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
	uv run pre-commit clean
	rm -rf .pytest_cache .venv
	uv cache clean
	docker volume prune -f

diagnostics:
	echo "Collecting system diagnostics..."
	echo "Operating System:"
	-uname -a
	echo "\nDocker Version:"
	-docker --version || echo "Docker is not installed."
	echo "\nDocker Disk Usage:"
	-docker system df || echo "Docker is not installed or not running."
	echo "\nDocker Memory and CPU Usage:"
	-docker stats --no-stream || echo "Docker is not installed or not running."
	echo "\nGCloud Version:"
	-gcloud --version || echo "GCloud is not installed."
	echo "\nJava Version:"
	-java -version || echo "Java is not installed."
	echo "\nUV Version:"
	-uv --version || echo "UV is not installed."
	echo "\nCheck Virtual Environment (if applicable):"
	-uv run which python || echo "Python is not in PATH."
	echo "\nPython Version:"
	-uv run python3 --version || echo "Python is not installed."
	echo "\nSystem Information:"
	-uv run python -c "import platform; print(platform.uname())"
	echo "\nInstalled Python Packages:"
	-uv run pip freeze || echo "pip is not installed or no packages found."
	echo "\nEnvironment Variables (related to Python):"
	-uv run printenv | grep PYTHON || echo "No Python-related environment variables found."
	echo "\nMemory & CPU Usage:"
	-uv run python -c "import psutil; print(psutil.virtual_memory(), psutil.cpu_percent())" || echo "psutil is not installed."
	echo "\nJAVA_HOME"
	-printenv JAVA_HOME
	echo "\nSPARK_HOME"
	-printenv SPARK_HOME
	echo "\nDisk Space:"
	-df -h
	echo "\nDiagnostics complete."

# NOTE: This Makefile target was partially generated using AI assistance.
