docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix

TAG ?= ${USER}
GIT_SHA ?= $(shell git rev-parse --short HEAD)
export TQDM_DISABLE ?= 1
TARGET_PLATFORM ?= linux/amd64
PREREQUISITES := docker gcloud python3 java uv kubectl

# Add GCP project variables
GCP_PROJECT ?= mtrx-hub-dev-3of
SA_KEY_SECRET_NAME ?= storage-viewer-sa-key
SA_KEY_PATH ?= conf/local/service-account.json
OAUTH_CLIENT_SECRET_SECRET_NAME ?= oauth_client_secret
OAUTH_CLIENT_SECRET_PATH ?= conf/local/oauth_client_secret.txt

fetch_secrets:
	echo "Fetching secrets from Secret Manager..."; \
	mkdir -p conf/local; \
	for pair in "$(SA_KEY_SECRET_NAME) $(SA_KEY_PATH)" "$(OAUTH_CLIENT_SECRET_SECRET_NAME) $(OAUTH_CLIENT_SECRET_PATH)"; do \
		set -- $$pair; \
		echo "Fetching secret $$1..."; \
		gcloud secrets versions access latest --secret="$$1" --project=$(GCP_PROJECT) > "$$2"; \
		chmod 600 "$$2"; \
		echo "Secret $$1 saved to $$2"; \
	done

# ensures all make targets run in one shell (rather than line by line in new shell)
.ONESHELL: 

default: prerequisites fetch_sa_key install fast_test integration_test

local_test: install fast_test integration_test

.PHONY: always 
# steps here get run on _every_ make call, no matter the target. Make sure this is 100% idempotent and fast.
always:
	# never track notebooks in the pipeline subfolder
	git update-index --assume-unchanged *.ipynb
%: always
	@:
default: prerequisites fetch_secrets install fast_test integration_test
	echo "done!"

prerequisites:
	$(info Checking if prerequisites are installed...)
	$(foreach exec,$(PREREQUISITES),\
        $(if $(shell command -v $(exec) 2>/dev/null),,$(error "$(exec) is not installed.")))
	$(info All prerequisites are installed.)
	$(info Checking gcloud auth...)
	@if ! gcloud auth print-access-token >/dev/null 2>&1; then \
		echo "Not logged into gcloud. Please run 'gcloud auth login' first."; \
		exit 1; \
	fi


install: precommit-hooks 
	uv sync # UV-native approach to install dependencies

install_argo:
	curl -sLO https://github.com/argoproj/argo-workflows/releases/latest/download/argo-linux-amd64.gz
	gunzip argo-linux-amd64.gz
	chmod +x argo-linux-amd64
	sudo mv argo-linux-amd64 /usr/local/bin/argo
	argo version

precommit: precommit-hooks
	git fetch origin
	uv run pre-commit run --from-ref origin/main --to-ref HEAD

precommit-hooks:
	uv sync --group pre-commit
	uv run pre-commit install --install-hooks

check_release_version_environment_variable:
	if [ -z "${RELEASE_VERSION}" ]; then
	echo "::error::release_version input is required"
	exit 1
	fi
	echo "Using release version: ${RELEASE_VERSION}"

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
	IMG="$(docker_image)" docker compose -f compose/docker-compose.yml \
		-f compose/docker-compose.ci.yml \
		up \
		--force-recreate \
		--abort-on-container-exit \
		--exit-code-from matrix-pipeline

docker_build:
	# NOTE: This Makefile was partially generated using AI assistance.
	# note the && operator is essential to catch docker buildx errors on linux systems
	docker buildx build \
		--progress=plain \
		--build-arg GIT_SHA=${GIT_SHA} \
		--build-arg DISABLE_UV_CACHE=0 \
		--platform $(TARGET_PLATFORM) \
		-t $(docker_image) \
		--load \
		-f Dockerfile ../../ && \
	docker tag $(docker_image) $(docker_image):${TAG}

docker_auth:
	gcloud auth configure-docker us-central1-docker.pkg.dev

docker_push: docker_build
	docker push $(docker_image):${TAG}

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
	uv lock

wipe_neo:
	# assumes you have neo4j running using docker-compose from `compose_up`
	uv run scripts/wipe_neo4j.py 'everycure-test'

integration_test: compose_up wipe_neo
	ulimit -n 10000
	uv run kedro run --env test -p test

fabricate:
	uv run kedro run -p fabricator --env test

licenses_container: docker_build
	docker run \
		-v $(PWD)/trivy.yaml:/trivy.yaml \
		-v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy  \
		image --scanners license --severity UNKNOWN,CRITICAL $(docker_image)

clean_sa_key:
	rm -f $(SA_KEY_PATH)

clean: clean_sa_key
	@echo "cleaning various cache locations to ensure clean installation is possible"
	@echo "this may be necessary e.g. when updating one of our local packages"
	uv run pre-commit clean
	rm -rf .pytest_cache .venv
	uv cache clean
	docker volume prune -f
	rm -rf data/cache
	rm -rf data/test/cache

diagnostics:
	echo "Collecting system diagnostics..."
	echo "Operating System:"
	-uname -a
	echo "\nDocker Version:"
	-if ! docker --version >/dev/null 2>&1; then echo "Docker is not installed."; fi
	echo "\nDocker Disk Usage:"
	-if ! docker system df >/dev/null 2>&1; then echo "Docker is not installed or not running."; else docker system df; fi
	echo "\nDocker Memory and CPU Usage:"
	-if ! docker stats --no-stream >/dev/null 2>&1; then echo "Docker is not installed or not running."; else docker stats --no-stream; fi
	echo "\nGCloud Version:"
	-if ! gcloud --version >/dev/null 2>&1; then echo "GCloud is not installed."; fi
	echo "\nJava Version:"
	-if ! java -version >/dev/null 2>&1; then echo "Java is not installed."; fi
	echo "\nUV Version:"
	-if ! uv --version >/dev/null 2>&1; then echo "UV is not installed."; fi
	echo "\nCheck Virtual Environment (if applicable):"
	-if ! uv run which python >/dev/null 2>&1; then echo "Python is not in PATH."; fi
	echo "\nKubectl permissions:"
	-kubectl auth can-i --list
	echo "\nPython Version:"
	-if ! uv run python3 --version >/dev/null 2>&1; then echo "Python is not installed."; fi
	echo "\nSystem Information:"
	-uv run python -c "import platform; print(platform.uname())"
	echo "\nInstalled Python Packages:"
	-if ! uv run pip freeze >/dev/null 2>&1; then echo "pip is not installed or no packages found."; else uv run pip freeze; fi
	echo "\nEnvironment Variables (related to Python):"
	-if ! uv run printenv | grep -q PYTHON; then echo "No Python-related environment variables found."; else uv run printenv | grep PYTHON; fi
	echo "\nMemory & CPU Usage:"
	-if ! uv run python -c "import psutil; print(psutil.virtual_memory(), psutil.cpu_percent())" >/dev/null 2>&1; then echo "psutil is not installed."; else uv run python -c "import psutil; print(psutil.virtual_memory(), psutil.cpu_percent())"; fi
	echo "\nJAVA_HOME"
	-printenv JAVA_HOME
	echo "\nSPARK_HOME"
	-printenv SPARK_HOME
	echo "\nDisk Space:"
	-df -h
	echo "\nDiagnostics complete."


