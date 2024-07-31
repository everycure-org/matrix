dev_docker_image = us-central1-docker.pkg.dev/mtrx-hub-dev-3of/matrix-images/matrix


install:
	uv pip install -r requirements.in

lock:
	uv pip compile requirements.in > requirements.txt

test:
	pytest --testmon -v tests/


wipe_neo:
	# assumes you have neo4j running using docker-compose setup from deployments/compose/
	python scripts/wipe_neo4j.py

integration_test: wipe_neo
	kedro run --env test -p test --runner ThreadRunner --without-tags xgc

fabricate:
	kedro run -p fabricator

build: lock
	docker build -t matrix .

docker_push_dev:
	docker buildx build --platform linux/amd64 -t matrix_amd ./
	docker tag matrix_amd $(dev_docker_image):${USER}
	docker push $(dev_docker_image):${USER}

argo_template:
	python ./src/matrix/argo.py generate-argo-config $(dev_docker_image):latest

licenses:  build
	trivy image --scanners license --severity UNKNOWN,CRITICAL matrix

licenses_container: build
	docker run \
		-v $(PWD)/trivy.yaml:/trivy.yaml \
		-v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy  \
		image --scanners license --severity UNKNOWN,CRITICAL matrix