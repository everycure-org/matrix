install:
	uv pip install -r requirements.in

lock:
	uv pip compile requirements.in > requirements.txt
test:
        pytest --testmon -v tests/
integration_test:
	kedro run -p fabricator
	kedro run

build: lock
	docker build -t matrix .

licenses:  build
	trivy image --scanners license --severity UNKNOWN,CRITICAL matrix

licenses_container: build
	docker run \
		-v $(PWD)/trivy.yaml:/trivy.yaml \
		-v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy  \
		image --scanners license --severity UNKNOWN,CRITICAL matrix