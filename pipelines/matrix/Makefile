build:
	docker build -t matrix .

install:
	pip install -r requirements.txt

lock:
	pip-compile --output-file=requirements.txt requirements.in

licenses:  build
	# trivy image --scanners license --severity UNKNOWN,CRITICAL matrix
	trivy image --scanners license --severity CRITICAL matrix
	trivy image --scanners license --severity HIGH,CRITICAL matrix

licenses_container: build
	docker run \
		-v $(PWD)/trivy.yaml:/trivy.yaml \
		-v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy  \
		image --scanners license --severity UNKNOWN,CRITICAL matrix