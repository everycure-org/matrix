# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	.venv/bin/mkdocs build -s # build in strict mode, all links must function

serve: install
	.venv/bin/mkdocs serve

install:
	deactivate || echo 'no venv active, continuing'
	uv venv -p 3.11
	uv pip install -r requirements.txt -p .venv/bin/python

image: 
	docker build -t docs ./

serve_docker: image
	docker run --rm docs

deploy: build
	gcloud app deploy