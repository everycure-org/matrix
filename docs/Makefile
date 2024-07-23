# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	.venv/bin/mkdocs build -s # build in strict mode, all links must function

serve: install
	.venv/bin/mkdocs serve

install:
	uv venv
<<<<<<< HEAD
	uv pip install --target .venv -r requirements.txt


image: 
	docker build -t docs ./

serve_docker: image
	docker run --rm docs

deploy: build
	gcloud app deploy
=======
	uv pip install -r requirements.txt
>>>>>>> main
