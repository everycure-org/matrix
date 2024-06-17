# Makefile that builds or serves the docs.
.PHONY: build serve

build: install
	.venv/bin/mkdocs build -s #build in strict mode, all links must function

serve: install
	.venv/bin/mkdocs serve

install:
	uv venv
	uv pip install -r requirements.txt


image: 
	docker build -t docs ./

# FUTURE move away from gh pages based deployment as it creates public content. But acceptable for now
deploy: install
	firebase deploy --only hosting