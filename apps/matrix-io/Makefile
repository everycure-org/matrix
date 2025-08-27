PREREQUISITES := uv

default: prerequisites
	echo "done!"

prerequisites:
	$(info Checking if prerequisites are installed...)
	$(foreach exec,$(PREREQUISITES),\
        $(if $(shell command -v $(exec) 2>/dev/null),,$(error "$(exec) is not installed.")))
	$(info All prerequisites are installed.)

venv:
	if [ ! -d .venv ]; then uv venv -p 3.11; fi
	uv run python -m ensurepip --upgrade || true

install: venv
	uv pip install -r requirements.txt #force it to install in our venv

prerequisites:
	pip install uv

format:
	uv run ruff check . --fix

lock:
	uv pip compile requirements.in > requirements.txt
