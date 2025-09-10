default: prerequisites
	echo "done!"

venv:
	if [ ! -d .venv ]; then uv venv -p 3.11; fi

install: venv
	uv sync

prerequisites:
	pip install uv

format:
	uv run ruff check . --fix

lock:
	uv pip compile requirements.in > requirements.txt
