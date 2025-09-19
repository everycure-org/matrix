default: prerequisites
	echo "done!"

install:
	uv sync

prerequisites:
	pip install uv

format:
	uv run ruff check . --fix

