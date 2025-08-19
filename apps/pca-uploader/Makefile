.PHONY: install run run-example clean help

# Default values
BATCH_SIZE ?= 2000
MAX_WORKERS ?= 8
PCA_FILEPATH ?= ../../data/embeddings/reporting/topological_pca

help:  ## Show this help message
	@echo "PCA Uploader - Available commands:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	uv sync

install-dev:  ## Install dependencies including dev tools
	uv sync --dev

run:  ## Run the PCA uploader with default settings
	python upload_pca_to_neo4j.py upload

run-custom:  ## Run with custom parameters (set BATCH_SIZE, MAX_WORKERS)
	python upload_pca_to_neo4j.py upload --batch-size $(BATCH_SIZE) --max-workers $(MAX_WORKERS)

run-example:  ## Run the example usage script
	python example_kedro_usage.py

status:  ## Check upload status and progress
	python upload_pca_to_neo4j.py status

clean-checkpoints:  ## Clean checkpoint files
	python upload_pca_to_neo4j.py clean

resume:  ## Resume interrupted upload
	python upload_pca_to_neo4j.py upload

clean:  ## Clean up any temporary files
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
	rm -rf .uv/ 2>/dev/null || true
	rm -rf dist/ 2>/dev/null || true
	rm -rf build/ 2>/dev/null || true

test-connection:  ## Test Neo4j connection (requires environment variables set)
	python -c "from upload_pca_to_neo4j import Neo4jPCAUploader; import os; u = Neo4jPCAUploader(os.getenv('NEO4J_HOST', 'bolt://127.0.0.1:7687'), os.getenv('NEO4J_USER', 'neo4j'), os.getenv('NEO4J_PASSWORD', 'admin')); print('Connection test:', u.test_connection())"
