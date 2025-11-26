# MATRIX - AI Coding Agent Instructions

MATRIX is a drug repurposing platform built as a monorepo with ML pipelines, infrastructure, and supporting services. This guide helps AI agents understand the project's unique architecture and conventions.

## Repository Structure & Key Patterns

### uv Workspace Architecture
- **Root workspace**: All packages share one virtual environment via `uv sync`
- **Package structure**: `libs/` (shared), `pipelines/` (Kedro ML), `services/` (apps), `infra/` (Terraform)
- **Local development**: Libraries auto-linked in editable mode
- **Dependencies**: Root `pyproject.toml` includes all workspace members for single-command setup

### Kedro Pipeline Framework
- **Main pipeline**: `pipelines/matrix/` - drug repurposing ML pipeline
- **Pipeline structure**: Multiple specialized pipelines (ingestion, embeddings, modeling, inference)
- **Data catalog**: Kedro manages data flows between pipeline stages
- **Configuration**: Environment-specific configs in `conf/` (base, local, test, cloud)

### Technology Stack Integration
- **PySpark**: Large-scale data preprocessing with specific patterns for distributed computing
- **Neo4j**: Knowledge graph database (Docker Compose for local dev)
- **MLflow**: Experiment tracking with matrix-mlflow-utils library
- **Pandera**: Data validation schemas with custom matrix-pandera extensions
- **Joblib**: Caching expensive functions (`.cache` folder default)

## Essential Development Workflows

### Setup & Installation
```bash
make setup                    # Install dependencies + pre-commit hooks
cd pipelines/matrix && make   # Run full integration test
```

### Testing Hierarchy
- `make fast_test`: Quick tests with testmon
- `make integration_test`: Fabricated data + Docker services, pipeline local
- `make docker_test`: Full E2E with pipeline in Docker
- Use `TARGET_PLATFORM=linux/arm64` on ARM machines

### Docker Environment
```bash
make compose_up               # Start Neo4j + supporting services
make wipe_neo                 # Clear Neo4j data for fresh runs
```

### Pipeline Execution Patterns
```bash
uv run kedro run --env test -p test           # Test pipeline
uv run kedro run -p fabricator --env test     # Data fabrication
```

## Code Conventions & Patterns

### AI-Generated Code Marking
Always add to generated code:
```python
# NOTE: This <file/function/class> was partially generated using AI assistance.
```

### Testing Patterns
- **GivenWhenThen format**: Structure tests with clear phases
- **pytest markers**: Use `@pytest.mark.integration` and `@pytest.mark.spark`
- **Spark tests**: Require `PYSPARK_PYTHON` environment setup
- **Value-focused**: Only generate tests with clear, intuitive value

### Python Standards
- **Docstrings**: Google-style format
- **Functional style**: Preferred over OOP where applicable
- **Performance**: Cache expensive functions with `joblib`
- **Comments**: Explain "why", not "what" or version changes
- **Data validation**: Use Pandera schemas for DataFrame validation

### Infrastructure Patterns
- **Terragrunt + Terraform**: Navigate to specific env folder (e.g., `infra/deployments/hub/dev/`)
- **Always validate**: Run `terragrunt validate` before changes
- **Environment structure**: dev/prod environments with shared modules

## Integration Points & Dependencies

### Shared Libraries Usage
- **matrix-auth**: Environment utilities and authentication
- **matrix-gcp-datasets**: BigQuery and Spark GCP integration
- **matrix-fabricator**: Test data generation
- **matrix-inject**: Dependency injection patterns
- **matrix-pandera**: Extended data validation

### External Service Integration
- **Neo4j**: Knowledge graph storage (local: Docker, cloud: managed)
- **MLflow**: Experiment tracking with custom utilities
- **Google Cloud**: BigQuery, Storage, IAM integration
- **OpenAI**: LLM integration for embeddings

### Data Flow Architecture
1. **Ingestion**: RTX-KG2, ROBOKOP knowledge graphs
2. **Integration**: Graph merging and normalization
3. **Preprocessing**: Node normalization, Spark-based cleaning
4. **Embeddings**: Graph embeddings for ML features
5. **Matrix Generation**: Drug-disease association matrices
6. **Modeling**: ML training with experiment tracking
7. **Inference**: Prediction generation and visualization

## Environment & Configuration

### Environment Types
- **local**: Docker Compose development
- **sample**: Subset data for quick iteration
- **test**: Full test environment
- **cloud**: Production GCP environment

### Secret Management
- `make fetch_secrets`: Pull from GCP Secret Manager
- Local secrets in `conf/local/` (never commit)

## Critical Constraints

- **Never push to main**: Use feature branches
- **No destructive commands**: Avoid `rm -rf` on non-git files  
- **Branch workflow**: Create new branch before committing features
- **Dependency coordination**: Changes affecting multiple packages require workspace-level testing

## Key Files for Architecture Understanding

- `pyproject.toml` (root): Workspace configuration
- `pipelines/matrix/src/matrix/pipelines/`: Pipeline implementations
- `infra/root.hcl`: Terragrunt global configuration
- `docs/src/infrastructure/LiteLLM-New-Provider-Guide.md`: Runbook for onboarding new LLM providers (secrets + ArgoCD model_list updates)
- `pipelines/matrix/compose/docker-compose.yml`: Local development services
- `libs/*/src/`: Shared library implementations

This project prioritizes reproducible ML workflows, robust testing, and infrastructure automation. Focus on understanding the data flow through Kedro pipelines and the workspace dependency management when making changes.
