# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## Repository Overview

MATRIX is a drug repurposing platform organized as a monorepo containing machine learning pipelines,
infrastructure as code, applications, and services. The primary components are:

- `/pipelines/matrix` - Main ML pipeline for drug repurposing using Kedro framework
- `/infra` - Infrastructure as Code using Terraform/Terragrunt for GCP deployment
- `/services` - Supporting services (KG dashboard, MOA visualizer, synonymizer, etc.)

## Development Commands

### Main Pipeline (pipelines/matrix/)

**Setup and Installation:**

```bash
cd pipelines/matrix
make install                    # Install dependencies with uv
make install_compatibility      # Install with overrides for local machine limitations
make fetch_secrets             # Fetch secrets from GCP Secret Manager
```

**Testing:**

```bash
make fast_test                 # Quick tests with testmon
make full_test                 # Complete test suite
make integration_test          # Integration tests (requires Docker services)
make docker_test               # Full E2E test in Docker
```

**Linting and Formatting:**

```bash
make format                    # Fix code formatting with ruff
make precommit                 # Run pre-commit hooks
uv run ruff check . --fix      # Direct ruff usage
```

**Running Pipelines:**

```bash
uv run kedro run --env test -p test               # Run test pipeline
uv run kedro run -p fabricator --env test         # Run fabricator pipeline
make compose_up                                   # Start Docker services
make integration_test                             # Run integration tests
```

**Docker Operations:**

```bash
make docker_build              # Build Docker image
make docker_push               # Push to registry
make compose_up                # Start services
make compose_down              # Stop services
```

### Infrastructure (infra/)

Uses Terragrunt with Terraform:

```bash
cd infra/deployments/hub/dev   # Navigate to specific environment
terragrunt validate            # Validate Terraform files
terragrunt plan               # Plan changes
terragrunt apply              # Apply changes
```

## Architecture and Key Concepts

### Pipeline Architecture

- **Kedro Framework**: Structured ML pipelines with data catalog and parameter management
- **Apache Spark**: Large-scale data preprocessing with PySpark
- **Neo4j**: Graph database for knowledge graph storage and querying
- **MLflow**: Experiment tracking and model management
- **Docker Compose**: Local development environment orchestration

### Data Flow

1. **Ingestion**: Raw data from multiple knowledge graph sources (RTX-KG2, ROBOKOP)
2. **Integration**: Merging and normalizing knowledge graphs
3. **Preprocessing**: Node normalization and data cleaning
4. **Embeddings**: Graph embeddings generation for ML features
5. **Matrix Generation**: Drug-disease association matrices
6. **Modeling**: Machine learning model training and evaluation
7. **Inference**: Generating predictions and visualizations

### Key Technologies

- **Python 3.11+** with `uv` for dependency management
- **Kedro** for pipeline structure and data catalog
- **PySpark** for distributed data processing
- **Pandera** for data validation
- **FastAPI** for API services
- **Pydantic** for settings and data validation
- **Joblib** for caching expensive computations

### Testing Strategy

- **Unit Tests**: Individual component testing with pytest
- **Integration Tests**: Full pipeline testing with Docker services
- **Spark Tests**: Distributed processing validation
- **GivenWhenThen**: Test organization format for clarity

### Environment Configuration

- **Local**: Development with Docker Compose
- **Sample**: Subset data for quick testing
- **Test**: Full test environment
- **Cloud**: Production GCP environment

## Code Standards

- Use Google-style Python docstrings
- Functional programming preferred
- Cache expensive functions with joblib
- Add AI generation note:
  `# NOTE: This <file/function/class> was partially generated using AI assistance.`
- Comments explain "why" not "what" or changes between versions
- Use `terragrunt validate` for Terraform changes in respective folders

## Common Workflows

**Start Development:**

```bash
cd pipelines/matrix
make prerequisites              # Check system requirements
make fetch_secrets             # Get GCP secrets
make install                   # Install dependencies
make compose_up                # Start services
```

**Run Full Pipeline:**

```bash
make integration_test          # Full pipeline with services
```

**Debug Pipeline:**

```bash
make compose_up                # Start services
make wipe_neo                  # Clear Neo4j data
uv run kedro run --env test    # Run specific pipeline
```

**Infrastructure Changes:**

```bash
cd infra/deployments/hub/dev
terragrunt validate            # Validate before changes
terragrunt plan               # Review changes
terragrunt apply              # Apply if approved
```

## Claude instructions

- never push to main
- never rm -rf anything that is not git versioned
- when working on a feature, use a new branch before committing, if you're on main, you are doing it
  wrong
