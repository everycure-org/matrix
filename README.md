# MATRIX - Drug Repurposing Platform

[![CI pipline](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml/badge.svg?branch=main)](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml)
[![Documentation Page Deployment](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml)
[![KG Dashboard Deployment](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml)

Welcome to MATRIX! This repository contains our drug repurposing platform which includes data
science pipelines, infrastructure, and documentation.

MATRIX is a drug repurposing platform organized as a monorepo containing machine learning pipelines,
infrastructure as code, applications, and services. The repository uses **uv**'s workspace feature
for efficient multi-package development with infrastructure, machine learning pipelines, shared
libraries, applications and services.

## 📁 Repository Structure

### Core Directories:

- `/pipelines/matrix` - 🧬 [Main Drug Repurposing ML Pipeline](pipelines/matrix/README.md) - Main ML
  pipeline for drug repurposing using Kedro framework
- `/libs/` - 📚 Shared Libraries:
  - `matrix-auth/` - Authentication and environment utilities
  - `matrix-fabricator/` - Data fabrication and generation tools
  - `matrix-gcp-datasets/` - GCP integration and Spark utilities
  - `matrix-mlflow-utils/` - MLflow integration and metric utilities
- `/infra` - 🏗️ [Infrastructure as Code (IaC) configuration](docs/src/infrastructure) -
  Infrastructure as Code using Terraform/Terragrunt for GCP deployment
- `/services` - ⚙️ Supporting Services and APIs - Supporting services (KG dashboard, MOA visualizer,
  synonymizer, etc.)
- `/docs` - 📖 Documentation site generation

### 🔧 uv Workspace

The repository uses uv's workspace feature for efficient multi-package development:

- **Root `pyproject.toml`**: Defines the workspace configuration
- **Individual packages**: Each directory with a `pyproject.toml` is a separate package
- **Shared dependencies**: Common dependencies managed at the workspace level
- **Local development**: Libraries automatically linked in editable mode

## 🎬 Getting Started

Ready to get started? Go to our Getting Started section

**Start Development:**

```bash
make setup              # check for dependencies and install precommit hooks
cd pipelines/matrix     #
make                    # run full integration test locally
```

## Development Commands

### Main Pipeline (pipelines/matrix/)

**Setup and Installation:**

```bash
make install                   # Install dependencies with uv
cd pipelines/matrix
make fetch_secrets             # Fetch secrets from GCP Secret Manager
```

**Testing:**

```bash
make fast_test                 # Quick tests with testmon
make full_test                 # Complete test suite
make integration_test          # Integration tests using fabricated data. Services in Docker, pipeline not
make docker_test               # Full E2E test, pipeline also in docker
```

> 💡 use `make docker_test TARGET_PLATFORM=linux/arm64` on ARM machines for better performance

**Linting and Formatting:**

Run these at the root of the repo.

```bash
make format                    # Fix code formatting with ruff
make precommit                 # Run pre-commit hooks
uv run ruff check . --fix      # Direct ruff usage
```

**Running Pipelines:**

Inside of `pipelines/matrix/`

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
- Comments explain "why" not "what" or changes between versions
- Use `terragrunt validate` for Terraform changes in respective folders

## Common Workflows

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

## AI assistant instructions

- never push to main (you can't anyways but you also should not try)
- never rm -rf anything that is not git versioned
- when working on a feature, use a new branch before committing

## 📚 Documentation

Please visit our [Documentation Page](http://docs.dev.everycure.org) for all details regarding the
infrastructure, the repurposing pipeline or evaluation metrics.

## Contributing

We welcome and encourage all external contributions! Please see our
[Contributing Guide](CONTRIBUTING.md) for detailed information on how to contribute to the MATRIX
project.

## 🔗 Related Projects

- [MATRIX disease list](https://github.com/everycure-org/matrix-disease-list) - Repo to manage the
  MATRIX disease list.
- [MATRIX drug list](https://github.com/everycure-org/matrix-drug-list) - Repo to manage the MATRIX
  drug list.

> Note both of these will eventually be merged into this monorepo.

---

## Brand and Trademark Notice

**Important**: The "Every Cure" name, logo, and related trademarks are the exclusive property of
Every Cure. Contributors and users of this open-source project are not authorized to use the Every
Cure brand, logo, or trademarks in any way that suggests endorsement, affiliation, or sponsorship
without explicit written permission from Every Cure.

This project is open source and available under the terms of its license, but the Every Cure brand
and trademarks remain protected. Please respect these intellectual property rights.
