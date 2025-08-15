# MATRIX - Drug Repurposing Platform

[![CI pipline](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml/badge.svg?branch=main)](https://github.com/everycure-org/matrix/actions/workflows/matrix-ci.yml)
[![Documentation Page Deployment](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/docs-deploy.yml)
[![KG Dashboard Deployment](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml/badge.svg)](https://github.com/everycure-org/matrix/actions/workflows/kg-dashboard-deploy.yml)

Welcome to MATRIX! This repository contains our drug repurposing platform which includes data
science pipelines, infrastructure, and documentation.

MATRIX is organized as a monorepo with multiple packages managed by **uv**. The repository contains infrastructure, machine learning pipelines, shared libraries, applications and services. Each component resides in a dedicated directory with its own README providing detailed setup and usage instructions.

## 📁 Repository Structure

### Core Directories:
- `/pipelines/matrix` - 🧬 [Main Drug Repurposing ML Pipeline](pipelines/matrix/README.md)
- `/libs/` - 📚 Shared Libraries:
  - `matrix-auth/` - Authentication and environment utilities
  - `matrix-fabricator/` - Data fabrication and generation tools  
  - `matrix-gcp-datasets/` - GCP integration and Spark utilities
  - `matrix-mlflow-utils/` - MLflow integration and metric utilities
- `/infra` - 🏗️ [Infrastructure as Code (IaC) configuration](docs/src/infrastructure)
- `/apps/matrix-cli` - 🛠️ Command Line Interface Tools
- `/services` - ⚙️ Supporting Services and APIs
- `/docs` - 📖 Documentation site generation

### 🔧 uv Workspace
The repository uses uv's workspace feature for efficient multi-package development:
- **Root `pyproject.toml`**: Defines the workspace configuration
- **Individual packages**: Each directory with a `pyproject.toml` is a separate package
- **Shared dependencies**: Common dependencies managed at the workspace level
- **Local development**: Libraries automatically linked in editable mode

## 🎬 Getting Started

Ready to get started? Go to our Getting Started section

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

---

## Brand and Trademark Notice

**Important**: The "Every Cure" name, logo, and related trademarks are the exclusive property of
Every Cure. Contributors and users of this open-source project are not authorized to use the Every
Cure brand, logo, or trademarks in any way that suggests endorsement, affiliation, or sponsorship
without explicit written permission from Every Cure.

This project is open source and available under the terms of its license, but the Every Cure brand
and trademarks remain protected. Please respect these intellectual property rights.
