# Contribution Standards

## Overview

This document outlines the standards and practices for contributing to the MATRIX drug repurposing platform. These guidelines help maintain code quality, consistency, and facilitate collaboration across our development team.

## Technology Stack & Language Standards

### Languages to be used

- **Python 3.11+** - Primary language for all business logic, data processing, and ML pipelines
  - Package management via [uv](https://docs.astral.sh/uv/) (modern Python package manager)
  - Formatting and linting with [Ruff](https://docs.astral.sh/ruff/)
- **JavaScript/Node.js** - Dashboard applications using [Evidence.dev](https://evidence.dev/) framework only
- **Shell/Bash** - Infrastructure automation and utility scripts
- **SQL** - [BigQuery](https://cloud.google.com/bigquery/docs) analytics and data warehouse operations
- **HCL** - [Terraform](https://www.terraform.io/docs)/[Terragrunt](https://terragrunt.gruntwork.io/docs/) infrastructure definitions

### Restricted Languages

No other programming languages should be introduced without explicit team approval and architectural justification. This includes but is not limited to Rust, Go (for application code), Java, or C++.

## Development Workflow

### Git Workflow

- **Never commit directly** to `main` or `develop` branches
- Always work on feature branches with descriptive names:
  - `feat/descriptive-name` for new features
  - `fix/descriptive-name` for bug fixes  
  - `dev/your-name/task` for experimental work
- Create [draft pull requests](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/proposing-changes-to-your-work-with-pull-requests/about-pull-requests#draft-pull-requests) early for feedback and collaboration
- All PRs require core maintainer approval before merging
- CI checks must pass (linting, testing, security scans) before review

### Pull Request Process

1. Create feature branch from `main`
2. Make changes following code quality standards
3. Run local tests: `make fast_test`
4. Create draft PR with descriptive title and description
5. Request review from relevant team members
6. Address feedback and ensure CI passes
7. Core maintainer approval required for merge

## Code Quality Standards

### Python Standards

- **Formatting**: Use [Ruff](https://docs.astral.sh/ruff/) for formatting and linting (line length: 120)
- **Docstrings**: Use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for complex functions and classes
- **Import Organization**: Follow [PEP 8](https://pep8.org/) import ordering, automated by Ruff
- **Type Hints**: Use [Python type hints](https://docs.python.org/3/library/typing.html) for function signatures where beneficial

### Pre-commit Hooks

All commits must pass [pre-commit hooks](https://pre-commit.com/) configured in `.pre-commit-config.yaml`:

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

!!! tip
    Install the [Ruff VSCode extension](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff) for automatic formatting and import handling.

## Testing Requirements

### Test Hierarchy

Choose the appropriate test level based on your changes:

- **`make fast_test`** - Quick validation during development using [pytest-testmon](https://testmon.org/)
- **`make full_test`** - Complete test suite, required before PR submission
- **`make integration_test`** - Required for data pipeline changes, includes Docker services
- **`make docker_test`** - End-to-end functionality testing in containerized environment

### Testing Framework

- **Primary**: [pytest](https://docs.pytest.org/) with plugins for coverage and mocking
- **Coverage**: Maintain reasonable test coverage for new code
- **Data Testing**: Use [Pandera](https://pandera.readthedocs.io/) for data validation testing

## Infrastructure Standards

### Infrastructure as Code

All infrastructure changes must go through [Terraform](https://www.terraform.io/)/[Terragrunt](https://terragrunt.gruntwork.io/):

```bash
cd infra/deployments/hub/dev
terragrunt validate     # Always run before commits
terragrunt plan        # Review changes
```

- Infrastructure PRs require DevOps team review
- Test in development environment before production deployment
- Follow [GCP best practices](https://cloud.google.com/docs/terraform/best-practices-for-terraform) for cloud resources

### Container Standards

- **Docker**: Use multi-stage builds and [security best practices](https://docs.docker.com/develop/security-best-practices/)
- **Kubernetes**: Follow [Kubernetes best practices](https://kubernetes.io/docs/concepts/configuration/overview/) for resource definitions
- **Monitoring**: Integration with [Prometheus](https://prometheus.io/docs/) and [Grafana](https://grafana.com/docs/) via kube-prometheus-stack

## Documentation Standards

### Code Documentation

- Google-style docstrings for complex functions and public APIs
- Comments should explain "why" decisions were made, not "what" the code does
- Keep documentation close to code - update docs with code changes

### Development Documentation

- All development documentation stored in `/docs` folder
- Use [MkDocs](https://www.mkdocs.org/) for documentation site generation
- Update relevant documentation when changing workflows or adding features

## Code Review Guidelines

Code review is essential for knowledge sharing and code quality. 

### For Reviewers

- Ask questions to understand implementation choices
- Share alternative approaches when helpful
- Focus on learning opportunities and knowledge transfer
- Point out potential bugs or security issues
- Assign yourself if you have relevant expertise

### For Authors

- Explain design decisions and trade-offs made  
- Be receptive to suggestions and feedback
- Use reviews as mentoring opportunities
- Ensure PR description clearly explains the change

### Review Requirements

- At least one [core maintainer](https://github.com/orgs/everycure-org/teams/core-maintainers/) approval required
- All CI checks must pass
- Address reviewer feedback before merging
- Self-review your changes before requesting review

## Release Process

- Use [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) with [semantic versioning](https://semver.org/)
- Automated release PR creation with changelog generation
- Test releases in sample environment before production
- Follow [conventional commit](https://www.conventionalcommits.org/) format for clear release notes

## Core Tools and Frameworks

### Machine Learning & Data Science
- **[Kedro](https://docs.kedro.org/)** - ML pipeline framework and project structure
- **[PySpark](https://spark.apache.org/docs/latest/api/python/)** - Distributed data processing
- **[Neo4j](https://neo4j.com/docs/)** - Graph database for knowledge graphs
- **[MLflow](https://mlflow.org/docs/latest/index.html)** - Experiment tracking and model registry

### Cloud & Infrastructure  
- **[Google Cloud Platform](https://cloud.google.com/docs)** - Primary cloud provider
- **[Terraform](https://www.terraform.io/docs)** - Infrastructure provisioning
- **[Kubernetes](https://kubernetes.io/docs/)** - Container orchestration via GKE

### Development Tools
- **[uv](https://docs.astral.sh/uv/)** - Python package and project management
- **[Ruff](https://docs.astral.sh/ruff/)** - Python linting and formatting
- **[pytest](https://docs.pytest.org/)** - Testing framework
- **[Docker](https://docs.docker.com/)** - Containerization

## Getting Help

- Review existing issues and PRs for similar problems
- Check the [development documentation](../index.md) for setup and workflow guidance
- Ask questions in team communication channels
- Tag relevant team members for domain-specific questions
