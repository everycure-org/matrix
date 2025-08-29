# Repository Structure

After getting familiar with the high-level overview of MATRIX and kedro, let's explore the key files and folders in our codebase. Understanding these will help you navigate the repository more effectively.

!!! info
    As MATRIX is a monorepo, it contains codebase for the data science platform (`pipelines/matrix`) but also infrastructure (`infra`) , documentation (`docs`), and supplementary apps (`services`). Majority of this onboarding document refers to the sections within `pipelines/matrix`

## Multi-Package Architecture

MATRIX is structured as a **monorepo with multiple packages** using **uv** for dependency management:

- **`pipelines/matrix/`** - Main pipeline development (where you'll spend most of your time)
  - Uses `pyproject.toml` and `uv.lock` for dependency management
  - Main `Makefile` for pipeline development commands
  - Contains all Kedro pipelines and core application code
- **`libs/`** - Modular libraries for shared functionality:
  - `matrix-auth/` - Authentication and environment utilities
  - `matrix-fabricator/` - Data fabrication and generation tools  
  - `matrix-gcp-datasets/` - GCP integration and Spark utilities
  - `matrix-mlflow-utils/` - MLflow integration and metric utilities
  - Each lib has its own `pyproject.toml` and is managed as a separate package
- **`infra/`** - Infrastructure as Code configuration
- **`docs/`** - Documentation generation (also has its own `pyproject.toml`)
- **`services/`** - Supporting services and APIs
- **`apps/`** - CLI tools and applications (each with separate `pyproject.toml`)

!!! warning "Important: Work in the Right Directory"
    **Always run pipeline development commands from the `pipelines/matrix/` directory.** This is where the main `Makefile` and `pyproject.toml` for the pipeline are located.

### uv Workspace Structure

The repository uses uv's workspace feature to manage multiple packages efficiently:

- **Root `pyproject.toml`**: Defines the workspace and shared configuration
- **Individual packages**: Each directory with a `pyproject.toml` is a separate package
- **Shared dependencies**: Common dependencies are managed at the workspace level
- **Local development**: Libraries in `libs/` are installed in editable mode for development

## Core Project Files
On a high-level, our pipeline repo management relies on the following files for environment configuration:

- **Multiple `Makefiles`** for different purposes:
  - `pipelines/matrix/Makefile` - **Main pipeline development (use this for most tasks)**
  - `infra/Makefile` - Infrastructure deployment  
  - `docs/Makefile` - Documentation generation
  - Service-specific Makefiles in `services/*/Makefile`

!!! Makefiles
    The `pipelines/matrix/Makefile` contains _very many_ commands, many are designed for GCP usage or facilitated debugging. You don't need to have an understanding of every command there - we will show you the most relevant ones over this guide.

- We also utilize `.env.defaults` for sharing default environment variables for running the project. This version-controlled file serves as both default configuration and documentation. For local environment variables, we use a git ignored copy of that file which is called `.env` - this way you can store your personal credentials and variables without risk of commiting them.

- **Dependencies are managed with `uv`**: Instead of `requirements.txt`, we now use `pyproject.toml` and `uv.lock` files to define Python dependencies. We use `uv` for fast & reproducible builds across all packages in the workspace. Each package manages its own dependencies, while shared dependencies are defined at the workspace level.

## Key Directories
### Source Code

As mentioned in our [tech stack overview](./tech_stack.md) our project is built using Kedro pipelines, which provide a modular and maintainable way to organize our data science workflows. These pipelines are organized into distinct stages that reflect our drug repurposing workflow.

Each pipeline is a collection of kedro nodes (Python functions) that are connected through data dependencies, making it easy to manage complex data processing tasks. The pipelines are organized into distinct stages that reflect our drug repurposing workflow.

Our codebase is mainly located in the following locations:

- `src/` - Contains the main application code
  - `matrix/` - Our core package with pipeline, data processing and modeling code. This is where all our Kedro pipelines can be located
    - `pipeline_registry.py` - Registers and configures all pipeline nodes and their connections; this is where we define larger stages such as `data_engineering`, `feature` etc.
    - `pipelines/` - Contains all modular pipeline stages:
        - `ingestion/` - Code for ingesting data from various biomedical knowledge graphs
        - `integration/` - Components for integrating different data sources
        - `data_release/` - Code for releasing the unified biomedical knowledge graph
        - `filtering/` - Custom filters for the knowledge graph
        - `embeddings/` - Generation of topological embeddings
        - `modelling/` - ML model training and evaluation
        - `matrix_generation/` - Production of the drug-disease prediction matrix
        - `evaluation/` - Model and prediction evaluation
  - `tests/` - Corresponding test files following pytest conventions
  - `settings.py` - Project-wide settings and configuration parameters for kedro pipelines.

Note that the files/directories listed above are not exhaustive and will likely evolve but these are the most important from drug repurposing perspective. There will also be some code outside `src/` directory - e.g. `scripts/` - which are helper scripts and utilities which are not part of the main pipeline.

### Data
We organize our data pipeline into clear directories stages which correspond to both the pipeline stage as well as *maturity* of the data:

- `data/`
  - `{pipeline_name}/` - Each pipeline has its own directory (e.g., `ingestion/`, `integration/`, etc.) as well as some of the following sub-directories
    - `raw/` - Raw input data before any processing
    - `int/` - Intermediate data produced by the pipeline
    - `prm/` - Primary data, cleaned and processed
    - `feat/` - Feature engineered data
    - `model/` - Trained model artifacts
    - `model_output/` - Model predictions and results
    - `reporting/` - Analysis reports and visualizations
  - `test/` - Fabricated data for quick development; reflects the same structure as above

Note that each pipeline produces different data products and so not all directories might be there - eg `ingestion` pipeline only ingests the KGs int our system so there will be no `model` directory.

This structure allows us to adhere to an organized structure for storing data products which is essential for a healthy project and kedro configuration.

### Configuration
We use Kedro's configuration system to manage our pipeline settings and data catalog. It's located within the `conf` directory: 

  - `base/` - Core configuration shared across all environments:
    - `catalog.yml` - Defines how data is loaded and saved for each pipeline stage
    - `parameters.yml` - Contains configurable parameters used across pipelines (e.g., model hyperparameters, filtering thresholds)

  - `cloud/` - Environment for cloud development on our GCP cluster. Overwrites base environment with `-e cloud` or `--environment cloud` flags.

  - `test/` - Test environment settings, typically using smaller datasets. Overwrites base environment with `-e test` or `--environment test` flags.
  
You will learn more about these different environments in the [overview section](./environments_overview.md). The most important thing to know is that the configuration system allows us to:

- Define data inputs/outputs for each pipeline stage in a centralized catalog

- Adjust parameters without changing code

!!! info
    For more details on Kedro's configuration system, including data catalog, check out the [Kedro documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments).

Now that you understand our codebase structure and configuration approach, let's get started with the setup!

[Next: Installation  :material-skip-next:](./installation.md){ .md-button .md-button--primary }