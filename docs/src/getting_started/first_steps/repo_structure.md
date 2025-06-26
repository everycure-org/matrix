# Repository Structure

After getting familiar with the high-level overview of MATRIX and kedro, let's explore the key files and folders in our codebase. Understanding these will help you navigate the repository more effectively.

!!! info
    As MATRIX is a monorepo, it contains codebase for the data science platform (`pipelines/matrix`) but also infrastructure (`infra`) , documentation (`docs`), and supplementary apps (`services`). Majority of this onboarding document refers to the sections within `pipelines/matrix`

## Core Project Files
On a high-level, our pipeline repo management relies on the following files for environment configuration:

- The most useful commands (such as creating and setting-up `.venv`, running pre-commits or tests) are conveniently handles by `Makefiles`. `Makefile` contains standardized commands for common operations; to see available clommands, run: `make help`. [You can learn more about Makefiles here](https://makefiletutorial.com/)

!!! Makefiles
    Note that our `Makefile` script contains _very many_ commands, many are designed for GCP usage or facilitated debugging. You don't need to have an understanding of of every command there - we will show you the most relevant ones over this guide.

- We also utilize `.env.defaults` for sharing default environment variables for running the project. This version-controlled file serves as both default configuration and documentation. For local environment variables, we use a git ignored copy of that file which is called `.env` - this way you can store your personal credentials and variables without risk of commiting them.

- Lastly, the environment set-up relies on `requirements.txt` and related files - these define Python dependencies, we manage them through `uv` for quick & reproducible builds.

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

This structure allows us to adhere to an organized structure for storing data products which is essnetial for a healthy project and kedro configuration.

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