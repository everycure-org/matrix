# MATRIX Kedro Project

This directory is a codebase for data processing and ML modelling for our drug repurposing platform.
Our pipeline is built using [Kedro documentation](https://docs.kedro.org) and deployed using
[Hera workflows](https://hera-workflows.readthedocs.io/) for Argo Workflows orchestration on
Kubernetes.

## Architecture Overview

The MATRIX pipeline consists of:

- **Kedro pipelines** for data processing and ML workflows
- **Hera workflows** for generating Argo Workflows that orchestrate pipeline execution on Kubernetes  
- **uv workspace** for managing dependencies across the entire monorepo
- **Modular libraries** (from `/libs/`) for shared functionality across the platform:
  - `matrix-auth`: Authentication and environment utilities
  - `matrix-fabricator`: Data fabrication and generation tools
  - `matrix-gcp-datasets`: GCP integration and Spark utilities  
  - `matrix-mlflow-utils`: MLflow integration and metric utilities

### 🔧 Multi-Package Development

This pipeline is part of a larger uv workspace that includes multiple related packages. The workspace automatically:
- Links local libraries from `/libs/` for development
- Manages shared dependencies across all packages
- Provides fast, reproducible builds with `uv sync`

## How to install dependencies & run the pipeline?

More detailed instructions on setting up your MATRIX project can be found here however below we also
provide quick & simple instructions for Mac OS (instructions for other OS can be found within
documentation).

### Prerequisites

Install the following dependencies required for our pipeline:

```bash
# Install pyenv - for python version management
brew install pyenv
pyenv install 3.11

# Install uv - for library management
brew install uv python@3.11

# Install docker - required for Neo4j set-up
brew install --cask docker #installs docker desktop
brew install docker docker-compose #installs CLI commands

# Install Java - required for PySpark operations
brew install openjdk@17
brew link --overwrite openjdk@17
```

### Installation

> **Important**: This repository contains multiple Makefiles for different purposes:
>
> - `pipelines/matrix/Makefile` - Main pipeline development (use this for most tasks)
> - `infra/Makefile` - Infrastructure deployment
> - `docs/Makefile` - Documentation generation
> - Service-specific Makefiles in `services/*/Makefile`

**For pipeline development, run all commands from the `pipelines/matrix/` directory:**

1. Navigate to the pipeline directory and install dependencies:

```bash
cd pipelines/matrix/
make install
```

2. Or manually using uv:

```bash
cd pipelines/matrix/
uv sync
```

3. Verify installation:

```bash
cd pipelines/matrix/
make fast_test
```

Dependencies are declared in `pyproject.toml` and managed using uv package manager. The uv workspace automatically handles local dependencies from the `/libs/` directory.

**Common Commands (run from `pipelines/matrix/`):**

- `make install` - Install all dependencies and pre-commit hooks
- `make fast_test` - Run fast unit tests
- `make integration_test` - Run full integration tests
- `make clean` - Clean caches and rebuild environment
- `make fetch_secrets` - Fetch required GCP secrets

## Running Workflows

The MATRIX pipeline uses Hera to generate and submit Argo Workflows to Kubernetes. The workflow
system:

- Converts Kedro pipelines into Argo Workflows using Hera's Pythonic API
- Supports resource configuration for CPU, memory, and GPU requirements
- Integrates with Neo4j, MLflow, and GCP services
- Provides automatic dependency resolution between pipeline nodes

To run a workflow (from `pipelines/matrix/` directory):

```bash
# Run a specific pipeline locally
kedro run --pipeline <pipeline_name>

# Submit workflow to Argo (requires cluster access and authentication)
python -m matrix.cli_commands.experiment submit --pipeline <pipeline_name>

# Alternative using uv
uv run kedro run --pipeline <pipeline_name>
```

### Workflow Generation Architecture

The new Hera-based workflow system (`src/matrix/argo_hera.py`) has replaced the previous Jinja
templating approach and provides:

#### Key Components:

- **WorkflowConfig**: Central configuration object containing image, resources, environment settings
- **EnvironmentBuilder**: Builds environment variables for different container types (standard and
  Neo4j)
- **TemplateBuilder**: Creates Hera Container templates for different execution contexts
- **TaskBuilder**: Converts fused pipeline nodes into Argo Workflow tasks
- **ResourceConverter**: Handles resource configuration translation to Kubernetes format

#### Pipeline Fusion System:

The system uses an intelligent fusion algorithm (`src/matrix/fuse.py`) that:

- Groups Kedro nodes with compatible tags (`argowf.fuse` and `argowf.fuse-group.*`)
- Maintains execution dependencies while minimizing workflow steps
- Preserves resource configurations during fusion
- Supports both standard and Neo4j execution environments
- Creates `FusedNode` objects that combine multiple Kedro nodes into single Workflow tasks
- Uses topological sorting to ensure correct execution order during fusion

#### Workflow Structure:

1. **Direct Workflow Generation**: Creates Argo Workflows directly using Hera's Pythonic API (no
   more WorkflowTemplates)
2. **Container Templates**: Two main templates - standard Kedro tasks and Neo4j-enabled tasks with
   sidecars
3. **Resource Management**: Configurable CPU, memory, and GPU allocation per task
4. **Environment Integration**: Automatic injection of secrets, config maps, and service endpoints
5. **DAG-based Execution**: Tasks are organized in a Directed Acyclic Graph (DAG) with automatic
   dependency resolution
6. **Resource-aware Fusion**: Nodes with different resource requirements can be fused while
   preserving individual resource configurations

#### Migration from Jinja Templates

The previous workflow generation system used Jinja templating (`templates/argo_wf_spec.tmpl`) to
generate WorkflowTemplates. The new Hera-based system provides several advantages:

- **Type Safety**: Pythonic API with compile-time error checking
- **Direct Submission**: Generates Workflows directly instead of WorkflowTemplates
- **Better Integration**: Native integration with Python-based configuration
- **Maintainability**: Easier to extend and modify workflow generation logic

## Rules and guidelines

In order to get the best out of the kedro project template:

- Don't remove any lines from the `.gitignore` file we provide
- Make sure your results can be reproduced by following a
  [data engineering convention](https://docs.kedro.org/en/stable/faq/faq.html#what-is-data-engineering-convention)
- Don't commit data to your repository
- Don't commit any credentials or your local configuration to your repository. Keep all your
  credentials and local configuration in `conf/local/`

## How to test your Kedro project

Have a look at the files `src/tests/test_run.py` and
`src/tests/pipelines/data_science/test_pipeline.py` for instructions on how to write your tests. Run
the tests as follows:

```
pytest
```

To configure the coverage threshold, look at the `.coveragerc` file.

## Project dependencies

Dependencies are managed through `pyproject.toml` files in the uv workspace:
- **Pipeline dependencies**: Defined in `pipelines/matrix/pyproject.toml`
- **Local libraries**: Automatically installed from `/libs/` in editable mode
- **Workspace dependencies**: Shared dependencies managed at the root level

Install all dependencies with `uv sync` or use the Makefile:
```bash
make install  # Recommended: handles workspace sync + pre-commit setup
# OR
uv sync       # Direct uv command
```

[Further information about project dependencies](https://docs.kedro.org/en/stable/kedro_project_setup/dependencies.html#project-specific-dependencies)

## How to work with Kedro and notebooks

> Note: Using `kedro jupyter` or `kedro ipython` to run your notebook provides these variables in
> scope: `catalog`, `context`, `pipelines` and `session`.
>
> Jupyter, JupyterLab, and IPython are already included in the project requirements by default, so
> once you have run `uv sync` you will not need to take any extra steps before you use them.

### Jupyter

Jupyter is already included in the project dependencies. You can start a local notebook server:

```
kedro jupyter notebook
```

### JupyterLab

JupyterLab is already included in the project dependencies. You can start JupyterLab:

```
kedro jupyter lab
```

### IPython

And if you want to run an IPython session:

```
kedro ipython
```

### How to ignore notebook output cells in `git`

To automatically strip out all output cell contents before committing to `git`, you can use tools
like [`nbstripout`](https://github.com/kynan/nbstripout). For example, you can add a hook in
`.git/config` with `nbstripout --install`. This will run `nbstripout` before anything is committed
to `git`.

> _Note:_ Your output cells will be retained locally.

[Further information about using notebooks for experiments within Kedro projects](https://docs.kedro.org/en/develop/notebooks_and_ipython/kedro_and_notebooks.html).

## Package your Kedro project

[Further information about building project documentation and packaging your project](https://docs.kedro.org/en/stable/tutorial/package_a_project.html).
