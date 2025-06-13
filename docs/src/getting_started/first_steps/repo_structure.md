# Important Files and Folders

After getting familiar with the high-level overview of MATRIX, let's explore the key files and folders in our codebase. Understanding these will help you navigate and contribute effectively.

## Core Project Files

### Environment Configuration
- `.env.defaults`: Contains shared default environment variables and documentation. This version-controlled file serves as both default configuration and documentation.
- `.env`: Your local environment variables and credentials file. This is gitignored to protect sensitive information.

### Project Management
- `Makefile`: Contains standardized commands for common operations like building, testing, and deployment. Run `make help` to see available commands. [Learn more about Makefiles](https://makefiletutorial.com/)
- `requirements.txt` and related files: Define Python dependencies, managed through `uv` for reproducible builds.

## Key Directories

### Source Code
- `src/`: Contains the main application code
  - `matrix/`: Our core package with pipeline, data processing and modeling code
  - `tests/`: Corresponding test files following pytest conventions

### Data
- `data/`: Stores datasets organized by processing stage:
  - `01_raw/`: Original unprocessed data
  - `02_intermediate/`: Data after initial processing
  - `03_primary/`: Clean, transformed data ready for modeling
  - `04_feature/`: Engineered features
  - `05_model_input/`: Prepared data for model training
  - `06_models/`: Trained models
  - `07_model_output/`: Model predictions and evaluations
  - `08_reporting/`: Analysis results and metrics

### Configuration
- `conf/`: Contains environment-specific configurations
  - `base/`: Default settings shared across environments
  - `local/`: Local development settings
  - `test/`: Test environment configuration
  - Learn more in the [Kedro documentation](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments)

### Supporting Files
- `scripts/`: Helper scripts and utilities not part of the main pipeline
- `docs/`: Project documentation (you're reading it!)
- `notebooks/`: Jupyter notebooks for exploration and analysis

This structure follows standard Python project conventions while incorporating Kedro-specific organization for data science workflows. As you continue through this documentation, you'll learn how these components work together in our pipeline.

Now that you have a good understanding of our pipeline, kedro, and repo structure, let's get started with the set-up!

[Next: Installation :material-arrow-right:](./installation.md){ .md-button .md-button--primary }