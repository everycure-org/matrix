---
draft: false
date: 2025-02-11
categories:
  - Release
authors:
  - eKathleenCarter
  - lvijnck
  - Siyan-Luo
  - emil-k
  - JacquesVergine
  - alexeistepa
  - pascalwhoop
  - piotrkan
  - oliverw1
  - amyford
---
# Matrix `v0.x.x` Release: Enhanced CLI, Scheduled Sampling, and Extensive Bug Fixes

This release of the Matrix Platform focuses on improving the command-line interface, introducing scheduled data sampling, resolving several key bugs across the platform, and implementing various technical enhancements for improved performance and maintainability.

<!-- more -->

## Key Enhancements

### Enhanced Command-Line Interface (CLI)

The Matrix CLI now supports adding users to multiple teams simultaneously via the `matrix gh-users add` command. This simplifies user management and improves efficiency for administrators.  This enhancement complements existing CLI functionalities for GitHub user and team management, release note generation, and code change summarization.

### Scheduled Data Sampling

A new scheduled sampling pipeline (`scheduled-sampling-pipeline.yml`) has been implemented. This pipeline executes daily at 5am GMT using a dedicated GitHub action, automating the collection of data samples and enabling ongoing monitoring and analysis.  This new feature facilitates data exploration and supports various experimental workflows.  A new experiment is underway to evaluate the efficacy and performance of this scheduled sampling approach.

### Extensive Bug Fixes

This release addresses numerous bugs across various components of the platform, significantly improving stability and reliability:

- **Clinical Trial Preprocessing:**  Resolved issues impacting the correct functionality of clinical trial preprocessing nodes, ensuring accurate data preparation.
- **Data Normalization:**  Fixed a bug where the normalizer consistently returned `normalization_success=True`, regardless of the actual outcome.  This fix ensures accurate reporting of normalization success.
- **Argo Workflow Linting:**  Corrected an issue that prevented proper linting of Argo templates, improving workflow definition and validation.
- **Subprocess Deadlocks:**  Resolved a deadlock issue in subprocess calls, improving the robustness and reliability of various pipeline operations.
- **Release Note Generation:**  Fixed a bug in the release notes generation process related to the scope of information used, ensuring accurate and comprehensive release notes.
- **MLflow Metric Tracking:**  Resolved an issue that prevented proper MLflow metric tracking, restoring the ability to monitor experiment progress and results effectively.
- **EC Medical Node Processing:**  Fixed an issue in the preprocessing pipeline that prevented correct processing of EC medical nodes.
- **GitHub Action Runner Token Interpretation:** Addressed a bug in the GitHub action runner where tokens within JSON blobs were misinterpreted, improving the reliability of GitHub action workflows.
- **BigQuery Table Updates:**  Fixed a bug that prevented correct updates to existing BigQuery tables, ensuring data consistency.
- **Makefile Target:** Added a missing Makefile target, enhancing build process automation and flexibility.
- **Broken Documentation Link:**  Fixed a broken link in the documentation, improving the user experience and ensuring access to relevant information.
- **Preprocessing Pipeline Schema Check:** Resolved a schema check issue in the preprocessing pipeline, ensuring data integrity and consistency.
- **GT Version Issue in Ingestion Catalog:** Fixed a version issue for GT in the ingestion catalog, ensuring the use of the correct data version.
- **Integration Pipeline Interpolation Key:**  Resolved an error in the integration pipeline caused by a missing interpolation key, restoring pipeline functionality.


## Technical Enhancements

Several technical enhancements improve the platform's performance, maintainability, and usability:

- **Improved CLI:**  The implementation of the `matrix gh-users add` command streamlines user management within the CLI.
- **Enhanced Data Ingestion:** Leveraging BigQuery for data ingestion improves efficiency and scalability.
- **Code Refactoring:** Refactoring efforts across various modules improve code readability and maintainability.
- **Improved Release Management:**  Refined the logic for generating release notes and articles, ensuring they are only generated for significant version changes.  Also improved the logic for determining the latest minor release.
- **Robust Subprocess Handling:** Improved the robustness of subprocess calls by addressing deadlock issues.
- **Enhanced Docker Image Generation:** Streamlined the docker image generation pipeline for improved efficiency.
- **Improved Local Argo Workflow Handling:** Enhanced the way Argo workflows are handled locally, simplifying development and testing.
- **Secrets Management:** Replaced `git-crypt` with a script for managing secrets within the repository, improving security.  This change primarily affects system administrators; most users now utilize Secret Manager.
- **Default Resource Allocation:** Changed the default resources allocated for Argo workflows and ArgoNodes, optimizing resource utilization.


## Documentation Improvements

Extensive updates to the documentation enhance clarity and usability:

- **Onboarding:**  Updated onboarding documentation to include container registry authentication, simplifying the setup process for new users.
- **FAQ:** Added a Frequently Asked Questions (FAQ) section for common errors, providing quick solutions to common problems.
- **AI-Assisted Documentation Generation:**  Added a new runbook for generating notes and articles using AI, streamlining documentation creation.
- **Local Argo Workflows:** Added a runbook for setting up Argo workflows locally, facilitating development and testing.
- **Public Data Zone Infrastructure:** Documented the Public Data Zone infrastructure, providing information on data access and usage.
- **DNS Configuration:**  Added information about DNS configuration, improving clarity for infrastructure setup and management.
- **Secrets Management:**  Updated the `git-crypt` documentation to reflect the transition to Secret Manager.
- **Compute Cluster (Kubernetes Cluster):** Updated and renamed the documentation for the Compute Cluster, which is now the Kubernetes Cluster, reflecting changes in infrastructure.
- **Observability Stack:** Added documentation for the observability stack, providing information on monitoring and logging.


## Other Changes

Several other improvements enhance code quality, testing, and data management:

- **Dependency Updates:**  Updated project dependencies to their latest versions, ensuring compatibility and leveraging the latest features.
- **Unit Tests:**  Added unit tests to improve code reliability and facilitate regression testing.
- **Error Handling:**  Improved error handling across the platform for better robustness and user experience.
- **Configuration Files:** Added new configuration files to separate concerns and improve organization.
- **Modular Pipelines:** Added several new pipelines to handle different data sources, supporting a more modular pipeline structure.
- **Flexible Data Management:** Added new datasets for more flexible data management within pipelines.
- **Clinical Trial Data:** Added functionality to resolve names to CURIEs for source and target columns in clinical trials data, improving data integration and consistency.
- **Data Release Pipeline:** Updated the way the data release pipeline handles SemMed filtering, refining data processing.


This release represents a significant step forward in terms of platform stability, usability, and maintainability.  The numerous bug fixes, technical enhancements, and documentation improvements contribute to a more robust and user-friendly experience for both developers and users.  The introduction of scheduled data sampling and enhanced CLI functionalities further expands the platform's capabilities and supports ongoing research and development efforts.
