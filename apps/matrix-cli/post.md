---
draft: false
date: 2025-03-10
categories:
  - Release
authors:
  - app/github-actions
  - Siyan-Luo
  - JacquesVergine
  - pascalwhoop
  - piotrkan
  - alexeistepa
  - eKathleenCarter
  - emil-k
  - lvijnck
  - amyford
  - oliverw1
  - matentzn
---
# Matrix Platform `v0.4.0`:  Enhanced Release Management, Pipeline Optimization, and Expanded Infrastructure

This release significantly enhances the Matrix platform's automation, pipeline efficiency, and infrastructure capabilities.  Key improvements include automated release processes, optimized Kedro pipelines and catalog, extended infrastructure for security and accessibility, and more comprehensive documentation.

<!-- more -->

## Automated Release Management and Workflow

This release introduces major improvements in automating and streamlining the release process:

* **Automated Release Notes and Articles:** Release notes and articles are now automatically generated, eliminating manual effort and ensuring consistency.  Semantic versioning checks determine the necessity of release notes generation, preventing unnecessary steps for patch releases.
* **Scheduled Sampling Pipeline:** A new scheduled workflow automates the daily execution of the sampling pipeline, enabling consistent data generation for development and testing.
* **Configurable Release Bump Type:**  The pipeline submission workflow now accepts a `bump_type` input, enabling granular control over version increments during manual release triggers.  Slack notifications alert maintainers of submission failures.
* **Streamlined Pre-commit Hooks:**  Unused pre-commit checks are removed, simplifying the development workflow.
* **Consistent Catalog Resolution:** The CI workflow now resolves the data catalog for all environments (base, cloud, test), guaranteeing consistency during testing.  Authentication for GCP integration tests is also improved.
* **Corrupted KG Release Handling:**  A dedicated issue template simplifies the tracking and resolution of corrupted KG releases, streamlining the investigation and fix deployment process.

## Kedro Pipeline & Catalog Optimization

Several optimizations enhance the efficiency and functionality of the Kedro pipelines and catalog:

* **`kg_release` Pipeline Enhancement:** The pipeline now includes an `embeddings` step to capture potential errors early in the process.
* **Resource Optimization for Data Release:**  Resource requirements for node and edge ingestion are reduced for improved efficiency.  Re-inclusion of previously commented-out datasets (`spoke`, `ec_medical_team`, `ec_clinical_trails`) and addition of the `drugmech` dataset expands the knowledge base.
* **Expanded Evaluation Metrics:** The evaluation pipeline now includes additional metrics like `amin` and `amax`, as well as stability metrics based on overlapping pairs and ranking, providing a more comprehensive model assessment.
* **Unified Integration Layer:**  A new `GraphTransformer` class streamlines data ingestion and processing, simplifying the addition of new knowledge graphs.  The `transform` method now produces both nodes and edges, saved to BigQuery for efficient access.  Normalization and SemMedDB filtering are also enhanced.
* **Refactored Ingestion Pipeline:** Dynamically generated ingestion pipelines for each data source, along with `has_nodes` and `has_edges` flags, offer granular control over data intake. Drug and disease lists are read from GitHub release artifacts for improved version control.  The ground truth data intake process is updated, and drugmech pairs are generated within the pipeline.
* **Preprocessing Pipeline Refinements:** Improvements include cleaning and standardization of clinical trial data, normalization of EC medical nodes, and reporting to Google Sheets for enhanced visibility.
* **`embeddings` Pipeline Resource Adjustments:** Resource requirements are adjusted for PCA and topological PCA calculations, with lower test parameters for faster test runs.
* **`matrix_generation` Pipeline Enhancements:** Clinical trial data is incorporated into pair generation, and predictions now include rank and quantile rank columns.
* **New `GitHubReleaseCSVDataset`:** This new dataset type streamlines reading CSV files from GitHub releases, improving version control for data sources.
* **Enhanced `SparkDatasetWithBQExternalTable`:**  This dataset now sets labels on BigQuery tables and increases save timeouts.
* **Updated `NodeNorm` Endpoint Version:** The `releases_aggregated.yaml` file is updated with the latest version of the `NodeNorm` endpoint.
* **Improved Null Handling and Data Quality:** Null handling in SemMedDB filtering is improved, enhancing data quality.
* **New BigQuery Tables for Ingested Data:** New tables (`nodes_ingested`, `edges_ingested`) store integrated knowledge graph data.


## Infrastructure Enhancements: Security, Accessibility, and Monitoring

This release focuses on improving security, accessibility, and monitoring of the infrastructure:

* **Enhanced Documentation:** Expanded documentation covers debugging memory issues, modeller workbenches, runbooks for various tasks (data catalog usage, local Argo workflows, Kedro experiments, OAuth clients, release notes generation), and programmatic access to MLFlow through IAP.
* **Streamlined Secret Management:**  Documentation clarifies the use of Secret Manager and git-crypt for enhanced security.
* **Improved Kubernetes Access Control:** RBAC using Google Groups for Argo workflows enhances security and access management.
* **Secure Public Data Zone Access:** HTTPS access to the public data zone ensures secure and performant data sharing.
* **Grafana and Prometheus Deployment:** Deployment of these monitoring tools provides enhanced observability of the cluster and experiment runs, with updated documentation on access and configuration.
* **Vertex AI Workbenches for Modelling:**  Post-startup scripts ensure all dependencies are installed in the modelling workbenches.
* **Improved IAM Roles and Storage Access:** Updated IAM roles and a dedicated storage viewer service account provide controlled access to storage buckets across projects.


## Miscellaneous Improvements

This release also includes several miscellaneous improvements:

* **Improved Contributing Guidelines:**  Documentation on code review best practices and the use of AI for code generation is added.
* **Enhanced Error Handling:** Specific error handling for missing PR details improves robustness.
* **Extended `matrix-cli` Functionality:** The CLI now supports bulk addition of GitHub users to teams.
* **Updated Python Dependencies:** Several packages are updated to newer versions.
* **Improved JSON Blob Handling in CI:** Fixes issues with JSON blob rendering and improves security in CI.
* **Updated Disease List and Release List Name:** The disease list version is updated, and a naming issue is resolved.
* **Replacement of `git-crypt`:**  `git-crypt` is replaced with a script for improved security and infrastructure management.
* **Data Quality Checks for Evidence.dev:** Data quality checks and dashboards on Evidence.dev are improved.
* **Miscellaneous Bug Fixes and Enhancements:** Several bug fixes and enhancements address issues related to schema checks, Makefile targets, CLI improvements, MLFlow dataset logging, Argo monitoring, documentation links, SILC configuration, and more.
* **MLFlow Run Archiving:** Functionality to archive old runs and experiments improves MLFlow management.
* **Optimized Resource Usage for Neo4j Ingestion:**  Resource requirements for edge and node ingestion into Neo4j are reduced.


These updates significantly improve the Matrix platform's automation, pipeline efficiency, infrastructure management, security, and overall developer experience. The improved documentation and streamlined workflows empower contributors to more effectively develop, test, and deploy models.
