```yaml
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - JacquesVergine
  - emil-k
  - marcello-deluca
  - lvijnck
  - alexeistepa
  - jdr0887
  - matwasilewski
  - Siyan-Luo
  - oliverw1
  - MariaHei
  - app/github-actions
  - elliottsharp
  - pascalwhoop
  - piotrkan
  - matej-macak
```

# `v0.2.7`:  K-Fold Cross-Validation, Enhanced Argo Workflows, and Improved Data Release Management

This release introduces significant improvements to the Matrix Platform, focusing on enhanced model evaluation, streamlined workflows, and robust data release management.  Key features include the implementation of k-fold cross-validation, improved Argo workflow integration, automated periodic data releases, and numerous bug fixes and technical enhancements.

<!-- more -->

## Enhanced Model Evaluation with K-Fold Cross-Validation

A major improvement in this release is the introduction of k-fold cross-validation (#683). This feature significantly enhances model evaluation robustness by generating multiple train-test splits, providing a more comprehensive and reliable assessment of model performance.  This enables more rigorous model comparison and selection, ultimately leading to more confident predictions.

## Streamlined Workflows and Automation

Several enhancements streamline workflows and automate key processes:

* **Automated Periodic Data Releases (#877):** Automating data releases ensures data consistency and reduces manual effort. This automation improves the reliability and frequency of data updates.

* **Argo Workflow Enhancements (#936, #885):**  Adding trigger release labels to Argo workflows provides better visibility into release triggers and improves workflow monitoring. Refactoring the `argo_node` function into the `ArgoNode` class improves code organization and maintainability.  Furthermore, the `ArgoNode` now supports specifying resource requests for CPU, memory, and GPU, optimizing resource utilization on Kubernetes.

* **Dynamic Pipeline Options and Resolver (#901):**  Overriding dynamic pipeline options and loading settings from the catalog provides greater flexibility and control over pipeline configuration.

* **Modelling Pipeline Enhancements (#907, #924):** Unifying split generation and enforcing one model per modelling pipeline improves the modelling process' clarity and consistency.

* **Release Management Improvements (#921, #950, #974):** Ensuring release branches originate from the correct commit and verifying branch names start with "release" enhances release management consistency and reliability.  Fetching the Git SHA using the correct command improves reliability.

## Infrastructure and Dependency Management

This release includes several improvements to infrastructure and dependency management:

* **Simplified Neo4j SSL Setup (#878, #880, #972, #977):**  Simplified SSL configuration, updated connection strings, and disabling SSL for local deployments streamline Neo4j setup and management.  Fixing resource allocation in the Neo4j Docker Compose template enhances performance.

* **Removal of `refit` Library (#811):** Removing the `refit` library simplifies dependencies and reduces maintenance overhead.

* **Consistent `pyspark.sql` Usage (#923):** Consistent use of `pyspark.sql` improves code clarity and maintainability.

* **Batch Pipeline and Preprocessing Improvements (#766):** Improvements to the batch pipeline and simplified preprocessing enhance efficiency and code clarity.

* **Java Version Upgrade (#903):** Upgrading the required Java version to 17 addresses potential compatibility issues.

## Enhanced Documentation and Onboarding

Several improvements to documentation and onboarding make it easier for new users to get started and for existing users to find information:

* **Expanded Kedro Documentation (#919, #902, #906):** Adding documentation for Kedro resources, virtual environment setup, and general onboarding improves the developer experience.

* **Updated Release Documentation (#940):** Updating the release creation documentation streamlines the release process.

* **Common Errors Documentation Update (#925):**  Updating the common errors document provides solutions to new and recurring problems.

* **`libomp` Installation Documentation (#934):**  Adding the `libomp` library to the installation documentation ensures all necessary dependencies are installed.

* **Improved Onboarding Process (#883, #886, #904, #944):**  Adding git-crypt keys for new team members simplifies onboarding and secure key management.  Moving key sharing to Pull Requests improves security.

## Bug Fixes and Stability Improvements

Numerous bug fixes enhance platform stability and reliability:

* **`wipe_neo` Script Fix (#899):** Correcting the protocol used by the `wipe_neo` script improves Neo4j interaction.

* **Kedro Hooks Test Coverage (#900):**  Increased test coverage for Kedro hooks enhances release stability.

* **Schema Error Fix (#943):**  Handling null values in the `ingest_nodes` function resolves a schema error.

* **Release Branching Fix (#950):**  Ensuring release branches are created from the correct commit improves release management consistency.

* **CI Check Fixes (#967):** Resolving issues that prevented CI checks from passing on the main branch improves integration and testing.

* **Git SHA Fetch Fix (#974):**  Fetching the Git SHA using the correct command ensures reliability.

* **Neo4j Resource Allocation Fix (#977):**  Fixing resource allocation in the Neo4j template improves performance.

* **GitHub Token Access Fix (#990):** Resolving GitHub token access issues in the workflow ensures proper functionality.


## Other Notable Changes

* **Node Normalizer Improvements (#972):** The new `NodeNormalizer` with multiple API calls improves data cleaning and normalization. The refactoring of normalization logic improves code organization.  Code refactoring ensures compatibility with the latest Kedro version.  Release logic refactoring improves code organization and modularity.

* **Changelog Generation Tests (#968):**  Adding integration tests for changelog generation improves reliability.

* **Matrix CLI Enhancements (#959):** Including mkdocs frontmatter in the Matrix CLI improves release notes and articles.

* **Testing Enhancements (#989, #961):** Adding a test data release workflow to the main branch simplifies testing. Expanding Git checks before data release submission enhances quality control.


This release represents a significant step forward in the Matrix Platform's development, enhancing model evaluation, streamlining workflows, and improving data management. The introduction of k-fold cross-validation and the numerous bug fixes and technical enhancements contribute to a more robust and reliable platform for drug repurposing research.
