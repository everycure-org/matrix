---
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
  - matwasilewski
  - jdr0887
  - Siyan-Luo
  - MariaHei
  - oliverw1
  - elliottsharp
  - app/github-actions
  - pascalwhoop
  - piotrkan
  - matej-macak
---
# `v0.2.7`:  K-Fold Cross-Validation, Enhanced Pipeline Control, and Automated Releases

This release of the Matrix Platform introduces major enhancements to model evaluation, pipeline management, and automation.  K-fold cross-validation significantly improves model robustness, dynamic pipeline options provide greater flexibility, and automated release workflows streamline the release process.

<!-- more -->

## Enhanced Model Evaluation and Pipeline Control

This release brings significant improvements to model evaluation and pipeline management:

* **K-fold Cross-Validation (#683):** Model evaluation robustness is substantially improved with the integration of k-fold cross-validation. This generates multiple train-test splits, providing a more comprehensive assessment of model performance and generalization ability.

* **Flexible Ensemble Model Aggregation (#905):** Users can now specify the aggregation function for ensemble models, providing greater control over how individual model predictions are combined.  This enhances performance customization and is documented in the updated documentation.

* **Overridable Dynamic Pipeline Options (#901):**  Dynamic pipeline options can now be overridden using environment variables, offering greater flexibility in configuring pipelines.

* **Single Model per Modelling Pipeline (#924):** Ensuring only one model is trained per modelling pipeline run improves pipeline management and resource allocation.

## Automated Release Workflows and Infrastructure Enhancements

Several key enhancements automate release workflows and improve infrastructure:

* **Automated Kedro Release Submission (#877):** A new GitHub Actions workflow automates the periodic submission of Kedro pipelines for releases, streamlining the release process.

* **Grafana and Prometheus Deployment (#834):**  The deployment of Grafana and Prometheus provides enhanced observability and monitoring for the cluster and experiment runs, allowing for better performance tracking and issue identification.

* **Simplified Neo4J SSL Setup (#878):** The Neo4j SSL configuration is now simplified, improving setup ease and security.

* **Updated Neo4j Connection String (#880):**  Updating the Neo4j connection string in templates ensures consistency and clarity across the platform.

* **`ArgoNode` Class Refactoring (#885):** Refactoring the `argo_node` function into the `ArgoNode` class improves code structure, organization, and maintainability.

## Bug Fixes and Stability Improvements

Several bug fixes enhance platform stability and reliability:

* **`wipe_neo` Script Fix (#899):** Correcting the protocol used in the `wipe_neo` script improves Neo4j interaction and resolves potential issues.

* **Kedro Hooks Test Coverage (#900):** Increased test coverage for Kedro hooks enhances release stability and reliability.

* **Schema Error Fix (#943):** Handling null values in the `ingest_nodes` function's `name` column resolves a schema error and ensures data integrity.

* **Release Branching Fix (#950):**  Correcting an issue with release branch creation from the wrong commit ensures consistency in release management.

* **Argo Release Trigger Fix (#936):**  Addressing an issue with untriggered data releases ensures proper workflow execution.

* **Release Tag Check (#983):**  Preventing accidental overwrites of existing releases by checking for existing tags enhances data integrity.

* **Git SHA Retrieval Fix (#974):** Using the correct Git command for SHA retrieval resolves inconsistencies and ensures accurate version tracking.

* **GitHub Token Access Fix (#990):** Resolving a GitHub Actions workflow issue ensures proper access to the GitHub token.

## Documentation and Onboarding Enhancements

Numerous documentation updates improve onboarding and provide clearer guidance:

* **Updated Release Creation Docs (#940):** Clearer instructions simplify the release process.

* **Kedro Resources Documentation (#919):** Added documentation enhances understanding and usage of Kedro resources.

* **Onboarding Fixes (#902, #956):**  Improvements to onboarding documentation and materials streamline the process for new contributors.

* **Updated Virtual Environment Docs (#906):**  Clearer virtual environment setup instructions simplify the development process.

* **Kedro Extensions Typos Fixed (#913):** Correcting typos improves documentation quality.

* **Updated Java Version in Docs (#903):** Specifying Java 17 as the required version prevents compatibility issues.

* **Added `libomp` Library to Installation Docs (#934):** Including the `libomp` library ensures all dependencies are installed.

* **Onboarding Improvements and New Keys (#883, #886, #904, #944):** Onboarding documentation updates and the addition of keys for new team members improve collaboration.

* **Updated Git-Crypt Instructions:**  The documentation now clarifies the correct path for adding GPG keys, preventing confusion during onboarding.  New users should now submit their GPG keys via a Pull Request for better security and tracking.

## Other Notable Changes

Other improvements enhance code quality, testing, and pipeline functionality:

* **Removal of `refit` Library (#811):** Simplifying dependencies and reducing maintenance overhead.

* **Consistent `pyspark.sql` Usage (#923):** Improving code consistency and clarity.

* **`object` Keyword Rename (#922):** Improving code readability and avoiding conflicts.

* **Refactored Test Suite (#930):** Improving test structure and maintainability.

* **Added Import Sorting (#931):** Enhancing code readability and consistency.

* **Modelling Cleanup - Unified Splits (#907):** Improving code clarity and consistency in the modelling process.

* **Release Branch Name Validation (#921):** Adding validation for branch names ensures consistency in release management.

* **Improved CI Checks (#967):**  Resolving CI issues ensures smooth integration and testing.

* **Disabled SSL for Local Neo4j (#972):** Simplifying local development setup.

* **Improved Git Checks for Kedro Submit (#961):** Preventing invalid submissions enhances code quality.

* **Added Changelog Integration Tests (#968):** Ensuring correct changelog generation.

* **Intermediate Release Support (#957):** Enabling creation of intermediate releases without marking them as latest.

* **Faster Data Release Testing (#989):**  Enabling faster testing improves development efficiency.

* **Enhanced CLI Markdown Generation (#959):** Improving metadata for release notes and articles.

This release significantly enhances the Matrix Platform through improved model evaluation, enhanced pipeline control, and streamlined automation.  The introduction of k-fold cross-validation, flexible ensemble aggregation, and automated releases strengthens the platform's robustness and efficiency, while various bug fixes and documentation improvements enhance stability and usability.
