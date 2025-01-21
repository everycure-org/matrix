---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - JacquesVergine
  - emil-k
  - lvijnck
  - marcello-deluca
  - alexeistepa
  - jdr0887
  - matwasilewski
  - Siyan-Luo
  - MariaHei
  - oliverw1
  - app/github-actions
  - elliottsharp
  - pascalwhoop
  - piotrkan
  - matej-macak
---
# `v0.2.7`: Enhanced Pipeline Flexibility, Automated Releases, and Refined Development Workflow

This release marks a significant advancement in the Matrix Platform, focusing on pipeline flexibility, automated data releases, and a more refined development workflow.  Key improvements include k-fold cross-validation for robust model evaluation, flexible ensemble aggregation, periodic Kedro pipeline submission for automated data releases, and substantial refactoring for improved code maintainability and performance.

<!-- more -->

## Enhanced Model Evaluation and Pipeline Automation

* **K-fold Cross-Validation (#683):**  Model evaluation robustness is significantly improved with the implementation of k-fold cross-validation. This generates multiple train-test splits, providing a more comprehensive assessment of model performance and generalization ability.

* **Flexible Ensemble Aggregation (#905):**  The introduction of a flexible aggregation function parameter allows users to define how predictions from individual models in an ensemble are combined.  This provides greater control over model behavior and allows for customization based on specific use cases. The documentation has been updated with details on how to leverage this functionality.

* **Periodic Kedro Pipeline Submission (#877):** Automates data release runs, streamlining the release process and reducing manual intervention.  This automation ensures consistent and timely data releases.

## Bug Fixes and Stability Enhancements

Several bug fixes contribute to a more stable and reliable platform:

* **`wipe_neo` Script Protocol Fix (#899):** Corrects the protocol used by the `wipe_neo` script, ensuring proper interaction with the Neo4j database.

* **Improved Kedro Hooks Test Coverage (#900):** Enhanced test coverage for Kedro hooks increases the stability and reliability of the release process.

* **Schema Error Fix in `ingest_nodes` (#943):**  Resolves a schema error in the `ingest_nodes` function, ensuring data integrity.

* **Release Branching Fix (#950):**  Fixes an issue where release branches were created from incorrect commits, ensuring consistency in release management.

* **CI Fixes (#967):**  Resolves issues with CI checks, ensuring that the main branch remains in a consistently buildable state.

* **GitHub Action Fix (#990):** Fixes a missing GitHub token access issue in the workflow, enabling proper functionality of GitHub actions.

* **Git SHA Fix (#974):** Corrects the command used to retrieve the Git SHA, ensuring accurate version tracking.

## Technical Enhancements and Refactoring

This release includes significant technical enhancements and refactoring efforts:

* **Removal of `refit` Library (#811):**  Simplifies the codebase and dependency management by removing the `refit` library.

* **Simplified Neo4j SSL Setup (#878):**  Streamlines the Neo4j SSL configuration process, improving security and ease of setup. The documentation has been updated to reflect these changes.

* **`argo_node` Refactoring (#885):**  Improves code structure and readability by refactoring the `argo_node` function.

* **Consistent `pyspark.sql` Usage (#923):**  Enforces consistent usage of `pyspark.sql` for improved code clarity and maintainability.

* **Refactored Test Suite (#930):**  Enhances the structure and maintainability of the test suite, making it easier to maintain and extend test coverage.

* **Added Import Sorting (#931):**  Improves code readability and consistency by enforcing import sorting.

* **Added Release Trigger Label to Argo (#936):**  Adds labels to Argo workflows to indicate data release triggers, improving monitoring and traceability.

* **Improved Modelling Process (#907):**  Unifies split generation in the modelling process, enhancing code clarity and consistency.

* **Removal of Batching in Inference Pipeline (#909):** Simplifies the inference pipeline by removing batching.

* **Improved Handling of Intermediate Releases (#957, #951):** Allows for the creation of intermediate releases without designating them as the latest release, facilitating more granular version control.

* **Resource Allocation in Neo4j Template (#977):**  Improves resource management by allocating resources to the main container in the Neo4j template.

* **Removed Unnecessary Reliance on Normalizer Outputs (#766):**  Improves pipeline efficiency by removing an unnecessary dependency on normalizer outputs.

* **Introduced `inject` Decorator (#901):** Simplifies code and promotes modularity by introducing the `inject` decorator. This change, combined with the introduction of separate nodes for each source (#901), enhances code organization and allows for greater parallelization.

* **Unification of Splits Generation (#907):** Ensures consistent schema across all data splits.

* **Single Model per Modelling Pipeline (#924):** Simplifies the modelling pipeline by enforcing a single model per pipeline.

* **Removal of Schema Validation in Evaluation Pipeline (#901):** Improves performance and reduces complexity in the evaluation step.

* **Using Full Dataset for Training (#901):** Improves model performance and training stability.

* **PySpark for All Data Manipulation (#901):**  Enhances performance and ensures consistency by using PySpark for all data manipulation tasks.


## Documentation and Onboarding Improvements

Extensive documentation updates improve onboarding and provide clearer guidance:

* **Improved Onboarding Materials (#902, #883, #886, #904):**  Updated onboarding documentation streamlines the onboarding process for new contributors.  This includes a change to the onboarding guide requiring new users to share their GPG public key via a Pull Request (PR) for enhanced security.

* **Updated Java Version Documentation (#903):**  Clarifies the required Java version.

* **Updated `libomp` Library Installation Instructions (#934):** Provides clear instructions for installing the `libomp` library.

* **Updated Release Creation Documentation (#940):**  Updates the documentation for creating releases to reflect the streamlined process.

* **Minor Typos Fixed in Kedro Extensions Documentation (#913):** Improves the clarity and professionalism of the documentation.

* **Updated Virtual Environment Setup Documentation (#906):**  Provides updated instructions for setting up a virtual environment.

* **Added Documentation for Kedro Resources (#919):**  Provides comprehensive information about using Kedro resources.

* **Updated Neo4j SSL Configuration Documentation (#878):**  Reflects the simplified Neo4j SSL setup process.

* **Updated Documentation on Ensemble Aggregation Function (#905):**  Explains how to use the new flexible ensemble aggregation function.

* **Improved Common Errors Documentation (#925):**  Adds solutions to newly encountered problems.

* **Updated `index.md` (#956):** Corrects a typo.

## Newly Onboarded Colleagues

We welcome the following new colleagues to the team:

* **Jacques (#883)**
* **Matej (#886)**
* **Kushal (#904)**
* **MariaHei (#944)**

## Other Notable Changes

* **Rendering Release Info in Docs (#858):**  Makes release information easily accessible within the documentation.

* **Added Matrix CLI Frontmatter (#959):** Adds support for MkDocs frontmatter for release notes and articles in the Matrix CLI.

* **Improved Git Checks for Kedro Submit (#961):**  Enhances the reliability of the `kedro submit` command by adding stricter Git checks, including branch name validation.

* **Added Integration Tests for Changelog Generation (#968):**  Improves the reliability of changelog generation.

* **Disable SSL for Local Neo4j (#972):** Simplifies local development by allowing developers to disable SSL for local Neo4j instances.

* **Updated `index.md` (#956):**  Ensures that the `index.md` file contains up-to-date information.


This release significantly enhances the Matrix Platform's capabilities and usability.  The introduction of k-fold cross-validation, flexible ensemble aggregation, and automated data releases improves model evaluation, customization, and efficiency.  The numerous bug fixes, technical enhancements, refactoring efforts, and documentation updates contribute to a more robust, reliable, and developer-friendly platform.
