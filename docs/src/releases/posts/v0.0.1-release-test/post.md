---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - JacquesVergine
  - marcello-deluca
  - lvijnck
  - emil-k
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
---
# `v0.2.7`: Automated Data Releases, Enhanced Pipeline Flexibility, and Improved Infrastructure

This release of the Matrix Platform introduces significant advancements in automation, pipeline flexibility, and infrastructure improvements. Key highlights include automated periodic data releases, enhanced control over ensemble model aggregation, and streamlined dependency management.

<!-- more -->

## Automated and Enhanced Pipelines

This release focuses on automating key processes and improving pipeline flexibility:

* **Automated Periodic Data Releases (#877):** A scheduled job now automatically submits the Kedro pipeline for periodic data releases, streamlining the release process and ensuring regular updates.  This automation reduces manual effort and improves the frequency of data releases.

* **Flexible Ensemble Model Aggregation (#905):**  A new parameter allows users to specify the aggregation function for ensemble models, providing greater control over how predictions from individual models are combined. This enhances model performance and customization options.

* **K-fold Cross-Validation (#683):** Model evaluation robustness is significantly improved by implementing k-fold cross-validation, generating multiple train-test splits for a more comprehensive performance assessment.

* **Dynamic Pipeline Options and Resolver (#901):** Overriding dynamic pipeline options and using a resolver to load settings from the catalog provides greater flexibility in pipeline configuration and execution.

* **Single Model per Modelling Pipeline (#924):** Restricting modelling pipelines to a single model improves clarity and prevents potential conflicts, simplifying pipeline management.

## Infrastructure and Dependency Management

Several updates enhance infrastructure and simplify dependency management:

* **Refactoring to Remove `refit` Library (#811):**  Removing the `refit` library streamlines the codebase and reduces dependencies, simplifying maintenance and improving code clarity.

* **Simplified Neo4j SSL Setup (#878):**  Improvements to Neo4j SSL configuration simplify setup and enhance security.  Updated documentation reflects these changes.

* **Updated Neo4j Connection String (#880):**  Updating the Neo4j connection string in templates ensures consistency and clarity.

* **`argo_node` Refactoring (#885):**  Replacing the `argo_node` function with the `ArgoNode` class improves code structure and organization.

* **Consistent `pyspark.sql` Usage (#923):**  Standardizing the use of `pyspark.sql` enhances code consistency and readability.

* **Resource Allocation in Neo4j Template (#977):** Allocating resources to the main container in the Neo4j template improves performance and stability.

* **Disabling SSL for Local Neo4j (#972):** Disabling SSL for local Neo4j instances simplifies local development setup.

## Bug Fixes and Stability Improvements

Numerous bug fixes enhance platform stability and reliability:

* **`wipe_neo` Script Protocol Fix (#899):** Correcting the protocol improves Neo4j interaction.

* **Improved Kedro Hooks Test Coverage (#900):** Increased test coverage enhances release stability.

* **Schema Error Fix in `ingest_nodes` (#943):** Handling null values in the `name` column resolves a schema error.

* **Release Branching Fix (#950):** Creating release branches from the correct commit ensures consistency in release management.

* **CI Fixes (#967):** Addressing CI failures on the `main` branch improves development workflow.

* **Git SHA Fetch Fix (#974):** Using the correct command (`git rev-parse HEAD`) for fetching the Git SHA ensures accuracy.

* **GitHub Token Access Fix (#990):**  Resolving a GitHub token access issue ensures proper workflow execution.

## Documentation and Onboarding Enhancements

Several updates improve documentation and onboarding:

* **Kedro Resource Documentation (#919):**  New documentation for Kedro resources aids understanding and usage.

* **Updated Common Errors Documentation (#925):**  Adding solutions for new problems improves troubleshooting.

* **Onboarding Fixes (#902):**  Improved onboarding documentation streamlines the process for new contributors.

* **Updated Java Version Documentation (#903):**  Specifying Java 17 as the required version ensures compatibility.

* **Added `libomp` Library to Installation Documentation (#934):**  Including the `libomp` library in installation instructions ensures necessary dependencies are installed.

* **Updated Release Creation Documentation (#940):**  Clearer instructions simplify the release process.

* **Typo Fixes in Kedro Extensions Documentation (#913):**  Correcting typos improves documentation clarity.

* **Updated Virtual Environment Documentation (#906):**  Updated instructions simplify virtual environment setup.

* **Updated `index.md` (#956):** Updated instructions in `index.md` improve clarity.

* **Matrix CLI Mkdocs Frontmatter (#959):** Including Mkdocs frontmatter in the Matrix CLI streamlines documentation generation.


## Other Notable Changes

* **`apply_transform` Improvements (#808):** Adding unit tests and minor improvements enhances the `apply_transform` function.

* **`object` Keyword Renaming (#922):** Renaming improves readability and avoids conflicts.

* **Test Suite Refactoring (#930):** Improves structure and maintainability of the testing suite.

* **Import Sorting (#931):** Enhances code readability and consistency.

* **Argo Release Trigger Label (#936):** Adding labels improves workflow tracking.

* **Modelling Cleanup - Unify Splits (#907):**  Improves clarity and consistency in the modelling process.

* **Release Info Rendering (#858):** Improves accessibility of release information.

* **SPOKE Version Increment (#914):** Keeps the SPOKE version up-to-date.

* **Debug Directory Listing (#988):** Aids in troubleshooting file not found errors.

* **Intermediate Non-Latest Releases (#957):** Enables more flexible release management.

* **Release Branch Check (#921):**  Ensures data releases are triggered from the correct branches.

* **Tag Existence Check (#983):** Prevents redundant release submissions.

* **Test Data Release on Main (#989):** Simplifies testing of the data release pipeline.

## Newly Onboarded Colleagues

* **Jacques (#883):**  Key added to git-crypt.
* **Matej (#886):** Key added to git-crypt.
* **Kushal (#904):** Key added to git-crypt.
* **MariaHei (#944):** Key added to git-crypt.

This release significantly advances the Matrix Platform through automation, enhanced pipeline flexibility, and improved infrastructure.  The automated data releases, flexible ensemble aggregation, and numerous bug fixes contribute to a more robust and efficient platform.  The enhanced documentation and onboarding materials further improve the developer experience.
