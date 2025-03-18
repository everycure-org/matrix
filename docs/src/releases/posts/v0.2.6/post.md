---
draft: false
date: 2025-01-15
categories:
  - Release
authors:
  - JacquesVergine
  - emil-k
  - lvijnck
  - marcello-deluca
  - jdr0887
  - matwasilewski
  - alexeistepa
  - pascalwhoop
  - piotrkan
  - oliverw1
  - Siyan-Luo
  - webyrd
  - matej-macak
---
# `v0.2.6`: Robust Model Evaluation and Enhanced Pipeline Flexibility with KGX and Cross-Validation

This release marks a significant advancement in the Matrix platform with a focus on robust model evaluation and enhanced pipeline flexibility.  Key improvements include the transition to k-fold cross-validation, enabling more rigorous model assessment, the adoption of the KGX format for data releases, simplifying downstream integration, and significant refactoring for improved code maintainability.

<!-- more -->

## Enhanced Model Evaluation and Pipeline Flexibility

This release introduces substantial improvements in model evaluation and overall pipeline adaptability:

* **K-fold Cross-Validation (#683):** The modeling pipeline now employs k-fold cross-validation, a significant enhancement for robust model evaluation. This change provides a more comprehensive performance assessment by generating multiple train-test splits and aggregating results, offering better insights into model generalization.  Consequently, the output structure of the modeling and evaluation pipelines has been modified to accommodate the results from each fold.

* **KGX Format for Data Releases (#743):**  The data release pipeline now outputs data in the KGX format, a standardized knowledge graph exchange format.  This transition improves interoperability and simplifies the integration of Matrix data with downstream applications and pipelines. This change requires adjustments for pipelines consuming these data releases.

* **Dynamic Pipeline Options (#901):**  Pipeline configuration is more flexible with the addition of dynamic pipeline options. Settings can now be loaded directly from the catalog using a resolver, streamlining configuration management and enabling greater customization.

* **Customizable Ensemble Model Aggregation (#905):** A new parameter provides control over the ensemble model's aggregation function. This enables users to tailor the aggregation strategy to their specific needs, further enhancing model flexibility.  The updated documentation details how to utilize this feature.

## Spoke KG Integration and MOA Visualization

* **Spoke Knowledge Graph Integration (#772):** The Spoke Knowledge Graph is now integrated into the pipeline, broadening the knowledge base and enriching the context for drug repurposing analysis.

* **MOA Visualizer App (#798):**  A new MOA (Mechanism of Action) visualizer app enhances the exploration and interpretation of MOA prediction results, providing a more intuitive understanding of predicted relationships.  Accompanying documentation details the usage and functionality of the visualizer.

## Infrastructure Enhancements: Monitoring and Deployment

* **Grafana and Prometheus Deployment (#834):**  The deployment of Grafana and Prometheus provides enhanced monitoring capabilities for both the cluster and individual experiments.  This allows for real-time performance tracking and better resource management.

## Bug Fixes and Technical Enhancements

This release addresses several bugs and incorporates various technical enhancements:

* **Missing Edges Bug Fix (#781):** A bug causing missing edges after deduplication in the integration pipeline has been resolved, ensuring data integrity.

* **Import Error Fix (#823):** An import error arising from branch drift has been corrected, improving code stability.

* **Kedro Hooks Test Coverage (#900):**  Improved test coverage for Kedro hooks enhances the reliability and stability of the release process.

* **Schema Error Fix (#893):** A schema error related to null values in the `ingest_nodes` process has been resolved, ensuring data consistency.  The `is_ground_pos` column has been renamed to `in_ground_pos` for clarity.

* **Neo4j Connection Protocol Fix (#899):**  The Neo4j connection protocol in the `wipe_neo` script has been corrected, resolving potential connection issues.

* **Argo Release Trigger Fix (#936):** An issue with the trigger release label in Argo has been resolved, streamlining the release process.

* **Refactoring and Code Improvements:** Multiple refactoring efforts, including the removal of the `refit` library (#811), improvements to type annotations (#806), and the use of sets (#806), result in cleaner, more maintainable code.  The `argo_node` function has been replaced with the `ArgoNode` class (#885) for better code structure.  Consistent usage of `pyspark.sql` (#923) and added import sorting (#931) enhance code readability. Redundant catalog files have been removed (#795), and the `kedro submit` command's verbose flag has been replaced with a more intuitive quiet flag (#828).

* **MLflow Local Disable Option (#756):** A flag to disable MLflow tracking locally has been added, reducing overhead during development. This option is now disabled by default.

* **Improved Error Verbosity (#791):** Enhanced error messages improve debugging and troubleshooting.

* **Modeling Cleanup (#907):** The modeling pipeline benefits from cleaner data management by unifying how test-train splits are generated and managed.

* **Simplified Neo4j SSL Setup (#878):**  The setup process for Neo4j SSL has been streamlined, improving developer experience.

## Documentation and Onboarding Enhancements

Several documentation updates improve clarity and accessibility for both new and existing users:

* **Release Article Date Fix (#796):**  Date formatting in release articles has been corrected.

* **MathJax Support (#796):** Support for MathJax enables the rendering of mathematical formulas in documentation.

* **Google Analytics Integration (#796):**  Google Analytics tracking has been enabled for website analytics.

* **MOA Codebase Documentation (#798):** Documentation for the MOA codebase has been significantly improved.

* **VS Code Debugging Documentation (#799):**  Debugging instructions for VS Code have been enhanced.

* **Virtual Environment Documentation Update (#906):**  Virtual environment setup instructions in the onboarding documentation have been updated for clarity.

* **Kedro Resource Documentation (#919):**  Documentation for Kedro resources has been added.

* **Onboarding Documentation Fixes (#883, #902):**  Typos and clarity issues in onboarding documentation have been addressed, including adding a Git-crypt key for Jacques.

* **Common Errors Documentation Update (#925):** Frequently encountered errors and their solutions have been updated in the `common_errors.md` document.

* **`libomp` Installation Instructions (#934):**  Instructions for installing the `libomp` library have been added.

* **`pyenv` Installation Instructions (#812):** Improved instructions for using `pyenv` to install Python have been added.

* **SILC Troubleshooting Documentation (#836):**  A troubleshooting document for the SILC (Service Integration, Logging, and Configuration) process has been added.

## New Contributors

Several new colleagues have joined the project:

* Jacques (#883)
* Kushal (#904)
* Marcello (#892)
* Matej (#886)

## Other Notable Changes

* **Code Ownership Cleanup (#822):** Code ownership definitions have been refined by assigning specific groups rather than individuals.

* **Git SHA Labeling and Clean State Check (#771):**  A Git SHA label is now added to the workflow template, and a clean Git state is enforced before submission.

* **`RELEASE_NAME` Environment Variable Update (#750):**  The `RELEASE_NAME` environment variable has been updated to a more specific version format.

* **Technical Debt Addressing (#757):**  Missing and unused Kedro catalog entries have been addressed, along with more robust node category selection during integration.  Code review best practices have been documented.

* **Score Saving for Full Matrix (#853):** Scores for "not treat" and "unknown" probabilities are now stored in the full matrix.

* **Release PR Label (#852):**  A label to hide Release PRs from release notes has been added.

* **`apply_transform` Refactoring (#808):** The `apply_transform` function has been refactored for improved structure and clarity.

* **Spoke Version Increment (#914):**  The Spoke KG integration version has been incremented.

* **Headless CLI Flag (#861):**  A `--headless` flag has been added to the CLI to disable user prompts.

* **SemMedDB Filtering Reversion (#826):** Changes to the SemMedDB filtering logic have been reverted.

* **Kedro Extensions Documentation Fixes (#913):** Minor typos in the Kedro Extensions documentation have been corrected.

* **CLI Error Handling Improvement (#827):**  Error handling in the CLI has been improved to capture and display output to both stdout and stderr during streaming.

* **Test Suite Refactoring (#930):** The test suite has been refactored for easier maintenance and expandability.

* **Java Version Upgrade (#903):** The required Java version has been upgraded to 17.

* **Missing Column Addition (#811):** A missing `knowledge_source` column has been added to the `transform_edges` function.

* **Argo Node Output Correction (#844):** The Argo node's output has been corrected to match the single item returned by its function.

* **`is_async` Flag Renamed (#811):** The `is_async` flag passed to the `kedro run` command has been renamed to `is_test`.

This release represents a substantial step forward for the Matrix Platform. The introduction of k-fold cross-validation and the KGX format strengthens the platform's capabilities for robust model evaluation and data integration. The various infrastructure enhancements, bug fixes, and documentation improvements contribute to a more stable, reliable, and user-friendly platform.
