---
draft: false
date: 2025-03-10
categories:
  - Release
authors:
  - oliverw1
  - pascalwhoop
  - eKathleenCarter
  - JacquesVergine
  - Siyan-Luo
  - piotrkan
  - emil-k
  - amyford
  - app/github-actions
  - alexeistepa
  - lvijnck
  - matentzn
---

# Matrix Platform `v0.4.0`: Enhanced Pipeline Efficiency, Robustness, and Data Management

This release of the Matrix Platform significantly improves pipeline efficiency, robustness, and data management. Key enhancements include a unified integration layer streamlining new data source incorporation, a sampling pipeline for faster development, k-fold cross-validation for robust model evaluation, stability metrics for assessing model consistency, and numerous technical enhancements for optimized performance and data quality.  Breaking changes include the adoption of the KGX format for data releases and the structural change introduced by k-fold cross-validation.

<!-- more -->

## Key Enhancements

### Streamlined Data Integration and Faster Development

* **Unified Integration Layer:** A new unified integration layer simplifies the process of incorporating new data sources. This standardized process enhances maintainability and reduces the complexity of integrating future knowledge graphs.

* **Sample Pipeline:** A sample pipeline now enables running the platform on a subset of the data, accelerating development and testing cycles. This allows for quicker iteration and identification of potential issues without the overhead of full data runs.

* **Run sampling pipeline on schedule (#1105):** The sampling pipeline is now automated and runs on a daily schedule, ensuring readily available sample datasets for development and testing purposes.

### Robust Model Evaluation and Stability Assessment

* **K-fold Cross-Validation (#683):** Model evaluation now utilizes k-fold cross-validation, providing a more robust assessment of model performance by training and evaluating on multiple data folds.  This mitigates the potential impact of data splits on evaluation results and offers insights into model generalization. **This is a breaking change that alters the output structure of modeling and evaluation pipelines.**

* **Stability Metrics:** New stability metrics provide a quantitative assessment of model stability across different data subsets and random seeds. These metrics offer valuable insights into the consistency and reliability of model predictions.

### Enhanced Monitoring and Pipeline Control

* **Deploy Grafana and Prometheus (#834):** The deployment of Grafana and Prometheus enhances monitoring capabilities, providing real-time insights into cluster performance and experiment runs. This improved observability allows for better resource management and proactive identification of potential issues.

* **Dynamic Pipeline Options (#901):**  Pipeline settings can now be dynamically loaded from the catalog using a resolver, enhancing flexibility and simplifying configuration management.

* **Customizable Ensemble Model Aggregation (#905):** Users can now define the aggregation function for ensemble models, providing greater control over the model combination process.

* **Specify MLFlow Experiment by Name (#1093):** Simplifies experiment management in MLFlow by allowing users to specify experiments by name, improving organization and traceability.

* **Add --nodes to Kedro submit (#1142):**  Finer control over pipeline execution is now available through the `--nodes` option in `kedro submit`, allowing users to specify specific nodes to run.

* **Workflow dispatch trigger for pipeline submission with bump type selection (#1138, #1223):** Pipeline runs can be manually triggered via the UI with specified version bump types, streamlining the release process.


## Data Management and Pipeline Efficiency

* **KGX Format for Data Releases (#743):** Data releases are now in KGX format, a standardized format for knowledge graph exchange. **This is a breaking change requiring adjustments in downstream pipelines consuming these releases.**

* **Modeling Cleanup (#907):**  Split generation in modeling is unified, improving data management within the pipeline.

* **Add GitHub release dataset for drug and disease list ingestion (#1050):** Streamlines drug and disease list ingestion by retrieving data from GitHub release artifacts.

* **Move de-duplication to integration from preprocessing (#1118):** Improves pipeline efficiency by moving the de-duplication step to the integration stage.

* **Add upstream data source to Neo4j edges (#1131):** Improves data provenance tracking by adding upstream data source information to Neo4j edges.

* **Correct BQ reporting table names and change tests to cover cloud catalog (#1133):**  Improves testing and data consistency by correcting BigQuery table names and enhancing cloud catalog test coverage.

* **Update BigQuery table if it exists instead of creating it (#1110):** Improves data handling efficiency by updating existing BigQuery tables rather than recreating them.

* **Resource allocation changes for embeddings pipeline (#1179, #1170):** Optimizes resource utilization for the embeddings pipeline, improving efficiency and performance.

* **Only log MLFlow dataset if it hasn't been logged before (#1180):**  Prevents redundant dataset logging in MLFlow, improving logging efficiency.

* **Feat/archive MLFlow runs (#1181):** Enables archiving of old MLFlow runs and experiments, improving organization and resource management.

* **Revert window size to 10 for Node2Vec Embeddings (#1184):** Restores the window size parameter to its previous value for Node2Vec embeddings.

* **Add rank columns (#1186):** Adds rank columns to prediction results, providing additional information for analysis.

* **Reduce resource requirements for edge and node ingestion into Neo4j (#1195):** Optimizes resource usage for Neo4j ingestion, improving efficiency.

## Bug Fixes

* **Missing Edges Bug Fix (#781):** Addresses missing edges after deduplication.

* **Import Error Fix (#823):**  Corrects an import error due to branch drift.

* **Schema Error Fix (#893):** Resolves a schema error and renames `is_ground_pos` to `in_ground_pos`.

* **Neo4j Connection Protocol Fix (#899):** Fixes the protocol in the `wipe_neo` script.

* **Argo Release Trigger Fix (#936):** Resolves an issue with the trigger release label in Argo.

* **Fix clinical trial preprocessing nodes (#1039):** Ensures correct handling of clinical trial data.

* **Fix normalizer always returning `normalization_success=True` (#1060):** Ensures accurate reporting of normalization status.

* **Fix MLFlow metric tracking (#1075):**  Ensures accurate recording of metrics.

* **Fix integration pipeline error with missing interpolation key (#1123):** Resolves pipeline errors due to missing interpolation keys.

* **Fix modelling bug - modelling cloud catalog (#1165):** Corrects issues with the modeling catalog in the cloud environment.


## Technical Enhancements

* **Refactoring and Code Improvements:** Multiple refactoring efforts have simplified dependencies, improved type annotations, enhanced error messages, and streamlined the codebase.

* **MLFlow Local Disable Option (#756):** Added a flag to disable local MLFlow tracking.

* **Simplified Neo4j SSL Setup (#878):**  Streamlined Neo4j SSL configuration.

* **Fix ec medical nodes in preprocessing (#1052):** Ensures correct processing of EC medical nodes.

* **Update onboarding docs to include container registry auth (#1081):** Improves the onboarding process.

* **Pinned torch and re-generate requirements on mac (#1109):**  Improves dependency management for Mac environments.


## Documentation Improvements

* **Release Article Date Fix (#796):** Corrected date formatting.

* **MathJax Support (#796):**  Added support for mathematical formulas.

* **Google Analytics Integration (#796):** Integrated Google Analytics for website tracking.

* **MOA Codebase Documentation (#798):** Improved documentation for MOA code.

* **VS Code Debugging Documentation (#799):** Enhanced debugging instructions.

* **Virtual Environment Documentation Update (#906):** Updated virtual environment setup guide.

* **Kedro Resource Documentation (#919):** Added documentation for Kedro resources.

* **Onboarding Documentation Fixes (#883, #902, #1081):** Addressed typos and clarity issues.

* **Common Errors Documentation Update (#925):** Added solutions for commonly encountered problems.

* **`libomp` Installation Instructions (#934):**  Added `libomp` installation instructions.

* **`pyenv` Installation Instructions (#812):** Added instructions for Python installation with `pyenv`.

* **SILC Troubleshooting Documentation (#836):**  Added troubleshooting guide for SILC.

* **Improve sampling documentation with release specific instructions (#1166):** Clarified sampling instructions.

* **Add documentation for disease tagging / categorisation feature (#955):** Documented the disease tagging/categorization feature.


## Newly Onboarded Colleagues

* **Kathleen Carter (#1032):** Added Kathleen's GPG key.


## Other Notable Changes

* **Code Ownership Cleanup (#822):** Refined code ownership to groups.

* **Git SHA Labeling and Clean State Check (#771):** Enforced clean Git state and added SHA labeling.

* **`RELEASE_NAME` Environment Variable Update (#750):**  Updated environment variable format.

* **Technical Debt Addressing (#757):** Addressed missing Kedro catalog entries.

* **Score Saving for Full Matrix (#853):** Saved "not treat" and "unknown" scores.

* **Release PR Label (#852):** Added label to hide release PRs.

* **Spoke Version Increment (#914):** Updated Spoke version.

* **Headless CLI Flag (#861):** Added `--headless` flag.

* **SemMedDB Filtering Reversion (#826):** Reverted filtering changes.

* **Kedro Extensions Documentation Fixes (#913):** Fixed typos.

* **CLI Error Handling Improvement (#827):** Improved error handling.

* **Test Suite Refactoring (#930):** Refactored tests.

* **Java Version Upgrade (#903):** Upgraded Java to version 17.

* **Missing Column Addition (#811):** Added missing column.

* **Argo Node Output Correction (#844):** Corrected Argo output.

* **`is_async` Flag Renamed (#811):** Renamed flag to `is_test`.


This release represents a significant advancement in the Matrix platform, with improvements in data integration, model evaluation, pipeline efficiency, and overall robustness. The highlighted changes empower users with more efficient workflows, enhanced data management capabilities, and greater control over the drug repurposing pipeline.
