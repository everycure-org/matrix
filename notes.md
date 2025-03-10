---
draft: false
date: 2025-03-10
categories:
  - Release
authors:
  - Siyan-Luo
  - amyford
  - alexeistepa
  - piotrkan
  - pascalwhoop
  - emil-k
  - eKathleenCarter
  - JacquesVergine
  - app/github-actions
  - lvijnck
  - oliverw1
  - matentzn
---
## Breaking Changes 🛠

* Transitioned to k-fold cross-validation for more robust model evaluation (#1078). Output structure of modelling and evaluation pipelines modified to accommodate results from each fold.
* Data release pipeline now outputs data in KGX format for improved interoperability (#1078). Requires adjustments for downstream pipelines.
* Unified integration layer for streamlined dataset integration (#1078).  Simplified the process of integrating new knowledge graphs.
* Introduced a sampling mechanism to enable efficient pipeline testing (#1078). Enables running the pipeline on smaller datasets.
* Implemented model stability metrics (#1078). Assesses model stability across data subsampling.


## Exciting New Features 🎉

* Added functionality to run the sampling pipeline periodically (#1105). Allows for automatic data sampling updates.
* Implemented a MOA (Mechanism of Action) visualizer app for intuitive MOA prediction exploration (#1078). Enhances understanding of predicted relationships.
* Created a new feature that allows logging datasets used in the MLFlow runs (#1048). Enhances reproducibility.
* Introduced the ability to submit a workflow to a specific experiment by name using the new kedro experiment command (#1093). Allows for more fine-grained control over experiments.
* Added an "archive" command to the kedro experiment to help cleanup the mlflow server (#1181). 
* Added a new mechanism to allow running pipelines on subsets of nodes (#1142).

## Experiments 🧪

* No new experiments reported in this release.

## Bugfixes 🐛

* Resolved a bug causing missing edges after deduplication in the integration pipeline (#1078). Ensures data integrity.
* Corrected an import error arising from branch drift (#1078). Improves code stability.
* Fixed a bug in clinical trial preprocessing nodes (#1039). Ensures correct handling of clinical trial data.
* Resolved an issue affecting MLflow metric tracking (#1075). Ensures accurate recording of metrics.
* Corrected the normalizer to accurately reflect the normalization success status (#1060). Improves reliability of data normalization feedback.
* Fixed a schema error related to null values in the ingest_nodes process (#1123). Improves data consistency.
* Resolved an issue with the trigger release label in Argo (#1078). Streamlines release process.
* Addressed several minor typos and clarified instructions in the documentation (#1150).
* Fixed a bug causing deadlocks in subprocess calls (#1089)


## Technical Enhancements 🧰

* Improved the structure and maintainability of the testing suite (#1150). Improves code quality and issue identification.
* Refactored the modelling pipeline to produce a single model per run, streamlining comparability across runs (#1078). Enhances ease of comparing run results.
* Simplified pipeline configuration by using dynamic pipeline options (#1078). Improves configuration management.
* Implemented a flexible mechanism to disable hooks using environment variables (#1078). Improves local development and debugging.
* Added support for MathJax to enable rendering of mathematical formulas (#1078). Improves document clarity.
* Added Google Analytics tracking for website analytics (#1078). Improves tracking of website usage.
* Added a new GitHubReleaseCSVDataset for streamlined ingestion of sources from GitHub releases (#1050).
* Implemented automation to orchestrate periodic data releases (#1078). Ensures regular data release cycles.
* Updated the pipeline to use Pandera for runtime data quality checks (#1078). Improves data validation and reduces reliance on private packages.
* Unified how test-train splits are generated and managed, improving data handling efficiency (#1078).
* Improved error messages for better debugging and troubleshooting (#1078).
* Added a `--headless` flag to the CLI to disable user prompts (#1078). Improves efficiency in non-interactive environments.
* Refined code ownership definitions, using specific groups instead of individuals (#1078). Improves clarity and maintainability.
* Improved the structure and readability of code by refactoring multiple functions and improving code style (#1078).
* Updated several sections of the documentation to improve clarity and address various minor issues (#1150).
* Upgraded the required Java version to 17 (#1078). Addresses compatibility issues and improves setup process.
* Migrated from git-crypt to using gcloud secrets to handle the cryptographic keys and avoid any security breaches (#1073).


## Documentation ✏️

* Updated various sections of the documentation to improve clarity, address minor typos, and improve the instructions. (#1150, #1159, #1166, #1081, #1114)
* Added documentation on how to use the new flexible ensemble aggregation function (#1078).
* Added documentation for Kedro resources (#1078).
* Improved onboarding documentation and materials (#1078).
* Updated the documentation on how to create a release (#1078).
* Added documentation for a new process on how to fix a corrupted KG release (#1207).
* Added documentation explaining how tags work over various releases (#1209).
* Added a new page to the documentation explaining how to use the kedro experiment CLI (#1159).
* Added a new page explaining the design decisions behind the public data zone (#1221).
* Improved and updated existing documentation sections, including details about setup and usage (#1150).
* Added new documentation pages including: an overview of the public data zone (#1221), a glossary of common terms and definitions (#1221), and a runbook for creating OAuth clients. (#1168)
* Added documentation on how to use the `--nodes` flag in the kedro submit command (#1142)


## Newly onboarded colleagues 🚤

* eKathleenCarter (#1032)
* No other new colleagues onboarded in this release.

## Other Changes

* Added support for rendering release information in the documentation (#1078).
* Added labels to Argo workflows to indicate data release triggers (#1078). Improves monitoring and tracking.
* Added automation to the pipeline for regularly submitting periodic data releases (#1078). Improves efficiency in data management.
* Added various quality-of-life improvements to the repository, including streamlining the process of adding users to teams (#1040).
* Improved the workflow for creating and managing experiments in MLFlow (#1181).
* Updated the pipeline to reflect the new versions for RTX-KG2 and other datasets (#1199).
* Added new pages to the Evidence.dev dashboard, including an explorer page for discovering nodes by category or prefix (#1153).


