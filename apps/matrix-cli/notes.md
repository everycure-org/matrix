---
draft: false
date: 2025-03-10
categories:
  - Release
authors:
  - JacquesVergine
  - eKathleenCarter
  - emil-k
  - amyford
  - pascalwhoop
  - Siyan-Luo
  - oliverw1
  - app/github-actions
  - piotrkan
  - alexeistepa
  - lvijnck
  - matentzn
---
## Breaking Changes 🛠

No breaking changes were introduced in this release.


## Exciting New Features 🎉

- **Unified Integration Layer:**  A streamlined integration process for all data sources, simplifying the addition of new sources. This includes the integration of Spoke KG.
- **Sampling Mechanism:** Ability to run the pipeline on a sample of data for faster development and testing, providing more confidence in end-to-end pipeline execution.
- **K-fold Cross-Validation:** Updated pipeline trains multiple models on different data subsets to assess model stability and avoid lucky seeds.
- **Model Stability Metrics:** Extended pipeline evaluation provides model stability metrics, offering deeper insights into model robustness.
- **Grafana and Prometheus Deployment:** Enhanced monitoring of the Kubernetes cluster and experiments using Grafana and Prometheus for improved observability.
- **Improved Data Tracking:** Release history webpage provides detailed information about each data release, including datasets used. MLFlow now logs datasets.
- **Kedro Experiment:** New CLI command for running kedro pipelines and logging results to MLFlow. Allows for better organization and tracking of experiments.


## Experiments 🧪

No new experiments were reported in this release.


## Bugfixes 🐛

- **Clinical Trial Preprocessing:** Resolved an issue in the clinical trial preprocessing, ensuring correct data handling.
- **Data Normalization Status:** Corrected the normalizer to accurately reflect `normalization_success` status.
- **MLflow Metric Tracking:** Fixed a problem in MLflow metric tracking, ensuring accurate recording.
- **Missing Edges:** Fixed a bug causing missing edges after deduplication in the integration pipeline.
- **Import Error:** Corrected an import error caused by branch drift.
- **Schema Error:** Resolved a schema error concerning null values in the `ingest_nodes` process.
- **Neo4j Connection Protocol:** Corrected the Neo4j connection protocol in the `wipe_neo` script.
- **Argo Release Trigger:** Resolved an issue with the trigger release label in Argo.
- **CLI Error Handling:** Improved error handling in the CLI to better display output during streaming.
- **Writing to Google Sheets:** Fixed issues with writing to Google Sheets for the SILC sheet.


## Technical Enhancements 🧰

- **GitHubReleaseCSVDataset:** Added a new dataset type for streamlined ingestion from GitHub releases.
- **Periodic Data Releases Automation:** Automated monthly data release process.
- **Simplified Pipeline:** Streamlined pipeline to produce a single model per run.
- **Pandera Integration:** Replaced a private package dependency with Pandera for runtime data quality checks.
- **Git SHA Labeling:** Added a Git SHA label to the workflow template.
- **Code Ownership Cleanup:** Refined code ownership definitions by assigning specific groups.
- **Clean Git State Check:** Enforced a clean Git state before submission.
- **`RELEASE_NAME` Environment Variable Update:** Updated `RELEASE_NAME` environment variable format.
- **Technical Debt:** Addressed missing and unused Kedro catalog entries and improved node category selection. Documented code review best practices.
- **Score Saving:** Added "not treat" and "unknown" probability scores to the full matrix.
- **Release PR Label:** Added a label to hide Release PRs from release notes.
- **`apply_transform` Refactoring:** Refactored the `apply_transform` function.
- **Spoke Version Increment:** Incremented the Spoke KG integration version.
- **Headless CLI Flag:** Added a `--headless` flag to the CLI.
- **Test Suite Refactoring:** Refactored the test suite.
- **Java Version Upgrade:** Upgraded the required Java version.
- **Missing Column Addition:** Added a missing `knowledge_source` column.
- **Argo Node Output Correction:** Corrected the Argo node output.
- **`is_async` Flag Renamed:** Renamed the `is_async` flag.


## Documentation ✏️

- **Improved Release Article:** Corrected date formatting, enabled MathJax support and Google Analytics tracking in release articles.
- **MOA Codebase Documentation:** Improved MOA codebase documentation.
- **VS Code Debugging Documentation:** Enhanced debugging instructions for VS Code.
- **Virtual Environment Documentation Update:** Updated virtual environment setup instructions.
- **Kedro Resources Documentation:** Added documentation for Kedro resources.
- **Onboarding Documentation Fixes:** Corrected typos and clarity issues in onboarding documentation.
- **Common Errors Documentation Update:** Updated frequently encountered errors and their solutions.
- **`libomp` Installation Instructions:** Added instructions for installing the `libomp` library.
- **`pyenv` Installation Instructions:** Improved instructions for using `pyenv`.
- **SILC Troubleshooting Documentation:** Added a troubleshooting document for the SILC process.
- **Disease Tagging/Categorization Documentation:** Added documentation for the disease tagging feature.
- **Kedro Experiment Documentation:** Improved documentation for `kedro experiment` CLI command.
- **Updated Release Info:** Include drug and disease information in release info.
- **Release notes doc improvement**: improved the wording in the doc for generating release notes.


## Newly onboarded colleagues 🚤

- Jacques
- Kushal
- Marcello
- Matej
- Kathleen


## Other Changes

- IAP OAuth Setup for MLFlow: Implemented Identity-Aware Proxy (IAP) OAuth for programmatic access to MLFlow.
- Removed Hardcoded SILC Config: Removed hardcoded SILC configuration for better flexibility and management.
- Improved Parameterized Dashboards:  Added additional dashboards and improved existing ones.
- Add ability to specify MLFlow experiment by name: Added the possibility to specify the MLFlow experiment to run from the CLI using the `experiment-name` option.
- Archive Runs and Experiments: Added the ability to archive MLFlow runs and experiments.
- Add support for specifying nodes to run from and nodes to run: Added ability to run a subset of a pipeline's nodes.
- Add --nodes option: Added the option to specify nodes to run using the `--nodes` option.
- Added run name prompt: Added a prompt to enter a run name if it's not already specified.
- Add workflow dispatch trigger: Add a trigger to initiate the workflow manually via GitHub.
- Add custom prometheus metric to track failed workflow status: Added a custom Prometheus metric to track the failed workflow statuses.
