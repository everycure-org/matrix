---
draft: false
date: 2025-03-06
categories:
  - Release
authors:
  - emil-k
  - eKathleenCarter
  - Siyan-Luo
  - alexeistepa
  - JacquesVergine
  - lvijnck
  - pascalwhoop
  - piotrkan
  - amyford
  - app/github-actions
  - oliverw1
  - matentzn
---

# Matrix Platform `v0.4.0`: Enhanced Pipeline Control, Monitoring, and Data Integration

This release of the Matrix Platform focuses on enhanced pipeline control, improved monitoring capabilities, and streamlined data integration, alongside crucial bug fixes and technical enhancements.  Key changes include a more flexible pipeline submission process,  improved resource management within the Kubernetes cluster, and refined data integration workflows.

<!-- more -->

## Breaking Changes

* **KGX Format for Data Releases (#743):** Data releases are now provided in the KGX format. This standardization simplifies downstream integration and interoperability but requires adjustments for any pipelines consuming these data releases.


## Enhanced Pipeline Control and Monitoring

* **Unified Integration Layer (#772, #1039):** A streamlined data integration process simplifies the addition of new data sources.  This layer incorporates Spoke KG integration and facilitates a unified, standardized approach to data ingestion.

* **Sample Pipeline:**  A new sample pipeline enables running pipelines on a representative subset of real data. This facilitates faster development cycles and increased confidence in pipeline functionality before full data runs.  Scheduled sample data generation (#1105) further supports development workflows.

* **Grafana and Prometheus Deployment (#834):** Enhanced monitoring through Grafana and Prometheus provides comprehensive insights into cluster performance and individual experiment resource usage.

* **Dynamic Pipeline Options (#901):** Pipeline configurations are now more flexible, loading settings directly from the catalog using a resolver.

* **Ability to specify mlflow experiment by name (#1093):** Workflow submission is simplified with the ability to specify MLflow experiments by name.

* **Add --nodes to Kedro submit (#1142):**  Granular control over pipeline execution is now possible through the command line, allowing for the execution of specific nodes within a pipeline.
* **Expose integration pipeline's datasets in BigQuery (#1076):**enable direct SQL querying in in BQ and monitoring in [KG dashboard](https://data.dev.everycure.org/versions/latest/evidence/) for better QC and debugging capabilities.
## Data Integration and Management

* **Spoke KG Integration (#772):** The Spoke Knowledge Graph is now integrated, enriching the platform's knowledge base and analysis capabilities.

* **Add GitHub release dataset for drug and disease list ingestion (#1050):** Drug and disease list ingestion is streamlined through automated retrieval from GitHub releases.

* **Move de-duplication to integration from preprocessing (#1118):**  Pipeline efficiency is improved by shifting deduplication of custom datasets (provided by EC medical team) to the integration stage.

* **Add upstream data source to Neo4j edges (#1131):** Data provenance tracking is enhanced within Neo4j by adding the upstream data source information to edges.

## Visualization and Interpretation

* **MOA Visualizer App (#798):** This new application allows for interactive exploration and interpretation of Mechanism of Action prediction results.

## Bug Fixes

* **Missing Edges Bug Fix (#781):**  Resolved an issue causing missing edges after deduplication.

* **Import Error Fix (#823):** Corrected an import error stemming from branch drift.

* **Kedro Hooks Test Coverage (#900):** Improved test coverage for Kedro hooks ensures greater reliability.

* **Schema Error Fix (#893):** Resolved a schema error and improved column naming clarity by renaming `is_ground_pos` to `in_ground_pos`.

* **Neo4j Connection Protocol Fix (#899):** Corrected the protocol used in the `wipe_neo` script.

* **Argo Release Trigger Fix (#936):** Resolved an issue related to the trigger release label in Argo workflows.




* **Fix integration pipeline error with missing interpolation key (#1123):** Corrected an error in the integration pipeline related to missing interpolation keys.

* **Fix writing to the gsheets for SILC sheet (#1193):** Ensured correct data reporting to Google Sheets for SILC.

## Technical Enhancements

* **Refactoring and Code Improvements:** Multiple refactoring efforts and code improvements enhance maintainability and readability (e.g., #811, #806, #885, #923, #931, #795, #828).

* **MLflow Local Disable Option (#756):** Added a flag to disable local MLflow tracking, now disabled by default.

* **Improved Error Verbosity (#791):** Enhanced error messages facilitate debugging.

* **Modeling Cleanup (#907):** Unified split generation within the modeling pipeline.

* **Simplified Neo4j SSL Setup (#878):** Streamlined Neo4j SSL configuration.

* **Fix normalizer always returning `normalization_success=True` (#1060):**  Ensures accurate normalization status reporting.

* **'Infra 2 main sync: git-crypt replaced with script' (#1073):** Improved secrets management by replacing `git-crypt`.

* **Modify the location of AI-generated notes file (#1129):** Improved organization of release-related files.

* **Bug/add gh token for kedro submit step in GH Actions (#1132):** Ensured proper authentication for Kedro submissions in GitHub Actions.

* **Correct BQ reporting table names and change tests to cover cloud catalog (#1133):**  Improved consistency and test coverage for BigQuery reporting.

* **Disable stability metrics (temporarily) (#1126):**  Temporarily disabled stability metrics.

* **Disable notes generation (#1137):** Temporarily disabled automated notes generation.

* **Add repository_dispatch trigger to pipeline submission (#1138):**  Enabled remote triggering of pipeline submissions.

* **Hotfix for evidence.dev deployment working in CI (#1143):**  Ensured correct deployment of Evidence.dev in the CI environment.

* **Increasing the timeout to handle api overloading (#1146):**  Addressed potential API timeout issues.

* **Add min max to reported aggregations (#1152):**  Provided more comprehensive statistics in aggregated reports.

* **Improved sankey on evdev dashboard (#1153):**  Enhanced data flow visualization on the Evidence.dev dashboard.

* **Update test configuration for topological embeddings to reduce integration test duration (#1161):** Optimized integration test execution time.

* **'Fix modelling bug - modelling cloud catalog ' (#1165):**  Corrected a bug in the modeling cloud catalog.

* **Comment out ingest_nodes_with_embeddings (#1175):**  Temporarily disabled a specific node in the pipeline.

* **Fix catalog in ingestion (#1176):**  Corrected issues in the ingestion catalog.

* **Debug/use git command instead of gh command (#1178):** Improved compatibility using Git commands directly.

* **Resource allocation changes for embeddings pipeline (#1179):**  Optimized resource allocation.

* **Only log mlflow dataset if it hasn't been logged before. (#1180):**  Improved MLflow logging efficiency.

* **Feat/archive mlflow runs (#1181):**  Implemented MLflow run archiving.

* **Revert window size to 10 for Node2Vec Embeddings (#1184):**  Restored a previous parameter value for Node2Vec.

* **Add rank columns (#1186):** Added rank information for better result interpretation.

* **Reduce resource requirements for edge and node ingestion into Neo4j. (#1195):**  Optimized Neo4j ingestion resource usage.

* **Debug/expand mlflow hook logging (#1204):** Expanded MLflow hook logging for debugging.


## Documentation Improvements

* **Release Article Date Fix (#796), MathJax Support (#796), Google Analytics Integration (#796):** Improved release articles and documentation presentation.

* **MOA Codebase Documentation (#798), VS Code Debugging Documentation (#799):** Enhanced developer documentation.

* **Virtual Environment Documentation Update (#906), Kedro Resource Documentation (#919):** Improved onboarding and user guides.

* **Onboarding Documentation Fixes (#883, #902):** Addressed onboarding material clarity.

* **Common Errors Documentation Update (#925):**  Added troubleshooting guidance.

* **libomp Installation Instructions (#934), pyenv Installation Instructions (#812), SILC Troubleshooting Documentation (#836):** Expanded installation and troubleshooting instructions.

* **Docs cleanup (#1150):** General documentation improvements.

* **Format kedro experiment docs (#1159):** Improved experiment documentation formatting.

* **Improve sampling documentation with release specific instructions (#1166):**  Enhanced sampling documentation clarity.

* **Add documentation for explaining more tags over releases (#1209):** Improved documentation for disease list subset generation.

* **Define process to fix a KG release (#1207):**  Added a process for handling corrupted KG releases.

* **Improved model evaluation documentation (#905, #878):** Improved documentation for ensemble function addition and Neo4j SSL setup.


