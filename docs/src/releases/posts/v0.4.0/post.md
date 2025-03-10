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

## Enhanced Pipeline Control and Monitoring

* **Sample Pipeline:**  A new sample pipeline enables running pipelines on a representative subset of real data. This facilitates faster development cycles and increased confidence in pipeline functionality before full data runs.  Scheduled sample data generation (#1105) further supports development workflows.

* **Ability to specify mlflow experiment by name (#1093):** Workflow submission is simplified with the ability to specify MLflow experiments by name.

* **Add --nodes to Kedro submit (#1142):**  Granular control over pipeline execution is now possible through the command line, allowing for the execution of specific nodes within a pipeline.
* **Expose integration pipeline's datasets in BigQuery (#1076):**enable direct SQL querying in in BQ and monitoring in [KG dashboard](https://data.dev.everycure.org/versions/latest/evidence/) for better QC and debugging capabilities.
## Data Integration and Management

* **Move de-duplication to integration from preprocessing (#1118):**  Pipeline efficiency is improved by shifting deduplication of custom datasets (provided by EC medical team) to the integration stage.

* **Add upstream data source to Neo4j edges (#1131):** Data provenance tracking is enhanced within Neo4j by adding the upstream data source information to edges.

## Bug Fixes

* **Fix integration pipeline error with missing interpolation key (#1123):** Corrected an error in the integration pipeline related to missing interpolation keys.

* **Fix writing to the gsheets for SILC sheet (#1193):** Ensured correct data reporting to Google Sheets for SILC.

## Technical Enhancements

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

* **Docs cleanup (#1150):** General documentation improvements.

* **Format kedro experiment docs (#1159):** Improved experiment documentation formatting.

* **Improve sampling documentation with release specific instructions (#1166):**  Enhanced sampling documentation clarity.

* **Add documentation for explaining more tags over releases (#1209):** Improved documentation for disease list subset generation.

* **Define process to fix a KG release (#1207):**  Added a process for handling corrupted KG releases.


