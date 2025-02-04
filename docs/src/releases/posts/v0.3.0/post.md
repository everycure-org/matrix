---
draft: false
date: 2025-02-04
categories:
  - Release
authors:
  - piotrkan
  - lvijnck
  - alexeistepa
  - emil-k
  - JacquesVergine
  - eKathleenCarter
---
# Matrix Platform `v0.2.7`: Enhanced Data Ingestion, MLflow Integration, and Pipeline Refinements

This release of the Matrix Platform introduces a new dataset for simplifying data ingestion from GitHub releases, integrates MLflow logging for improved tracking of datasets, and delivers several bug fixes and technical enhancements across the pipeline.

<!-- more -->

## Key Enhancements

### Simplified GitHub Release Data Ingestion

A new `GitHubReleaseCSVDataset` simplifies the process of ingesting data directly from GitHub releases. This streamlines data acquisition for resources managed on GitHub and reduces the need for manual downloads and updates.  This new dataset was utilized for updating the drug and disease lists used within the platform. (PR #1050)

### Enhanced MLflow Tracking

MLflow now logs the datasets used within the pipeline. This significantly improves the traceability of experiments by recording the exact data versions used for each run. This feature enhances reproducibility and provides valuable context for analyzing experiment results. (PR #1048)

## Bug Fixes

Several critical bugs were addressed in this release, improving the stability and reliability of the platform:

- **Clinical Trial Preprocessing:** Fixed an issue in the clinical trial preprocessing nodes, ensuring the correct handling and preparation of clinical trial data for downstream analysis. (PR #1039)
- **Data Normalization Status:** Corrected the normalizer to accurately reflect the `normalization_success` status, providing more reliable feedback on the data normalization process. (PR #1060)
- **MLflow Metric Tracking:**  Fixed an issue affecting MLflow metric tracking, ensuring accurate recording and reporting of experiment metrics. (PR #1075)

## Technical Enhancements

This release includes several technical enhancements that improve efficiency, robustness, and maintainability:

- **Removal of Hardcoded SILC Configuration (PR #973):** Eliminated hardcoded SILC configuration, improving flexibility and maintainability.
- **Improved Robokop Data Transformation (PR #1032):** Enhanced the efficiency and robustness of Robokop data transformation by requiring a biolink categories dataframe, ensuring more consistent and accurate data processing.
- **Updated Resource Defaults in `kedro4argo_node.py` (PR #1032):** Updated resource defaults from GiB to Gb, improving clarity and consistency in resource allocation.
- **Various Internal Code Simplifications and Improvements:** Several internal code improvements were implemented across multiple files, enhancing code readability, maintainability, and overall quality.

## Other Changes

- A new GPG key (`new-eKathleenCarter.asc`) was added. (PR #1032)
- Several configuration updates were applied to `parameters.yml`, `globals.yml`, and `catalog.yml` files, primarily affecting data sources and pipeline configurations.  These updates reflect changes in data sources, pipeline logic, and resource allocation.


This release of the Matrix Platform delivers valuable improvements in data ingestion, MLflow integration, bug fixes, and technical enhancements.  These changes enhance the platform's robustness, efficiency, and maintainability, paving the way for more advanced features and analyses in future releases.
