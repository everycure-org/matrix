---
title: v0.15.0
draft: false
date: 2026-02-02
categories:
  - Release
authors:
  - everycure
  - JacquesVergine
  - may-lim
  - jdr0887
  - kevinschaper
  - Dashing-Nelson
  - matentzn
  - piotrkan
  - amyford
  - pascalwhoop
  - alexeistepa
  - eKathleenCarter
---

## Summary

Version 0.15.0 represents a major consolidation and quality improvement release. The most significant change is the migration of the core entities pipeline into the matrix monorepo, bringing disease and drug list generation under unified infrastructure. This release also introduces the new matrix-validator library for comprehensive knowledge graph validation, upgrades to WHO-standard drug classification, and adds important new metrics like JaM and connectivity scoring. Infrastructure improvements focus on scaling, reliability, and operational excellence with enhanced CI/CD workflows and better resource management.

<!-- TODO: Add any additional context about deployment considerations or migration steps if needed -->

### Breaking Changes 🛠

No breaking changes in this release.

### Exciting New Features 🎉

- **Core Entities Pipeline Migration**: Major milestone - brought the core entities pipeline into the matrix monorepo, consolidating disease and drug list generation with the main pipeline infrastructure. This includes comprehensive Mondo disease ontology processing, LLM-based disease categorization, and WHOCC drug classification. [#2008](https://github.com/everycure-org/matrix/pull/2008)

- **Mondo-Disease List Refactor**: End-to-end refactoring of Mondo disease processing, integrating it with the core entities disease list pipeline for improved consistency. [#2064](https://github.com/everycure-org/matrix/pull/2064), [#2040](https://github.com/everycure-org/matrix/pull/2040)

- **JaM dataset (Jane and May) Integration**: Added JaM to both the evaluation suite and run comparison pipeline, providing a new metric for assessing prediction quality and model performance across different runs. [#1993](https://github.com/everycure-org/matrix/pull/1993), [#2000](https://github.com/everycure-org/matrix/pull/2000)

- **Neo4j Knowledge Graph Enrichment**: Enhanced the Neo4j KG with additional attributes valuable to the medical team, including drug-specific properties and improved metadata. [#2026](https://github.com/everycure-org/matrix/pull/2026)

- **Knowledge Graph Validator Migration**: Comprehensive migration of the KG validation system to a new matrix-validator library with Polars-based validation, improving performance and maintainability. The validator now includes extensive checks for Biolink model compliance, CURIE validation, and edge type verification. [#1987](https://github.com/everycure-org/matrix/pull/1987)

- **Known Entity Removal Filter**: Implemented a dataset-based filtering system to remove known drug-disease associations from evaluation sets, enabling better assessment of true novel predictions. [#1973](https://github.com/everycure-org/matrix/pull/1973), [#1984](https://github.com/everycure-org/matrix/pull/1984)

- **Knowledge Graphs Dashboard Pages**: New dashboard pages for exploring and analyzing knowledge graph structure, sources, and content. [#2001](https://github.com/everycure-org/matrix/pull/2001)

- **WHO Collaborating Centre (WHOCC) ATC Code Integration**: Upgraded drug ATC code sourcing to use the authoritative WHO Collaborating Centre database instead of relying solely on DrugBank. This ensures more accurate and up-to-date drug classification data, with enhanced error logging and synonym tracking. [#2045](https://github.com/everycure-org/matrix/pull/2045)

- **EC Core Connectivity Metrics**: Implemented comprehensive connectivity metrics for evaluating knowledge graph completeness and quality based on core entities. [#1956](https://github.com/everycure-org/matrix/pull/1956)

- **HuggingFace Hub Upload Pipeline**: Added new pipeline for managing dataset uploads to HuggingFace Hub. [#1967](https://github.com/everycure-org/matrix/pull/1967)

### Experiments 🧪

- **Embiology Experiments**: This captures various experiments with MATRIX pipeline and Embiology KG to assess its value in drug repurposing. [link to report](https://github.com/everycure-org/lab-notebooks/blob/03773f8c1ad9e20f2eaeafda6d14c3835ed82643/cross-kg/embiology/embiology_matrix.ipynb)

- **ESM2 Embeddings**: Experiment training a classifier for drug-target predictions using embeddings from ESM2 [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/3515b6197bfc1d60e678950a8f12c944912d4296/uab-new-models-for-improved-performance/experiment-december-2025/UAB_Michel_Reproduction_Embeddings_Experiment_Report.ipynb)

- **Patent Scraping Part 4**: The experiment is intended to explore improved text phrase-to-CURIE resolution for Patents [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/c6ad78cd0097f6839f2052adabbbced1b6d91dc1/uab-LLM-patent-scraping/experiment-december-2025/UAB_Patent_Text_Improved_Phrase_to_CURIE_Experiment_Report.ipynb) 

-**SPOKE Experiments**: This captures various experiments with MATRIX pipeline and SPOKE KG to assess its value in drug repurposing. [link to report](https://github.com/everycure-org/lab-notebooks/blob/03773f8c1ad9e20f2eaeafda6d14c3835ed82643/piotr/spoke/spoke_exp.ipynb) 

- **Final Cross-KG Experiments**: Cross-KG benchmark with MATRIX pipeline and all usable KGs within our system, including aggregated score combining all matrix predictions. [link to report](https://github.com/everycure-org/lab-notebooks/pull/286/changes#diff-6f9510cf6c45eb80444cf45d09a6c12f5ff0bfe335532a50a08b9d49e237aac0)



No experiment reports in this release.

### Infrastructure 🏗️

- **Cloud Build IAM Roles**: Added IAM roles for data-science group to access Cloud Build service account. [#2041](https://github.com/everycure-org/matrix/pull/2041)

- **GKE Node Pool Scaling**: Increased node pool sizes to support up to 80 nodes for n2d and standard configurations. [#2025](https://github.com/everycure-org/matrix/pull/2025)

- **CI Workflow Path Updates**: Adjusted CI workflow trigger paths and concurrency filters. [#2016](https://github.com/everycure-org/matrix/pull/2016)

- **Neo4j Certificate Refresh CronJob**: Enhanced Neo4j certificate refresh automation using rollout restart and improved error handling. [#2006](https://github.com/everycure-org/matrix/pull/2006)

- **Argo Workflow Resource Configuration**: Updated resource allocations and volume mounts in Argo workflow templates. [#2002](https://github.com/everycure-org/matrix/pull/2002), [#2003](https://github.com/everycure-org/matrix/pull/2003)

- **GitHub Actions Runner Scale Set Updates**: Updated gha-runner-scale-set-controller and gha-runner-scale-set to version 0.13.0. [#1991](https://github.com/everycure-org/matrix/pull/1991)

- **Logging Exclusions**: Added severity-based log exclusions (DEFAULT and NOTICE) for hub environments to reduce log noise. [#2050](https://github.com/everycure-org/matrix/pull/2050)

- **Disk Size Increase**: Enlarged disk allocations for compute resources. [#1978](https://github.com/everycure-org/matrix/pull/1978)

- **Vertex AI Workbench Cleanup**: Removed stale Vertex AI Workbench instances. [#2033](https://github.com/everycure-org/matrix/pull/2033)

### Bugfixes 🐛

- **Spark Checkpoint Directory Configuration**: Added ability to configure Spark checkpoint directories, improving reliability of long-running Spark jobs. [#1979](https://github.com/everycure-org/matrix/pull/1979)

- **Disease Category Version Update**: Fixed disease category file version mismatch. [#2063](https://github.com/everycure-org/matrix/pull/2063)

- **Automated Sampling Pipeline Removal**: Disabled problematic scheduled sampling runs that were causing issues. [#2054](https://github.com/everycure-org/matrix/pull/2054)

- **LLM Token Tuple Parsing**: Fixed core-entities CI failures due to incorrect tuple parsing in LLM token handling. [#2046](https://github.com/everycure-org/matrix/pull/2046)

- **BigQuery Dataset Name Sanitization**: Fixed bug where dataset names weren't properly sanitized before initialization in custom BQ Kedro datasets. [#2039](https://github.com/everycure-org/matrix/pull/2039)

- **Matrix Tag Version Parsing**: Corrected semantic version extraction logic from matrix release tags. [#2036](https://github.com/everycure-org/matrix/pull/2036), [#2032](https://github.com/everycure-org/matrix/pull/2032), [#2030](https://github.com/everycure-org/matrix/pull/2030)

- **Documentation Script Imports**: Fixed broken import statements in documentation generation scripts. [#2034](https://github.com/everycure-org/matrix/pull/2034)

- **Core Entities Release Comparison**: Fixed comparison logic in core entities release workflow. [#2023](https://github.com/everycure-org/matrix/pull/2023)

- **Dataset Release Naming**: Corrected dataset release name for all_pks_document. [#2022](https://github.com/everycure-org/matrix/pull/2022)

- **Core Entities GitHub Actions**: Fixed various issues in core entities CI/CD workflows. [#2015](https://github.com/everycure-org/matrix/pull/2015)

- **SparkSession Active Session Bug**: Resolved issue where SparkSession.getActiveSession() was returning None in connectivity metrics calculations. [#1985](https://github.com/everycure-org/matrix/pull/1985)

- **Core Entity Categories Preservation**: Fixed bug where core entity categories were being lost during node integration, ensuring disease and drug categories are properly maintained throughout the pipeline. [#1983](https://github.com/everycure-org/matrix/pull/1983)

### Technical Enhancements 🧰

- **Disease LLM Categorization**: Moved LLM-generated disease columns to a dedicated disease categories pipeline component. [#2012](https://github.com/everycure-org/matrix/pull/2012)

- **matrix-schema Package Migration**: Migrated matrix-schema dependency into the monorepo for better version control and consistency. [#2027](https://github.com/everycure-org/matrix/pull/2027)

- **Unified Nodes Dataset Rename**: Renamed integration.prm.unified_nodes datasets to include @spark suffix for clearer identification. [#1982](https://github.com/everycure-org/matrix/pull/1982)

- **Evaluation Pipeline EC_ID Join Refactor**: Refactored evaluation pipeline to support EC_ID-based joins, improving data lineage and traceability. [#1992](https://github.com/everycure-org/matrix/pull/1992)

- **GraphFrame SparkSession Race Condition Fix**: Resolved concurrent SparkSession initialization issues in parallel Kedro execution with GraphFrames. [#2009](https://github.com/everycure-org/matrix/pull/2009)

- **ATC Code Information Enhancement**: Added ATC name and synonym information to drug list pipeline with improved error logging for WHOCC data retrieval. [#2059](https://github.com/everycure-org/matrix/pull/2059)

- **DrugBank Prefix Updates**: Updated references to use consistent DrugBank prefix formatting. [#2055](https://github.com/everycure-org/matrix/pull/2055)

### Documentation ✏️

- **ROBOKOP License Link Update**: Updated broken ROBOKOP license link in documentation. [#2061](https://github.com/everycure-org/matrix/pull/2061)

- **EC Drugs List Curated Annotations**: Added comprehensive documentation on curated annotations in the Every Cure drugs list. [#1980](https://github.com/everycure-org/matrix/pull/1980)

### Other Changes

- **Core Entities Release PRs**: Multiple release-related PRs for the core entities pipeline (disease_list and drug_list releases). [#2064](https://github.com/everycure-org/matrix/pull/2064), [#2048](https://github.com/everycure-org/matrix/pull/2048), [#2024](https://github.com/everycure-org/matrix/pull/2024), [#2019](https://github.com/everycure-org/matrix/pull/2019)

