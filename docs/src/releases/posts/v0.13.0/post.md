---
title: v0.13.0
draft: false
date: 2025-12-03
categories:
  - Release
authors:
  - JacquesVergine
  - Dashing-Nelson
  - matentzn
  - lvijnck
  - jdr0887
  - eKathleenCarter
  - alexeistepa
  - may-lim
  - kevinschaper
  - pascalwhoop
  - leelancashire
---

### Breaking Changes 🛠

No breaking changes in this release.

### Exciting New Features 🎉

- **Run Comparison Pipeline**: Added a comprehensive run comparison pipeline that allows comparing multiple model runs with sophisticated evaluation metrics including recall@n, AUPRC, precision-recall curves, and Kendall rank correlation. This enables systematic comparison of different model configurations and embeddings across multiple folds with uncertainty estimation [#1890](https://github.com/everycure-org/matrix/pull/1890) [#1905](https://github.com/everycure-org/matrix/pull/1905)

- **Cloud Build for Docker Images**: Implemented Google Cloud Build integration for building Matrix Docker images, enabling automated container builds in the cloud with support for multiple platforms and build caching [#1822](https://github.com/everycure-org/matrix/pull/1822)

- **ROBOKOP Preprocessing Pipeline**: Added a new preprocessing pipeline specifically for ROBOKOP knowledge graph data, including normalization and data transformation steps that integrate with the existing ingestion workflow [#1904](https://github.com/everycure-org/matrix/pull/1904)

- **Evaluation Pipeline Enhancement**: Enhanced evaluation pipeline to merge on drug `ec_id` instead of translator ID, improving consistency with Every Cure's internal drug identification system [#1949](https://github.com/everycure-org/matrix/pull/1949)

- **Knowledge Graph Catalog Dataset**: Introduced a new catalog dataset system with `MultiPredictionsDataset` and enhanced storage utilities for managing multiple prediction matrices across different runs and folds [#1947](https://github.com/everycure-org/matrix/pull/1947)

- **Disease and Drug Version Bump**: Updated to latest versions of disease and drug lists, ensuring the pipeline uses the most current curated data [#1931](https://github.com/everycure-org/matrix/pull/1931)

### Experiments 🧪

No experiment reports in this release.

### Bugfixes 🐛

- **EC Clinical Trial Ingestion**: Fixed EC clinical trial data ingestion to properly handle parquet file format, resolving issues with data loading [#1972](https://github.com/everycure-org/matrix/pull/1972)

- **Evaluation Suite Revert**: Reverted evaluation suite to use `translator_id` for certain operations where the previous change caused compatibility issues [#1966](https://github.com/everycure-org/matrix/pull/1966)

- **Drug and Disease List Ingestion**: Fixed ingestion pipeline for drug and disease lists to properly handle updated data formats and ensure data consistency [#1942](https://github.com/everycure-org/matrix/pull/1942)

- **HPO Mappings**: Corrected Human Phenotype Ontology (HPO) mappings to improve accuracy of phenotype-disease associations [#1954](https://github.com/everycure-org/matrix/pull/1954)

### Technical Enhancements 🧰

- **CI Runtime Improvements**: Significantly improved continuous integration pipeline runtime by optimizing test execution and Docker operations, including running Kedro tests with ThreadRunner configuration [#1958](https://github.com/everycure-org/matrix/pull/1958) [#1961](https://github.com/everycure-org/matrix/pull/1961)

- **Topological Embeddings Resilience**: Made topological embeddings generation resilient to Google Cloud spot instance failures through improved retry logic and checkpointing [#1957](https://github.com/everycure-org/matrix/pull/1957)

- **LiteLLM Provider Expansion**: Added support for Gemini models and Anthropic provider in LiteLLM configuration, plus support for fine-tuned models, expanding the range of LLM options available [#1951](https://github.com/everycure-org/matrix/pull/1951) [#1946](https://github.com/everycure-org/matrix/pull/1946) [#1955](https://github.com/everycure-org/matrix/pull/1955)

- **LiteLLM Caching Investigation**: Investigated and addressed caching issues with the response API to improve reliability of LLM interactions [#1941](https://github.com/everycure-org/matrix/pull/1941)

- **XGBoost Parallelism**: Updated XGBoost configuration for improved parallelism and more accurate CPU count detection, optimizing model training performance [#1923](https://github.com/everycure-org/matrix/pull/1923)

- **GPU Removal**: Removed GPU usage from the pipeline, simplifying infrastructure requirements and reducing costs while maintaining performance through CPU optimizations [#1869](https://github.com/everycure-org/matrix/pull/1869)

- **Knowledge Graph Dashboard Enhancements**: Added key node pages and improved Knowledge Level and Agent Type queries in the Evidence.dev dashboard, plus ABox/TBox information display [#1887](https://github.com/everycure-org/matrix/pull/1887) [#1928](https://github.com/everycure-org/matrix/pull/1928) [#1930](https://github.com/everycure-org/matrix/pull/1930)

- **Unified Normalization Stats**: Updated dashboard to use `unified_normalization_summary` for more consistent normalization statistics display [#1892](https://github.com/everycure-org/matrix/pull/1892)

- **Kedro Version Bump**: Upgraded Kedro to version 0.19.15 for improved pipeline execution performance [#1940](https://github.com/everycure-org/matrix/pull/1940)

- **PandasBQDataset Simplification**: Removed shard parameter from PandasBQDataset for cleaner BigQuery dataset handling [#1939](https://github.com/everycure-org/matrix/pull/1939)

- **Logging Cleanup**: Removed redundant `logging.basicConfig` calls throughout the codebase to prevent logging configuration conflicts [#1959](https://github.com/everycure-org/matrix/pull/1959)

- **Neo4j Query Logging**: Enabled Neo4j query logging by default for better debugging and performance monitoring [#1906](https://github.com/everycure-org/matrix/pull/1906)

- **IAM Enhancements**: Added GitHub Actions service account with read access to dev bucket from prod environment for improved CI/CD workflows [#1926](https://github.com/everycure-org/matrix/pull/1926)

- **Dockerfile Optimization**: Updated Dockerfile to include README and src directory for better package builds [#1948](https://github.com/everycure-org/matrix/pull/1948)

### Documentation ✏️

- **LiteLLM Provider Guide**: Added comprehensive guide for adding new LLM providers to LiteLLM, including step-by-step instructions and usage documentation updates [#1964](https://github.com/everycure-org/matrix/pull/1964)

- **EC Drug List Documentation**: Added detailed documentation for the Every Cure drug list, explaining its structure and usage within the pipeline [#1925](https://github.com/everycure-org/matrix/pull/1925)

- **Run Comparison Pipeline Documentation**: Added comprehensive documentation for the new run comparison pipeline, including usage examples and metric explanations [TODO: verify this was added in this release]

- **Hyperparameter Tuning Guide**: Added documentation on making hyperparameter tuning CPU-first, reflecting the infrastructure changes [TODO: verify completeness]

- **CMake Installation Guide**: Added FAQ entry documenting CMake installation requirements for XGBoost on different platforms [#1935](https://github.com/everycure-org/matrix/pull/1935)

- **Drug List Version Documentation**: Updated drug list documentation to remove hardcoded version numbers, making maintenance easier [#1974](https://github.com/everycure-org/matrix/pull/1974) [#1934](https://github.com/everycure-org/matrix/pull/1934)

- **Python Version Bump**: Upgraded documentation site to Python 3.13 for latest features and performance improvements [#1933](https://github.com/everycure-org/matrix/pull/1933)

### Other Changes

- Updated subproject commit reference in infra/secrets [#1922](https://github.com/everycure-org/matrix/pull/1922)

- Internal tooling improvements for tracking MLflow experiments and runs over time [#1963](https://github.com/everycure-org/matrix/pull/1963)
