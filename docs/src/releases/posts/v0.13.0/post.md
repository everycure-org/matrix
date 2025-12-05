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

- **UAB 1 New Model Inital Embeddings**: This experiment was to begin training classifiers based on embeddings from ESM2 and Molecular Transformer. [link to report](https://github.com/everycure-org/lab-notebooks/blob/58d4334234edbc4951f192e4d8d85d0f21961723/uab-new-models-for-improved-performance/experiment-november-2025/UAB_New_Model_Initial_Embeddings_Experiment_Report.ipynb)

- **Patent Scraping Part 2**: Expertiment to determine the ballpark cost/time estimates for running ontology-aligned triple extraction from drug patents at scale using LLM APIs, to guide engineering choices. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/85468dbd75219a637665c6f0eb512435979a9aad/uab-LLM-patent-scraping/experiment-october-2025/UAB_Patent_Scraping_Scaling_Experiment_Report.ipynb)

- **CBR-X Explainer**: Evaluation of a case-based reasoning explainer (CBR-X) for drug–disease link prediction that is designed to be both predictive and mechanistically interpretable. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/1631f27658c9629304433b388743b3ccbcdb87a2/maria/07_triage_yield_over_time/src/plotting.py)

- **Measuring Triage Yield Over Time **: Experiment to assess whether triage yield changes over time and whether model rank explains yield, while accounting for reviewer and item heterogeneity. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/5f7d37deb573c8f36fdd1ff8ef4b6f2ededcafcc/lee/structural-bias/Structural_Bias_T3_Report.ipynb)

- **UAB3: PubMed Abstract Validation Tool Experiment**: Two-Round LLM Pipeline for Validating PubMed Abstract Support of Knowledge Graph Edges. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/afef2bab707a2b82236d16981ddfe0794e706949/uab-pubmed-embeddings/experiment-oct-2025/initial_PubMed_abstracts_validation_report.ipynb)

- **UAB4: PubMed Extension Pipeline**: Pipeline for Automating Literature Support of KG Edges. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/afef2bab707a2b82236d16981ddfe0794e706949/uab-pubmed-embeddings/experiment-oct-2025/initial_PubMed_abstract_extension_report.ipynb)

- **PrimeKG + Matrix Experiment**: Experiment with MATRIX pipeline and PrimeKG, using PrimeKG with disease nodes merged. This experiment explored different settings of Matrix pipeline together with PrimeGT, as well as examination of overfitting/structural bias. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/feat/prime-kg-exp/cross-kg/primekg/primekg_summary_report.ipynb)

- **PrimeKG + Matrix Experiment (Filtering)**: TExperiment with MATRIX pipeline and PrimeKG, using PrimeKG with disease nodes merged.This experiment explored different slices of PrimeKG, using both top-down and down-top approach to filtering. PrimeGT used. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/feat/prime-kg-exp/cross-kg/primekg/primekg_summary_report.ipynb)

- **[XG Synth] PrimeKG + Matrix Experiment (Disease Split)**: Experiment with MATRIX pipeline and PrimeKG, using PrimeKG with disease nodes merged.This experiment explored how is MATRIX pipeline performing in a disease-split setting using PrimeKG knowledge graph and PrimeGT. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/feat/prime-kg-exp/cross-kg/primekg/primekg_summary_report.ipynb)

- **[XG Ensemble] PrimeKG + Matrix Experiment (Disease Split)**: Experiment with MATRIX pipeline and PrimeKG, using PrimeKG with disease nodes merged.This experiment explored how is MATRIX pipeline performing in a disease-split setting using PrimeKG knowledge graph and PrimeGT. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/feat/prime-kg-exp/cross-kg/primekg/primekg_summary_report.ipynb)

- **Patent Scraping Part 3**: Additional Patent Scraping: Test newer Claude models (incl. Opus 4.5) and a lightweight CURIE lookup step. [link to notebook](https://github.com/everycure-org/lab-notebooks/blob/2b2c23239f6e369976126563793398f6a074f2d1/uab-LLM-patent-scraping/experiment-november-2025/UAB_Patent_Semantic_Triple_Experiment_Report.ipynb)

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
