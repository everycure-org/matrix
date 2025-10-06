---
title: v0.11.0
draft: false
date: 2025-10-04
categories:
  - Release
authors:
  - Dashing-Nelson
  - JacquesVergine
  - piotrkan
  - pascalwhoop
  - matentzn
  - lvijnck
  - kevinschaper
  - eKathleenCarter
  - jdr0887
  - amyford
---

## Exciting New Features 🎉

- **Multi-Model Training Pipeline**: Added support for training multiple models in a single pipeline run, enabling comprehensive model comparison and selection workflows. Configure models via the new `models` parameter in the modelling configuration. [#1843](https://github.com/everycure-org/matrix/pull/1843)

- **LiteLLM Gateway Infrastructure**: Deployed LiteLLM as a unified API gateway for managing multiple LLM providers (OpenAI, Anthropic, etc.) on Kubernetes. Includes Redis caching, PostgreSQL for analytics, and comprehensive admin/user documentation. [#1845](https://github.com/everycure-org/matrix/pull/1845)

- **PrimeKG Integration**: Integrated PrimeKG as a new knowledge source into the MATRIX pipeline. PrimeKG provides precision medicine knowledge with 129K nodes and 4M+ edges covering diseases, drugs, proteins, and biological pathways. [#1793](https://github.com/everycure-org/matrix/pull/1793)

- **Branching from Previous Runs**: Added `--from-run` CLI parameter enabling pipeline execution to pull specific inputs from a previous pipeline run, allowing efficient branching and iterative experimentation without recomputing earlier pipeline stages. [#1769](https://github.com/everycure-org/matrix/pull/1769)

- **KG Release Trends Dashboard**: Created interactive dashboard page showing knowledge graph statistics and trends across MATRIX releases, providing insights into KG growth and evolution over time. [#1830](https://github.com/everycure-org/matrix/pull/1830)

- **Interactive Knowledge Source Network Graph**: Replaced static knowledge source flow diagram with custom ECharts-based interactive network visualization, enabling dynamic exploration of primary knowledge sources and their relationships. [#1837](https://github.com/everycure-org/matrix/pull/1837)

- **Edge Predicate Navigation**: Added comprehensive edge predicate pages to KG dashboard with links to individual predicate statistics, counts, and examples. [#1809](https://github.com/everycure-org/matrix/pull/1809)

- **Validator Library for Data Integrity**: Created `matrix-pandera` library with reusable validation framework for ensuring data quality across ingestion, fabrication, and processing pipelines. [#1853](https://github.com/everycure-org/matrix/pull/1853)

- **Inject Library for Cross-Repository Code Reuse**: Extracted dependency injection utilities into standalone `matrix-inject` library for sharing configuration patterns across MATRIX repositories. [#1853](https://github.com/everycure-org/matrix/pull/1853)

- **Benchmark Release Link**: Added link to benchmark release page from KG dashboard. [#1827](https://github.com/everycure-org/matrix/pull/1827)

## Technical Enhancements 🧰

- **Updated Baseline Model**: Main branch now reflects new baseline model using integrated knowledge graph (RTX-KG2 + ROBOKOP) embeddings, improving model performance and reproducibility. [#1875](https://github.com/everycure-org/matrix/pull/1875)

- **Primary Knowledge Sources Tracking**: Added column to edges collecting all primary knowledge sources, improving provenance tracking and source attribution throughout the pipeline. [#1813](https://github.com/everycure-org/matrix/pull/1813)

- **MLflow Retry Logic**: Implemented retry mechanism for nodes when MLflow URL lookups fail, improving pipeline resilience to transient network issues. [#1866](https://github.com/everycure-org/matrix/pull/1866)

- **CI Optimization with Self-Hosted Runners**: Deployed GitHub Actions self-hosted runners on Kubernetes using Actions Runner Controller (ARC), significantly reducing CI costs and improving build performance. [#1812](https://github.com/everycure-org/matrix/pull/1812)

- **CloudNativePG PostgreSQL Infrastructure**: Deployed PostgreSQL using CloudNativePG operator on Kubernetes for LiteLLM and other services, providing production-grade database management with automated backups and high availability. [#1845](https://github.com/everycure-org/matrix/pull/1845)

- **Redis Operator Deployment**: Added Redis operator infrastructure for caching and session management, supporting LiteLLM gateway and future service requirements. [#1845](https://github.com/everycure-org/matrix/pull/1845)

- **LiteLLM Model Additions**: Added support for GPT-4 Mini and Claude Haiku models in LiteLLM configuration, expanding available model options for experimentation. [#1874](https://github.com/everycure-org/matrix/pull/1874)

- **Enhanced Google Sheets Integration**: Fixed Google Sheets dataset to properly handle worksheet selection by gid and added error handling for missing gids. [#1858](https://github.com/everycure-org/matrix/pull/1858) [#1860](https://github.com/everycure-org/matrix/pull/1860) [#1862](https://github.com/everycure-org/matrix/pull/1862)

- **Fabricator Improvements**: Updated data fabrication pipeline with enhanced KG generation, improved test coverage, and validator integration for data quality assurance. [#1807](https://github.com/everycure-org/matrix/pull/1807)

- **KG Dashboard Normalized Table Simplification**: Simplified normalized nodes/edges tables in KG dashboard for better query performance and data accessibility. [#1861](https://github.com/everycure-org/matrix/pull/1861)

- **Workflow Spec Pod Rejection Handling**: Fixed regex patterns for detecting pod rejection messages in Argo workflow specifications, improving workflow error handling. [#1863](https://github.com/everycure-org/matrix/pull/1863)

- **Enhanced Node Pool Configuration**: Updated management node pool to n2-standard-16 machine type for improved cluster management performance. [#1873](https://github.com/everycure-org/matrix/pull/1873)


## Documentation ✏️

- **Attribution Documentation**: Added comprehensive attribution documentation for the MATRIX project, acknowledging all knowledge sources, tools, and contributors. [#1867](https://github.com/everycure-org/matrix/pull/1867)

- **Primary Knowledge Sources Reference**: Created detailed documentation page describing primary knowledge sources used in MATRIX, including RTX-KG2, ROBOKOP, PrimeKG, and their characteristics. [#1829](https://github.com/everycure-org/matrix/pull/1829)

- **LiteLLM Deployment ADR**: Documented architectural decision to deploy LiteLLM on Kubernetes for unified LLM API management. [#1834](https://github.com/everycure-org/matrix/pull/1834)

- **LiteLLM Admin Guide**: Created comprehensive administrator guide for deploying, configuring, and maintaining LiteLLM infrastructure. Located at `docs/src/infrastructure/LiteLLM-Admin-Guide.md`.

- **LiteLLM User Guide**: Wrote user-facing documentation for accessing and using LiteLLM API gateway with examples. Located at `docs/src/infrastructure/LiteLLM-User-Guide.md`.

- **GitHub Actions Runner Controller Guide**: Documented setup and deployment of self-hosted GitHub runners using ARC on Kubernetes. Located at `docs/src/infrastructure/ci_optimization_self_hosted_runners.md`.

- **PostgreSQL CloudNativePG Setup**: Created guide for deploying PostgreSQL using CloudNativePG operator. Located at `docs/src/infrastructure/PostgreSQL-CloudNativePG-Setup.md`.

- **Redis Setup Documentation**: Documented Redis operator deployment and configuration. Located at `docs/src/infrastructure/Redis-Setup.md`.

- **Multi-Model Configuration Guide**: Added documentation explaining how to configure and run multiple models in a single pipeline execution. Located at `docs/src/pipeline/multi-model-configuration.md`.

- **Branching from Another Run Guide**: Created walkthrough for using `--from-run` to branch pipeline executions from previous runs. Located at `docs/src/getting_started/walkthroughs/branching_from_another_run.md`.

## Bugfixes 🐛

- **Drug List Version Bump**: Updated to drug list v0.1.4 to fix drug name normalization issues. [#1878](https://github.com/everycure-org/matrix/pull/1878)

- **Pandera Deprecation Warning**: Fixed Pandera deprecation warning by updating API usage to current best practices. [#1856](https://github.com/everycure-org/matrix/pull/1856)

- **Removed KG Validation**: Removed problematic KG validation step that was causing pipeline failures. [#1864](https://github.com/everycure-org/matrix/pull/1864)

- **Make Install Fix**: Removed `make install` target from pipelines/matrix Makefile as it conflicted with workspace-level dependency management. [#1859](https://github.com/everycure-org/matrix/pull/1859)


- **Pod Rejection Regex Fix**: Corrected regex patterns for detecting pod rejection messages in workflow specifications. [#1863](https://github.com/everycure-org/matrix/pull/1863)

- **UV Not Found in Create Draft PR**: Fixed CI workflow failure where uv was not available when creating draft PRs. [#1831](https://github.com/everycure-org/matrix/pull/1831)

## Other Changes

- **KG Dashboard Benchmark Version Update**: Updated benchmark (T3) version reference to v0.10.2 in KG dashboard. [#1877](https://github.com/everycure-org/matrix/pull/1877)

- **PrimeKG Default Color**: Added default color scheme for PrimeKG entities in KG dashboard visualizations. [#1876](https://github.com/everycure-org/matrix/pull/1876)

- **Python Setup and Dependency Improvements**: Updated Python setup action and improved dependency installation workflow in CI. [#1844](https://github.com/everycure-org/matrix/pull/1844)

- **Dependency Updates**: Updated npm and yarn dependencies across services. [#1849](https://github.com/everycure-org/matrix/pull/1849) [#1841](https://github.com/everycure-org/matrix/pull/1841)

- **Removed Update Dependencies Workflow**: Removed automated dependency update workflow to reduce maintenance overhead. [#1839](https://github.com/everycure-org/matrix/pull/1839)


---

**Full Changelog**: [v0.10.0...v0.11.0](https://github.com/everycure-org/matrix/pull/compare/v0.10.0...v0.11.0)
