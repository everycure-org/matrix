---
title: v0.10.0
draft: false
date: 2025-09-05
categories:
  - Release
authors:
  - matentzn
  - Dashing-Nelson
  - amyford
  - piotrkan
  - JacquesVergine
  - eKathleenCarter
  - pascalwhoop
  - kevinschaper
  - lvijnck
  - jdr0887
---

### Breaking Changes 🛠

- **Migration to UV Package Manager**: Major dependency management overhaul replacing requirements.txt with UV workspace. This introduces a new workspace structure with individual libraries (`matrix-auth`, `matrix-fabricator`, `matrix-gcp-datasets`, `matrix-mlflow-utils`) extracted from the main pipeline. This change improves dependency isolation and build times but requires developers to use `uv sync` instead of `pip install -r requirements.txt` 
  [#1768](https://github.com/everycure-org/matrix/pull/1768)

### Exciting New Features 🎉

- **Enhanced Validation for Fabricator Pipeline**: Added comprehensive data validation to the fabricator pipeline using Pandera schemas, improving data quality assurance and early error detection during synthetic data generation
  [#1714](https://github.com/everycure-org/matrix/pull/1714)

- **DrugBank & EC Ground Truth Lists Integration**: Integrated authoritative drug and indication lists from DrugBank and Every Cure, expanding the knowledge base with high-quality ground truth data for improved drug repurposing predictions
  [#1763](https://github.com/everycure-org/matrix/pull/1763)

- **EC Indication List Ingestion**: Added support for ingesting Every Cure's curated indication list, providing additional ground truth data for model training and validation
  [#1787](https://github.com/everycure-org/matrix/pull/1787)

- **Docker Image Cleanup Automation**: Implemented automated cleanup of Docker images on workflow success, reducing storage costs and improving resource management in the CI/CD pipeline
  [#1805](https://github.com/everycure-org/matrix/pull/1805)

### Experiments 🧪

- **Features and Modelling Integration**: Added features and modelling components to the weekly pipeline run, enabling regular evaluation of model performance and feature engineering improvements
  [#1631](https://github.com/everycure-org/matrix/pull/1631)

### Bugfixes 🐛

- **Neo4j Topological Embeddings Fix**: Resolved critical issue in Neo4j configuration that was preventing proper generation of topological embeddings, restoring graph-based feature extraction capabilities
  [#1815](https://github.com/everycure-org/matrix/pull/1815)

- **Module Name Correction**: Fixed incorrect module names that were causing import errors in production deployments
  [#1821](https://github.com/everycure-org/matrix/pull/1821)

- **Release Process UV Command Issue**: Fixed missing UV command in the automated release process that was preventing proper dependency resolution during release builds
  [#1825](https://github.com/everycure-org/matrix/pull/1825)

- **BigQuery SQL Query Fixes**: Corrected broken SQL queries in the KG dashboard that were preventing proper data visualization and reporting
  [#1808](https://github.com/everycure-org/matrix/pull/1808)

- **Node Normalization Error Logging**: Improved error logging in core node normalization process to provide better debugging information when data processing fails
  [#1806](https://github.com/everycure-org/matrix/pull/1806)

- **Ground Truth Table Names Update**: Updated ground truth table references in the KG dashboard to match the new table naming conventions
  [#1817](https://github.com/everycure-org/matrix/pull/1817)

- **Release History Page Fix**: Fixed broken release history page generation and display, ensuring proper documentation of version history
  [#1792](https://github.com/everycure-org/matrix/pull/1792)

- **Requirements.txt Synchronization**: Fixed synchronization issues with requirements.txt to ensure consistent dependency versions across environments
  [#1774](https://github.com/everycure-org/matrix/pull/1774)

- **Documentation .gitignore Fix**: Added docs data directory to .gitignore to prevent accidental commit of generated documentation files
  [#1828](https://github.com/everycure-org/matrix/pull/1828)

### Technical Enhancements 🧰

- **Spot Instance Implementation**: Migrated MATRIX pipeline runs to GKE Spot Instances with fallback mechanisms, reducing infrastructure costs by up to 80% while maintaining reliability
  [#1771](https://github.com/everycure-org/matrix/pull/1771), [#1788](https://github.com/everycure-org/matrix/pull/1788)

- **Artifact Registry with Cleanup Policies**: Added comprehensive Artifact Registry module with automated cleanup policies and documentation, improving container image lifecycle management
  [#1717](https://github.com/everycure-org/matrix/pull/1717)

- **GKE Node Capacity Increase**: Bumped GKE node disk size to 1.5TB and disabled image deletion policy to support larger workloads and improve storage reliability
  [#1798](https://github.com/everycure-org/matrix/pull/1798)

- **Spark Temporary Directory Configuration**: Enhanced Spark configuration with proper temporary directory management, preventing disk space issues during large data processing jobs
  [#1816](https://github.com/everycure-org/matrix/pull/1816)

- **Enhanced Node Deduplication**: Improved category assignment logic in node deduplication process, resulting in better data quality and reduced redundancy
  [#1786](https://github.com/everycure-org/matrix/pull/1786)

- **Matrix Transformations Output Repartitioning**: Optimized data partitioning for matrix transformation outputs, improving processing performance and reducing memory pressure
  [#1726](https://github.com/everycure-org/matrix/pull/1726)

- **Ephemeral Volume Management**: Created generic ephemeral volumes with persistent disk CSI tied to pods, improving storage performance and cost efficiency
  [#1799](https://github.com/everycure-org/matrix/pull/1799)

- **Argo Workflows Archive Logging**: Enabled archive logs for Argo Workflows controller, improving debugging capabilities and workflow monitoring
  [#1795](https://github.com/everycure-org/matrix/pull/1795)

- **Enhanced Monitoring Configuration**: Updated kube-state-metrics configuration to include pod containers in metric labels, providing better observability
  [#1733](https://github.com/everycure-org/matrix/pull/1733)

- **Dynamic Ground Truth Ingestion**: Made ground truth data ingestion more dynamic and configurable, allowing for easier addition of new data sources
  [#1766](https://github.com/everycure-org/matrix/pull/1766)

- **Weekly Dependency Updates**: Added automated weekly workflow to update MATRIX dependencies, ensuring security patches and performance improvements are regularly applied
  [#1775](https://github.com/everycure-org/matrix/pull/1775)

- **Cost Optimization Infrastructure**: Multiple cost-cutting measures including removal of local SSDs, backup agent configuration optimization, and improved resource allocation
  [#1796](https://github.com/everycure-org/matrix/pull/1796), [#1731](https://github.com/everycure-org/matrix/pull/1731)

### Documentation ✏️

- **Installation Instructions Update**: Enhanced Linux installation guide with pyenv setup steps and improved developer onboarding documentation
  [#1748](https://github.com/everycure-org/matrix/pull/1748)

- **External Contributor Documentation**: Updated documentation to reflect lessons learned from public external contributor testing, improving the contribution experience
  [#1764](https://github.com/everycure-org/matrix/pull/1764)

### Other Changes

- **Neo4j Ingestion Optimization**: Modified Neo4j ingestion to only occur on monthly minor releases, reducing resource usage and improving pipeline efficiency
  [#1823](https://github.com/everycure-org/matrix/pull/1823)

- **BigQuery Output Optimization**: Only write final filtered tables to BigQuery, reducing storage costs and improving query performance
  [#1819](https://github.com/everycure-org/matrix/pull/1819)

- **Orchard Feedback Data Integration**: Updated orchard transformer to map feedback data to MATRIX format, enabling integration of external validation data
  [#1782](https://github.com/everycure-org/matrix/pull/1782)

- **BigQuery Access Permissions**: Allowed MATRIX PROD environment to access Orchard Datasets in BigQuery for cross-project data integration
  [#1803](https://github.com/everycure-org/matrix/pull/1803)

- **Clinical Trials Data Migration**: Moved Clinical Trials and off-label data to public datasets, improving data accessibility and compliance
  [#1760](https://github.com/everycure-org/matrix/pull/1760)

- **Payload Size Optimization**: Increased payload size limits and fixed string conversion issues for better data handling capacity
  [#1773](https://github.com/everycure-org/matrix/pull/1773), [#1776](https://github.com/everycure-org/matrix/pull/1776)

- **Orchard Feedback Dataset Addition**: Added Orchard feedback dataset integration for external validation and feedback loop improvements
  [#1740](https://github.com/everycure-org/matrix/pull/1740)

- **PySpark Version Update**: Updated PySpark to version 3.5.6 for improved performance and bug fixes
  [#1753](https://github.com/everycure-org/matrix/pull/1753)

- **Disease List Ingestion Refactor**: Refactored disease list ingestion to use pandas.CSVDataset for better data handling and validation
  [#1750](https://github.com/everycure-org/matrix/pull/1750)

- **ARGO Configuration for Stability Pipeline**: Added ARGO configuration to core stability pipeline for better workflow management
  [#1747](https://github.com/everycure-org/matrix/pull/1747)

- **Node Category Filtering**: Added node category filters to the filtering pipeline, improving data quality and reducing noise
  [#1730](https://github.com/everycure-org/matrix/pull/1730)

- **Release History Link**: Added release history link to KG dashboard home page for better user navigation
  [#1790](https://github.com/everycure-org/matrix/pull/1790)

- **Sampling Pipeline Schedule**: Modified sampling pipeline to run only on weekdays, optimizing resource usage
  [#1804](https://github.com/everycure-org/matrix/pull/1804)

- **Platform Documentation**: Added comprehensive platform refactor and standardization documentation
  [#1706](https://github.com/everycure-org/matrix/pull/1706)