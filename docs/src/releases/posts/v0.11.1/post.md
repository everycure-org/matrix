---
title: v0.11.1
draft: false
date: 2025-10-08
categories:
  - Release
authors:
  - Dashing-Nelson
  - JacquesVergine
  - piotrkan
  - pascalwhoop
  - eKathleenCarter
  - matentzn
  - lvijnck
  - kevinschaper
  - jdr0887
  - amyford
---

### Breaking Changes 🛠

None in this release.

### Exciting New Features 🎉

- **LiteLLM Gateway Deployment**: Comprehensive deployment of LiteLLM as a unified LLM gateway on Kubernetes with Redis caching, PostgreSQL persistence, and support for multiple model providers (OpenAI GPT-5 family, Claude models). This provides centralized model access control, usage tracking, and cost management for the entire MATRIX platform [#1845](https://github.com/everycure-org/matrix/pull/1845)

- **Multi-Model Pipeline Support**: Added capability to run multiple models in parallel within the pipeline, enabling comparative experiments and ensemble approaches. This significantly expands experimental flexibility for model evaluation [#1843](https://github.com/everycure-org/matrix/pull/1843)

- **PrimeKG Integration**: Integrated PrimeKG as a new knowledge source, expanding the diversity of biomedical knowledge available for drug repurposing analysis. Includes data ingestion, validation, and dashboard visualization support [#1793](https://github.com/everycure-org/matrix/pull/1793)

- **Pipeline Run Branching**: Introduced `--from-run` CLI option allowing pipeline runs to branch from existing runs by reusing their output datasets as inputs. This enables efficient experimentation without re-running expensive upstream computations [#1769](https://github.com/everycure-org/matrix/pull/1769)

- **GitHub Actions Self-Hosted Runners**: Deployed self-hosted GitHub Actions runners on Kubernetes using Actions Runner Controller (ARC), providing dedicated CI/CD compute resources with Docker-in-Docker support for improved pipeline execution times and cost optimization [#1812](https://github.com/everycure-org/matrix/pull/1812)

### Experiments 🧪

None in this release.

### Bugfixes 🐛

- Fixed drug list version to v0.1.4, resolving inconsistencies in drug entity references [#1878](https://github.com/everycure-org/matrix/pull/1878)
- Resolved Pandera deprecation warning by updating import paths to use the new API structure [#1856](https://github.com/everycure-org/matrix/pull/1856)
- Fixed Google Sheet dataset loading to properly handle worksheet gid parameter [#1858](https://github.com/everycure-org/matrix/pull/1858)
- Fixed pod rejection regex patterns in Argo workflow specifications to properly handle Kubernetes scheduling errors [#1863](https://github.com/everycure-org/matrix/pull/1863)
- Resolved CI workflow failures by fixing UV installation issues in GitHub Actions [#1831](https://github.com/everycure-org/matrix/pull/1831)

### Technical Enhancements 🧰

- **Redis Cache Configuration**: Added support for Redis caching in LiteLLM with configurable call types and cache control for improved model response times [#1881](https://github.com/everycure-org/matrix/pull/1881)
- **IAM Role Management**: Improved workbench configuration with Service Account User role assignments for better permission management [#1883](https://github.com/everycure-org/matrix/pull/1883)
- **Knowledge Graph Dashboard Improvements**:
  - Added KG statistics dashboard showing metrics across different releases [#1830](https://github.com/everycure-org/matrix/pull/1830)
  - Replaced knowledge source flow visualization with custom EChart network graphs for better performance [#1837](https://github.com/everycure-org/matrix/pull/1837)
  - Simplified normalized data tables for improved query performance [#1861](https://github.com/everycure-org/matrix/pull/1861)
  - Added graph components section with category and predicate pages [#1809](https://github.com/everycure-org/matrix/pull/1809)
  - Added benchmark release version tracking [#1827](https://github.com/everycure-org/matrix/pull/1827)
- **Baseline Model Update**: Updated baseline model to use integrated knowledge graph (RTX-KG2 + ROBOKOP) embeddings, replacing single-source embeddings [#1875](https://github.com/everycure-org/matrix/pull/1875)
- **MLflow Resilience**: Added retry logic for MLflow URL lookups to handle transient failures in workflow execution [#1866](https://github.com/everycure-org/matrix/pull/1866)
- **Fabricator Updates**: Enhanced data fabrication pipeline with improved validation and error handling [#1807](https://github.com/everycure-org/matrix/pull/1807)
- **Ingestion Validation**: Added comprehensive validation layer to data ingestion pipeline, ensuring data quality before processing [#1794](https://github.com/everycure-org/matrix/pull/1794)
- **Library Modularization**: Extracted shared validator logic into `matrix-inject` library for reuse across repositories, improving code maintainability [#1853](https://github.com/everycure-org/matrix/pull/1853)
- **Primary Knowledge Source Tracking**: Added column to edge data tracking all primary knowledge sources, enabling better provenance analysis [#1813](https://github.com/everycure-org/matrix/pull/1813)
- Removed obsolete KG validation step that was causing false positive failures [#1864](https://github.com/everycure-org/matrix/pull/1864)
- Added PrimeKG default color scheme in KG dashboard for consistent visualization [#1876](https://github.com/everycure-org/matrix/pull/1876)
- Updated benchmark (T3) version reference to v0.10.2 in KG dashboard [#1877](https://github.com/everycure-org/matrix/pull/1877)
- Enhanced GitHub Actions workflows with improved Python setup and dependency installation [#1844](https://github.com/everycure-org/matrix/pull/1844)
- Added `.env` to .dockerignore for better security hygiene [#1835](https://github.com/everycure-org/matrix/pull/1835)
- Removed obsolete `make install` from pipelines/matrix Makefile to reduce confusion [#1859](https://github.com/everycure-org/matrix/pull/1859)

### Documentation ✏️

- **LiteLLM Documentation Suite**: Comprehensive guides for LiteLLM administration, usage, and deployment including ADR for architectural decisions
- **Self-Hosted Runners Guide**: Complete documentation for GitHub Actions self-hosted runners deployment and configuration
- **Attribution Documentation**: Added comprehensive attribution page documenting all data sources, contributors, and funding acknowledgments [#1867](https://github.com/everycure-org/matrix/pull/1867)
- **Knowledge Sources Documentation**: Added detailed documentation page describing primary knowledge sources used in the KG [#1829](https://github.com/everycure-org/matrix/pull/1829)
- **Pipeline Branching Walkthrough**: Added detailed guide on using `--from-run` to branch from existing pipeline runs
- **ADR for LiteLLM on Kubernetes**: Architecture decision record documenting the rationale and implementation approach for LiteLLM deployment [#1834](https://github.com/everycure-org/matrix/pull/1834)
- Updated pipeline and infrastructure documentation to reflect new capabilities

### Other Changes

- Added LiteLLM model support for GPT-4-mini and Claude Haiku [#1874](https://github.com/everycure-org/matrix/pull/1874)

---

## Release Highlights

### Infrastructure & Platform Maturity

This release represents a significant step forward in MATRIX's infrastructure maturity with the deployment of LiteLLM as a centralized LLM gateway and self-hosted GitHub Actions runners. These additions provide better cost control, improved monitoring, and faster CI/CD execution.

### Knowledge Graph Expansion

The integration of PrimeKG as a third major knowledge source (alongside RTX-KG2 and ROBOKOP) substantially expands the biomedical knowledge available for drug repurposing predictions. The updated baseline model now leverages embeddings from the integrated multi-source knowledge graph.

<!-- TODO: Data Science team should add a summary of the integrated KG baseline model performance improvements compared to single-source embeddings -->

### Developer Experience Improvements

The new `--from-run` capability enables much more efficient experimentation by allowing researchers to branch from existing pipeline runs without recomputing expensive upstream steps. This dramatically reduces iteration time for model development and hyperparameter tuning.

### Dashboard & Visualization Enhancements

The KG dashboard received substantial improvements including cross-release statistics, better performance through simplified queries, and enhanced visualizations with custom EChart network graphs.
