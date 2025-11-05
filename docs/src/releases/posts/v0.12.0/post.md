---
title: v0.12.0
draft: false
date: 2025-11-04
categories:
  - Release
authors:
  - Dashing-Nelson
  - JacquesVergine
  - piotrkan
  - pascalwhoop
  - alexeistepa
  - lvijnck
  - eKathleenCarter
  - matentzn
---

### Breaking Changes 🛠

No breaking changes in this release.

### Exciting New Features 🎉

- **Automated primary knowledge source documentation pipeline**: Introduced a new documentation pipeline that automatically generates content for primary knowledge sources, streamlining the documentation process and ensuring consistency across knowledge graph sources [#1846](https://github.com/everycure-org/matrix/pull/1846)

- **ABox/TBox node classification**: Added support for distinguishing between ABox (assertional) and TBox (terminological) nodes in the knowledge graph, enabling better ontological reasoning and knowledge representation [#1895](https://github.com/everycure-org/matrix/pull/1895)

### Experiments 🧪

> Lee placeholder

### Bugfixes 🐛

- **MLflow image pull issue resolution**: Fixed critical MLflow deployment issues caused by Bitnami registry changes, ensuring reliable experiment tracking and model management [#1891](https://github.com/everycure-org/matrix/pull/1891)

- **Release patch pipeline fix**: Added missing `document_kg` to the release patch pipeline, ensuring all necessary components are included in patch releases [#1913](https://github.com/everycure-org/matrix/pull/1913)

- **PKS markdown generation variable fix**: Corrected variable usage in primary knowledge source markdown generation, preventing template rendering errors [#1909](https://github.com/everycure-org/matrix/pull/1909)

- **Infrastructure typo fix**: Fixed minor typo in infrastructure file comments for improved code clarity [#1902](https://github.com/everycure-org/matrix/pull/1902)

### Technical Enhancements 🧰

- **New cross-validation strategy**: Implemented an improved cross-validation approach for model training, enhancing model evaluation robustness and reliability. <!-- TODO: Add details about the specific CV strategy and performance improvements --> [#1847](https://github.com/everycure-org/matrix/pull/1847)

- **Drug list ingestion refactor**: Refactored the matrix pipeline to support the new drug list ingestion format, improving data processing efficiency and maintainability [#1885](https://github.com/everycure-org/matrix/pull/1885)

- **Memory-efficient predictions**: Created a memory-efficient restrict predictions node and migrated to partitioned datasets, significantly reducing memory footprint for large-scale inference tasks [#1898](https://github.com/everycure-org/matrix/pull/1898)

- **BigQuery location support**: Added location parameter to SparkDatasetWithBQExternalTable for better multi-region support and data locality [#1897](https://github.com/everycure-org/matrix/pull/1897)

- **Epistemic robustness documentation**: Enhanced knowledge source pages with epistemic robustness information, providing transparency about data quality and reliability [#1896](https://github.com/everycure-org/matrix/pull/1896)

- **Spot instance improvements**: Disabled spot instances for non-dev environments and added conditional spot node pool configuration for improved production stability [#1907](https://github.com/everycure-org/matrix/pull/1907)

- **Spot instance removal**: Completely removed spot instances from both dev and prod environments to ensure consistent infrastructure performance [#1910](https://github.com/everycure-org/matrix/pull/1910)

- **Orchard compute IAM configuration**: Added orchard compute service accounts to IAM configuration for enhanced access management [#1912](https://github.com/everycure-org/matrix/pull/1912)

- **Py4J gateway timeout**: Added configurable Py4J gateway startup timeout to Spark configuration, preventing connection failures in resource-constrained environments [#1903](https://github.com/everycure-org/matrix/pull/1903)

- **Workbench IAM improvements**: Added IAM member resource for Service Account User role in workbench configuration, streamlining user access management [#1883](https://github.com/everycure-org/matrix/pull/1883)

- **LiteLLM Redis cache support**: Added supported call types for Redis cache configuration in litellm, improving caching capabilities for LLM operations [#1881](https://github.com/everycure-org/matrix/pull/1881)

### Documentation ✏️

- **Attribution documentation**: Added comprehensive attribution documentation for the Matrix project, properly crediting data sources and collaborators [#1867](https://github.com/everycure-org/matrix/pull/1867)

### Other Changes

- **Argo Events dependency update**: Updated argo-events dependency to version 2.4.16 and synchronized subproject commit for latest features and fixes [#1915](https://github.com/everycure-org/matrix/pull/1915)

- **Neo4j query logging**: Enabled Neo4j query logging by default for improved debugging and performance monitoring [#1906](https://github.com/everycure-org/matrix/pull/1906)

- **BigQuery permissions**: Added read permissions for the evidence project to access BigQuery datasets [#1901](https://github.com/everycure-org/matrix/pull/1901)
