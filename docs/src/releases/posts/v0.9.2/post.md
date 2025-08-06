---
title: v0.9.2
draft: false
date: 2025-08-06
categories:
  - Release
authors:
  - amyford
  - github-actions[bot]
  - Jacques Vergine
  - Jason Reilly
  - Kathleen Carter
  - Kevin Schaper
  - Laurens
  - leelancashire
  - may-lim
  - Nelson Alfonso
  - Pascal Bro
  - Pascal Brokmeier
  - Piotr Kaniewski
  - Shilpa Sundar
---

### Breaking Changes 🛠

- Removed deprecated matrix-cli application entirely, simplifying release processes and reducing maintenance overhead by transitioning to manual workflows
  [#1655](https://github.com/everycure-org/matrix/pull/1655)

- Replaced deprecated `kedro submit` command with `kedro experiment run` for better experiment management and consistency
  [#1725](https://github.com/everycure-org/matrix/pull/1725)

- Restructured data catalog paths by removing `kg_raw` folder and consolidating to single `raw` folder structure for simplified data management
  [#1723](https://github.com/everycure-org/matrix/pull/1723)

### Exciting New Features 🎉

- Enhanced EC Core entities normalization to prevent ID drift during the normalization process, ensuring consistent entity identifiers across pipeline runs
  [#1705](https://github.com/everycure-org/matrix/pull/1705)

- Implemented dynamic Node Normalizer API with improved error handling and retry mechanisms for more reliable entity normalization
  [#1636](https://github.com/everycure-org/matrix/pull/1636)

- Added comprehensive Knowledge Sources and EC Core Components dashboard providing detailed insights into data provenance and entity coverage
  [#1628](https://github.com/everycure-org/matrix/pull/1628)

- Introduced dynamic GCS bucket selection capability allowing flexible data source management across different environments
  [#1638](https://github.com/everycure-org/matrix/pull/1638)

- Exposed ground truth training data through improved data catalog access patterns, enabling better model validation workflows
  [#1639](https://github.com/everycure-org/matrix/pull/1639)

### Experiments 🧪

- Conducted comprehensive disease split analysis to improve model generalization and evaluate performance across different disease categories
  [#1560](https://github.com/everycure-org/matrix/pull/1560)

### Bugfixes 🐛

- Fixed OOM (Out of Memory) errors in pipeline stability by optimizing memory usage patterns and resource allocation
  [#1732](https://github.com/everycure-org/matrix/pull/1732)

- Resolved random failures when retrieving disease and drug lists from core services by implementing proper error handling and retry logic
  [#1688](https://github.com/everycure-org/matrix/pull/1688)

- Corrected sentinel node configuration to properly read from normalizer endpoint, fixing entity resolution issues
  [#1719](https://github.com/everycure-org/matrix/pull/1719)

- Fixed BigQuery filtering logic to operate per run rather than per release, ensuring accurate data processing
  [#1727](https://github.com/everycure-org/matrix/pull/1727)

- Resolved registry variable issues in scheduled sampling pipeline configuration for consistent deployment behavior
  [#1710](https://github.com/everycure-org/matrix/pull/1710)

- Updated Argo workflow templates to use correct `trimmed_kedro_nodes` labels, fixing workflow execution issues
  [#1693](https://github.com/everycure-org/matrix/pull/1693)

- Applied hotfix to replace ingested disease list with integrated and normalized disease list for data consistency
  [#1681](https://github.com/everycure-org/matrix/pull/1681)

- Fixed broken Neo4j documentation links in references section
  [#1690](https://github.com/everycure-org/matrix/pull/1690)

### Technical Enhancements 🧰

- Enhanced Kubernetes monitoring by configuring kube-state-metrics to include pod container metrics in allowlisted labels
  [#1733](https://github.com/everycure-org/matrix/pull/1733)

- Disabled GKE backup agent configuration in compute cluster module to reduce resource overhead and simplify cluster management
  [#1731](https://github.com/everycure-org/matrix/pull/1731)

- Implemented comprehensive GPU monitoring and kubelet metrics collection for better cluster observability and resource management
  [#1602](https://github.com/everycure-org/matrix/pull/1602)

- Added ArgoCD management pool configuration to dedicated workload management and improved resource isolation
  [#1643](https://github.com/everycure-org/matrix/pull/1643)

- Enhanced Helm release stability by adding atomic deployment and timeout configurations in Terraform
  [#1658](https://github.com/everycure-org/matrix/pull/1658)

- Improved Pandera integration by importing schemas from matrix-schema package for better data validation consistency
  [#1641](https://github.com/everycure-org/matrix/pull/1641)

- Added public GCS bucket configuration with updated data paths for public dataset accessibility
  [#1677](https://github.com/everycure-org/matrix/pull/1677)

- Implemented Kedro nodes monitoring and cost allocation for GKE pods enabling better resource tracking
  [#1679](https://github.com/everycure-org/matrix/pull/1679)

- Added normalized category assignment using Node Normalizer service for improved data quality
  [#1633](https://github.com/everycure-org/matrix/pull/1633)

- Configured max retry attempts for node normalizer calls to improve service reliability
  [#1711](https://github.com/everycure-org/matrix/pull/1711)

- Updated spoke nodes and edges file paths to include version information in filenames for better data lineage tracking
  [#1680](https://github.com/everycure-org/matrix/pull/1680)

- Added external subcontractor permissions for bucket listing access, supporting collaboration with Embiology team
  [#1702](https://github.com/everycure-org/matrix/pull/1702)

- Implemented cross-account infrastructure setup for improved security and resource isolation
  [#1704](https://github.com/everycure-org/matrix/pull/1704)

- Added `make clean` command for deleting local cached files, improving development workflow hygiene
  [#1722](https://github.com/everycure-org/matrix/pull/1722)

- Refactored variables service account configuration for better infrastructure management
  [#1685](https://github.com/everycure-org/matrix/pull/1685)

- Added path filters to docs deployment action for more efficient CI/CD pipeline execution
  [#1721](https://github.com/everycure-org/matrix/pull/1721)

- Removed standard node pools from GKE configuration to streamline node management and reduce complexity
  [#1731](https://github.com/everycure-org/matrix/pull/1731)

- Updated Vertex AI timeout configuration from 20 minutes to one hour for longer-running model operations
  [#1654](https://github.com/everycure-org/matrix/pull/1654)

- Updated Pandera version to 0.25.0 and fixed breaking changes for improved data validation capabilities
  [#1670](https://github.com/everycure-org/matrix/pull/1670)

### Documentation ✏️

- Added comprehensive platform refactor and standardization documentation for improved development guidelines
  [#1706](https://github.com/everycure-org/matrix/pull/1706)

- Enhanced CLAUDE.md with detailed development commands, architecture overview, and AI assistance guidelines
  [Multiple commits](https://github.com/everycure-org/matrix/commits/main)

- Added CONTRIBUTING.md with contributor guidelines and development standards to support open-source readiness
  [#1700](https://github.com/everycure-org/matrix/pull/1700)

- Refactored GCP documentation and removed deprecated Git-Crypt instructions for clearer infrastructure guidance
  [#1697](https://github.com/everycure-org/matrix/pull/1697)

- Improved release documentation with better process guidance and automation workflows
  [#1657](https://github.com/everycure-org/matrix/pull/1657)

- Added Main-Only Infrastructure Deployment Strategy documentation for standardized deployment practices
  [#1651](https://github.com/everycure-org/matrix/pull/1651)

- Updated getting started documentation to be more suitable for external contributors
  [#1577](https://github.com/everycure-org/matrix/pull/1577)

- Added ADR on OSS Storage setup for open-source deployment guidance
  [#1684](https://github.com/everycure-org/matrix/pull/1684)

- Removed onboarding template and simplified contributor onboarding process
  [#1656](https://github.com/everycure-org/matrix/pull/1656)

- Updated common errors documentation with additional troubleshooting information
  [#1674](https://github.com/everycure-org/matrix/pull/1674)

### Other Changes

- Added CLA and brand protection measures in preparation for open-source release
  [#1700](https://github.com/everycure-org/matrix/pull/1700)

- Implemented Claude-based automated changelog generation system for streamlined release processes
  [Multiple commits](https://github.com/everycure-org/matrix/commits/main)

- Updated data catalog configurations following directory restructuring changes
  [#1698](https://github.com/everycure-org/matrix/pull/1698)

- Removed obsolete workbenches and EC medical team dataset for codebase cleanup
  [#1687](https://github.com/everycure-org/matrix/pull/1687) [#1686](https://github.com/everycure-org/matrix/pull/1686)

- Disabled CDN in data_release_zone module for development environment to reduce operational costs
  [Multiple commits](https://github.com/everycure-org/matrix/commits/main)

- Added GitModule configuration updates for improved submodule management
  [#1716](https://github.com/everycure-org/matrix/pull/1716)

- Enhanced workflow configurations across multiple GitHub Actions for better CI/CD reliability
  [Multiple commits](https://github.com/everycure-org/matrix/commits/main)