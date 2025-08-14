---
title: v0.9.3
draft: false
date: 2025-08-14
categories:
  - Release
authors:
  - Dashing-Nelson
  - pascalwhoop
  - JacquesVergine
  - jdr0887
  - piotrkan
  - amyford
---

### Breaking Changes 🛠

### Exciting New Features 🎉

- Add validation capabilities to the fabricator pipeline with enhanced data quality checks [#1714](https://github.com/everycure-org/matrix/pull/1714)
- Add features and modeling components to weekly pipeline runs for automated processing [#1631](https://github.com/everycure-org/matrix/pull/1631)

### Experiments 🧪

### Bugfixes 🐛

- Enhanced node category filtering in the pipeline to properly handle different node types and improve data quality [#1730](https://github.com/everycure-org/matrix/pull/1730)

### Technical Enhancements 🧰

- Refactor disease list ingestion to use pandas.CSVDataset for better performance and maintainability [#1750](https://github.com/everycure-org/matrix/pull/1750)
- Update PySpark version to 3.5.6 for improved stability and performance [#1753](https://github.com/everycure-org/matrix/pull/1753)
- Move clinical trials and off-label data to public access for better data availability [#1760](https://github.com/everycure-org/matrix/pull/1760)
- Enhanced kube-state-metrics configuration with podcontainers in metric labels allowlist [#1733](https://github.com/everycure-org/matrix/pull/1733)
- Resolve memory issues in stability pipeline by addressing OOM errors [#1732](https://github.com/everycure-org/matrix/pull/1732)
- Disable GKE backup agent configuration in compute cluster module for cost optimization [#1731](https://github.com/everycure-org/matrix/pull/1731)
- Add comprehensive Artifact Registry module with automated cleanup policies and documentation [#1717](https://github.com/everycure-org/matrix/pull/1717)
- Add platform refactor and standardization documentation for better development practices [#1706](https://github.com/everycure-org/matrix/pull/1706)

### Documentation ✏️

- Update installation instructions for Linux with comprehensive pyenv setup steps [#1748](https://github.com/everycure-org/matrix/pull/1748)

### Other Changes

- Add ARGO configuration to core stability pipeline for improved workflow management [#1747](https://github.com/everycure-org/matrix/pull/1747)
- Update Argo workflows configuration for better pipeline orchestration [#1759](https://github.com/everycure-org/matrix/pull/1759)
- Fix secrets scanning configuration to enhance security [#1746](https://github.com/everycure-org/matrix/pull/1746)
- Update project LICENSE file [#1742](https://github.com/everycure-org/matrix/pull/1742)
- Update CODE_OF_CONDUCT.md with current standards [#1741](https://github.com/everycure-org/matrix/pull/1741)
- Update README.md with current project information [#1735](https://github.com/everycure-org/matrix/pull/1735)
- Remove infrastructure deployment badge from README [#1734](https://github.com/everycure-org/matrix/pull/1734)