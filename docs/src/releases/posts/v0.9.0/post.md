---
title: v0.9.0
draft: false
date: 2025-08-01
categories:
  - Release
authors:
  - JacquesVergine
  - Dashing-Nelson
  - piotrkan
  - may-lim
  - pascalwhoop
  - lvijnck
  - leelancashire
  - amyford
  - eKathleenCarter
  - Shilpasundar14
  - jdr0887
  - kevinschaper
---

### Breaking Changes 🛠
- Removed kg_raw and kept raw as the single folder path for all dev related datasets [#1723](https://github.com/everycure-org/matrix/pull/1723)
- Removed deprecated 'kedro submit' command in favor of 'kedro experiment run` [#1725](https://github.com/everycure-org/matrix/pull/1725)
- Update catalog names after GCP directory cleanup [#1698](https://github.com/everycure-org/matrix/pull/1698)
- Import pandera schema from matrix-schema package [#1641](https://github.com/everycure-org/matrix/pull/1641)
### Exciting New Features 🎉
- Add CLA and brand protection for open sourcing (AIP-339, AIP-340) [#1700](https://github.com/everycure-org/matrix/pull/1700)
- Enable Kedro Nodes Monitoring and Cost Allocation for GKE Pods [#1679](https://github.com/everycure-org/matrix/pull/1679)
- Knowledge Sources and EC Core Components dashboard update [#1628](https://github.com/everycure-org/matrix/pull/1628)
- More dynamic nn api [#1636](https://github.com/everycure-org/matrix/pull/1636)
- Add normalized category assignment using Node Normalizer (DATA-539) [#1633](https://github.com/everycure-org/matrix/pull/1633)
- Added GPU monitoring and kublet metrics [#1602](https://github.com/everycure-org/matrix/pull/1602)
- Dynamic GCS bucket selection for data sources [#1638](https://github.com/everycure-org/matrix/pull/1638)
- Expose ground truth train data (Take 2!) [#1639](https://github.com/everycure-org/matrix/pull/1639)
### Experiments 🧪
- Experiment with degree weighting to address the frequent flyer issue. [See notebook here](https://github.com/everycure-org/lab-notebooks/blob/3c8ddb9b271d7e1ddfb48aba9f4105a5e3271f8c/alan/frequent_flyers/shomer_degree_weighting/reports/weighting_report.ipynb)
- Experiment with degree bias to address the frequent flyer issue. [See notebook here](https://github.com/everycure-org/lab-notebooks/blob/3c8ddb9b271d7e1ddfb48aba9f4105a5e3271f8c/alan/frequent_flyers/shomer_degree_weighting/reports/gt_degree_bias_report.ipynb)
- Does filtering for biologically-relevant edges improve our predictions? An experiment using PRIME-KG. [See notebook here](https://github.com/everycure-org/lab-notebooks/blob/main/jane/filtering_experiments/primekg_filtering_experiments/0_experiment_report_primekg_filtering.ipynb)
- Identification of an optimal combinations of 3-5 large language models (LLMs) for biomedical publication validation - determining whether PubMed abstracts support given hypotheses/edges.	[See notebook here](https://github.com/everycure-org/lab-notebooks/blob/1a53ba3b0b4d2b52c6b258e523064249e5bb5cc0/uab-pubmed-embeddings/experiment-july-2025/UAB_PubMed_Embeddings_Tools_LLMs_selection.ipynb)
- Test the feasibility and utility of applying TracIn to interpret predictions from KGE models. [See report here](https://uploads.linear.app/b1cd352f-57bb-4750-b354-9fc9bde1225d/420eb38f-5a12-4273-b063-a45339abccbc/865a0e10-6848-4313-8b3e-e514b2d0a61e)
- Experiment with novel ensemble classifiers to increase prediction accuracy. [See notebook here](https://github.com/everycure-org/lab-notebooks/pull/180)
- Experiment with weighting synthesized negative samples to increase prediction accuracy. [See notebook here](https://github.com/everycure-org/lab-notebooks/blob/757ec2081bcc8481b944a04e4cc16129be1fd67f/chunyu_experiments/reports/T1_accuracy_improvement/T1_experiment_final_report_txgnn-8-3ce4ab2d.ipynb)
- Experiment with enhanced embeddings to increase prediction accuracy in zero shot settings. [See notebook here](https://github.com/everycure-org/lab-notebooks/blob/757ec2081bcc8481b944a04e4cc16129be1fd67f/chunyu_experiments/reports/T1_accuracy_improvement/T1_experiment_final_report_txgnn-8-3ce4ab2d.ipynb)
### Bugfixes 🐛
- Fix pipeline for disease split experiments [#1560](https://github.com/everycure-org/matrix/pull/1560)
- Update pandera version to 0.25.0 [#1670](https://github.com/everycure-org/matrix/pull/1670)
- Remove EC medical team dataset [#1686](https://github.com/everycure-org/matrix/pull/1686)
- Trim kedro nodes to 63 not 36 [#1694](https://github.com/everycure-org/matrix/pull/1694)
- Add max retry attemps to node normalizer call [#1711](https://github.com/everycure-org/matrix/pull/1711)
- Fix sentinel node's to read normalizer endpoint [#1719](https://github.com/everycure-org/matrix/pull/1719)
- Fix codebase following Pandera 0.24.0 breaking change [#1659](https://github.com/everycure-org/matrix/pull/1659)
- Revert pandera utils to previous pandera API [#1666](https://github.com/everycure-org/matrix/pull/1666)
- Moved PVC for services into one region [#1667](https://github.com/everycure-org/matrix/pull/1667)
- Revert pandera to safe version to avoid breaking changes [#1665](https://github.com/everycure-org/matrix/pull/1665)
- Update file paths for spoke nodes and edges [#1680](https://github.com/everycure-org/matrix/pull/1680)
- Hotfix: replace ingested disease list with integrated (and normalized) disease list [#1681](https://github.com/everycure-org/matrix/pull/1681)
- Random failure when pulling disease and drug lists from core [#1688](https://github.com/everycure-org/matrix/pull/1688)
- Added atomic and timeout to Helm release configuration in Terraform for stability [#1658](https://github.com/everycure-org/matrix/pull/1658)
- Removed kedro nodes label [#1660](https://github.com/everycure-org/matrix/pull/1660)
- Hotfix: Fix incorrect coalescing order for `normalize_edges` [#1663](https://github.com/everycure-org/matrix/pull/1663)
- Add path filter to docs deploy github action [#1721](https://github.com/everycure-org/matrix/pull/1721)
- Refactor CI tests to run sequentially for clarity and error handling [#1672](https://github.com/everycure-org/matrix/pull/1672)
- Update registry variable in scheduled sampling pipeline [#1710](https://github.com/everycure-org/matrix/pull/1710)
- Hotfix/hardcode public kg raw folder in catalog [#1676](https://github.com/everycure-org/matrix/pull/1676)
- Add QC and unit tests fixes post normalization bug fix [#1673](https://github.com/everycure-org/matrix/pull/1673)
### Technical Enhancements 🧰
- Add public GCS bucket configuration and update data paths for public datasets. [#1677](https://github.com/everycure-org/matrix/pull/1677)
- Refactor variables service account and [#1685](https://github.com/everycure-org/matrix/pull/1685)
- Delete obsolete workbenches [#1687](https://github.com/everycure-org/matrix/pull/1687)
- Update argo workflow to use `trimmed_kedro_nodes` in workflow template for labels [#1693](https://github.com/everycure-org/matrix/pull/1693)
- Add external subcon standard to bucket listing permissions for embiol… [#1702](https://github.com/everycure-org/matrix/pull/1702)
- Added getting variables from the github env [#1642](https://github.com/everycure-org/matrix/pull/1642)
- Added management pool for ArgoCD to put all workloads on it [#1643](https://github.com/everycure-org/matrix/pull/1643)
- Changed Vertex AI Timeout from 20 minutes to an hour [#1654](https://github.com/everycure-org/matrix/pull/1654)
- Removed dataminded from matrix repo [#1655](https://github.com/everycure-org/matrix/pull/1655)
- Allow orchard dev compute sa to read matrix dev bucket [#1662](https://github.com/everycure-org/matrix/pull/1662)
- Delete local cached files with make clean [#1722](https://github.com/everycure-org/matrix/pull/1722)
### Documentation ✏️
- Update common errors [#1674](https://github.com/everycure-org/matrix/pull/1674)
- Add ADR on OSS Storage setup [#1684](https://github.com/everycure-org/matrix/pull/1684)
- Fix broken neo4j link in references docs [#1690](https://github.com/everycure-org/matrix/pull/1690)
- Make docs more suitable for external contributors [#1577](https://github.com/everycure-org/matrix/pull/1577)
- Added document related to Main-Only Infrastructure Deployment Strategy [#1651](https://github.com/everycure-org/matrix/pull/1651)
- Refactor release documentation [#1657](https://github.com/everycure-org/matrix/pull/1657)
- Update onboarding issue link in 'Getting Started' [#1664](https://github.com/everycure-org/matrix/pull/1664)
- Refactor GCP documentation and remove deprecated Git-Crypt instructions [#1697](https://github.com/everycure-org/matrix/pull/1697)
### Other Changes
- Fixes for release/v0.8.2 in prod [#1668](https://github.com/everycure-org/matrix/pull/1668)
- Setup Orchard access for wg2 [#1704](https://github.com/everycure-org/matrix/pull/1704)
