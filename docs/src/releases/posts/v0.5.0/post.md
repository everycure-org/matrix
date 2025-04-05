---
draft: false
date: 2025-04-05
categories:
  - Release
authors:
  - Siyan-Luo
  - JacquesVergine
  - amyford
  - emil-k
  - eKathleenCarter
  - piotrkan
  - kevinschaper
  - james0032
  - app/github-actions
  - oliverw1
  - lvijnck
  - alexeistepa
  - pascalwhoop
  - chunyuma
---
Please write the article following the categories below. The relevant PR numbers, covering changes from the selected last release (or the most recent minor release in headless mode) to the current release, are provided for each category:

### Breaking Changes 🛠
### Exciting New Features 🎉
- Implement metric tracking using MLFlow  [#1051](https://github.com/everycure-org/matrix/pull/1051)
- Improve Performance of Filtering Pipeline [#1316](https://github.com/everycure-org/matrix/pull/1316)
- Add Hive Partitioning to Caches [#1336](https://github.com/everycure-org/matrix/pull/1336)
- Allow using (cold) cached results for any slow lookups. [#985](https://github.com/everycure-org/matrix/pull/985)
### Experiments 🧪
### Bugfixes 🐛
- Get IAP token for service account for authentication in GitHub Actions [#1282](https://github.com/everycure-org/matrix/pull/1282)
- Hotfix: Fuse the last node of the batch pipeline so that local output is reusable in the next node [#1329](https://github.com/everycure-org/matrix/pull/1329)
- Debug/pull after switching to release branch  [#1238](https://github.com/everycure-org/matrix/pull/1238)
- Allow retriggering the release PR workflow by catching up with previous commits [#1240](https://github.com/everycure-org/matrix/pull/1240)
- disable the mlflow hook that logs inputs, due to a transient bug. [#1241](https://github.com/everycure-org/matrix/pull/1241)
- Fix disease specific ranking runtime bug [#1278](https://github.com/everycure-org/matrix/pull/1278)
- Fix stability metrics & MLFlow modelling catalog [#1147](https://github.com/everycure-org/matrix/pull/1147)
- Remove embeddings pipeline from kg release patch [#1283](https://github.com/everycure-org/matrix/pull/1283)
- Fix automated KG dashboard deployment and release sample creation [#1288](https://github.com/everycure-org/matrix/pull/1288)
- Fix stability metrics & MLFlow modelling catalog [#1147](https://github.com/everycure-org/matrix/pull/1147)
### Technical Enhancements 🧰
- Create a filtering pipeline. Move edge deduplication out of release. [#1169](https://github.com/everycure-org/matrix/pull/1169)
- Reduce drug and disease name resolution time by using the batch endpoint [#1158](https://github.com/everycure-org/matrix/pull/1158)
- Add upstream data source information to KG dashboard summary page [#1289](https://github.com/everycure-org/matrix/pull/1289)
- Add last updated time information to KG dashboard [#1305](https://github.com/everycure-org/matrix/pull/1305)
- Improvement/parametrize spark memory setting [#1185](https://github.com/everycure-org/matrix/pull/1185)
- Improve Performance of Filtering Pipeline [#1316](https://github.com/everycure-org/matrix/pull/1316)
- Add Hive Partitioning to Caches [#1336](https://github.com/everycure-org/matrix/pull/1336)
- Disable Neo4J ingestion for weekly patch [#1219](https://github.com/everycure-org/matrix/pull/1219)
- Fix: use a single MLFlow run for every node [#1222](https://github.com/everycure-org/matrix/pull/1222)
- Automate KG dashboard deployment with each KG release [#1228](https://github.com/everycure-org/matrix/pull/1228)
- Automate the creation of the sample for each release  [#1249](https://github.com/everycure-org/matrix/pull/1249)
- Update release version in daily sample run [#1258](https://github.com/everycure-org/matrix/pull/1258)
- Feat/cache ncats node normalizer [#1269](https://github.com/everycure-org/matrix/pull/1269)
- Improve Performance of Filtering Pipeline [#1316](https://github.com/everycure-org/matrix/pull/1316)
### Documentation ✏️
- Add report benchmarking and comparing different ground truth sets [#1304](https://github.com/everycure-org/matrix/pull/1304)
-  Docs/update release-related documentations [#1333](https://github.com/everycure-org/matrix/pull/1333)
- Add initial docs for updating a data source [#1342](https://github.com/everycure-org/matrix/pull/1342)
- Docs/how the automated release workflow works on a high level [#1229](https://github.com/everycure-org/matrix/pull/1229)
- Docs/how to re-trigger pipeline submission and PR creation GitHub Actions [#1227](https://github.com/everycure-org/matrix/pull/1227)
- Add TxGNN experiment summary report #1218 [#1243](https://github.com/everycure-org/matrix/pull/1243)
- Add experimental report comparing RTX-KG2, ROBOKOP and Integrated KG matrix outputs [#1246](https://github.com/everycure-org/matrix/pull/1246)
### Other Changes
- Fix test sample pipeline by adding filtering step [#1300](https://github.com/everycure-org/matrix/pull/1300)
- Fix bug where MLFLOW_RUN_ID=None [#1281](https://github.com/everycure-org/matrix/pull/1281)
- Add kg_release_patch to trigger release flag in Argo label [#1312](https://github.com/everycure-org/matrix/pull/1312)
- Add Normalization to evidence.dev Dashboard [#1183](https://github.com/everycure-org/matrix/pull/1183)
- update documentation link for workbench creation [#1308](https://github.com/everycure-org/matrix/pull/1308)
- Introduce ADR for inclusion of private datasets  [#1189](https://github.com/everycure-org/matrix/pull/1189)
- Fix scheduled sampling GH action with credentials [#1323](https://github.com/everycure-org/matrix/pull/1323)
- Add backup restore runbook [#1337](https://github.com/everycure-org/matrix/pull/1337)
- Make Architecture Decision Record bold in documentation navigation bar [#1340](https://github.com/everycure-org/matrix/pull/1340)
- Allow auto-release version bumping to the available one [#1341](https://github.com/everycure-org/matrix/pull/1341)
- Source release info of node normalizer from Kedro params [#1345](https://github.com/everycure-org/matrix/pull/1345)
- Fix pair generator for recall@n [#1220](https://github.com/everycure-org/matrix/pull/1220)
- Switch CODEOWNERS file back to individuals & manual release of 0.3.4/0.3.5/0.3.6 [#1230](https://github.com/everycure-org/matrix/pull/1230)
- Release/v0.4.1 [#1237](https://github.com/everycure-org/matrix/pull/1237)
- minor ADR doc upgrade [#1359](https://github.com/everycure-org/matrix/pull/1359)
- Fix broken category & prefix links on dashboard [#1251](https://github.com/everycure-org/matrix/pull/1251)
- Have "kedro experiment" work headlessly as well [#1255](https://github.com/everycure-org/matrix/pull/1255)
- Adding DM folks to specific pieces [#1270](https://github.com/everycure-org/matrix/pull/1270)
- Remove LLM-generated article and provide a template [#1303](https://github.com/everycure-org/matrix/pull/1303)
- Add Hive Partitioning to Caches [#1336](https://github.com/everycure-org/matrix/pull/1336)
- Optimize evaluate function in specific ranking for disease pairs [#1338](https://github.com/everycure-org/matrix/pull/1338)
- Ensure the GitHub workflow for creating a release pr can be re-run [#1208](https://github.com/everycure-org/matrix/pull/1208)
-  Implement Version Cleanup to Avoid GAE Deployment Limits [#1256](https://github.com/everycure-org/matrix/pull/1256)
- Improve the performance of the Spark tasks in the modelling pipeline [#1263](https://github.com/everycure-org/matrix/pull/1263)
- Implement metric tracking using MLFlow  [#1051](https://github.com/everycure-org/matrix/pull/1051)
- Edit TxGNN summary report [#1253](https://github.com/everycure-org/matrix/pull/1253)