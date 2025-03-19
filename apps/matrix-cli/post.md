---
draft: false
date: 2025-03-19
categories:
  - Release
authors:
  - alexeistepa
  - JacquesVergine
  - piotrkan
  - lvijnck
  - Siyan-Luo
  - pascalwhoop
  - eKathleenCarter
  - emil-k
  - amyford
  - app/github-actions
  - oliverw1
  - matentzn
---
Please write the article following the categories below. The relevant PR numbers, covering changes from the selected last release (or the most recent minor release in headless mode) to the current release, are provided for each category:

### Breaking Changes 🛠
### Exciting New Features 🎉
- Quality control data for Evidence.dev [#1076](https://github.com/everycure-org/matrix/pull/1076)
- [Infra sync] Evidence.dev infrastructure  [#1112](https://github.com/everycure-org/matrix/pull/1112)
- Feat/run sampling pipeline on schedule [#1105](https://github.com/everycure-org/matrix/pull/1105)
- Add --nodes to Kedro submit [#1142](https://github.com/everycure-org/matrix/pull/1142)
- Add a summary page to Evidence with ARPA metrics [#1194](https://github.com/everycure-org/matrix/pull/1194)
### Experiments 🧪
### Bugfixes 🐛
- Debug/Notes and articles generation [#1059](https://github.com/everycure-org/matrix/pull/1059)
- Fix deadlocking on subprocess calls [#1089](https://github.com/everycure-org/matrix/pull/1089)
- Correct the scope of information used to generate the release notes. [#1096](https://github.com/everycure-org/matrix/pull/1096)
- hotfix: add missing Makefile target [#1103](https://github.com/everycure-org/matrix/pull/1103)
- Update BigQuery table if it exists instead of creating it [#1110](https://github.com/everycure-org/matrix/pull/1110)
- Fix integration pipeline error with missing interpolation key [#1123](https://github.com/everycure-org/matrix/pull/1123)
- Modify the location of AI-generated notes file [#1129](https://github.com/everycure-org/matrix/pull/1129)
- Add upstream data source to Neo4j edges [#1131](https://github.com/everycure-org/matrix/pull/1131)
- Bug/add gh token for kedro submit step in GH Actions [#1132](https://github.com/everycure-org/matrix/pull/1132)
- Delete release docs of v0.2.3 folder [#1127](https://github.com/everycure-org/matrix/pull/1127)
- Correct BQ reporting table names and change tests to cover cloud catalog [#1133](https://github.com/everycure-org/matrix/pull/1133)
- Fix writing to the gsheets for SILC sheet [#1193](https://github.com/everycure-org/matrix/pull/1193)
- Correct the scope of information used to generate the release notes. [#1096](https://github.com/everycure-org/matrix/pull/1096)
- hotfix: add missing Makefile target [#1103](https://github.com/everycure-org/matrix/pull/1103)
- Render the github.event content in CI without interpolation, avoiding prematurely closing it [#1107](https://github.com/everycure-org/matrix/pull/1107)
- Modify the location of AI-generated notes file [#1129](https://github.com/everycure-org/matrix/pull/1129)
- Bug/add gh token for kedro submit step in GH Actions [#1132](https://github.com/everycure-org/matrix/pull/1132)
- Delete release docs of v0.2.3 folder [#1127](https://github.com/everycure-org/matrix/pull/1127)
- Debug/use git command instead of gh command [#1178](https://github.com/everycure-org/matrix/pull/1178)
- Debug/allow bump type input from UI [#1223](https://github.com/everycure-org/matrix/pull/1223)
### Technical Enhancements 🧰
- Add GitHub release dataset for drug and disease list ingestion [#1050](https://github.com/everycure-org/matrix/pull/1050)
- Refactor preprocessing pipeline  [#1088](https://github.com/everycure-org/matrix/pull/1088)
- Add ability to specify mlflow experiment by name [#1093](https://github.com/everycure-org/matrix/pull/1093)
- Create Slack notification when pipeline submission fails on GHAction [#1141](https://github.com/everycure-org/matrix/pull/1141)
- Fix writing to the gsheets for SILC sheet [#1193](https://github.com/everycure-org/matrix/pull/1193)
- Allow sample run to be manually triggered [#1206](https://github.com/everycure-org/matrix/pull/1206)
- include drug and disease in release info [#1221](https://github.com/everycure-org/matrix/pull/1221)
- Remove hardcoded SILC config [#973](https://github.com/everycure-org/matrix/pull/973)
- Setup IAP OAuth for use with MLFlow [#897](https://github.com/everycure-org/matrix/pull/897)
### Documentation ✏️
- Remove GOOGLE_CREDENTIALS env variable from installation documentation [#1108](https://github.com/everycure-org/matrix/pull/1108)
- Docs cleanup [#1150](https://github.com/everycure-org/matrix/pull/1150)
- Improve sampling documentation with release specific instructions [#1166](https://github.com/everycure-org/matrix/pull/1166)
- Add documentation for disease tagging / categorisation feature [#955](https://github.com/everycure-org/matrix/pull/955)
- Define process to fix a KG release [#1207](https://github.com/everycure-org/matrix/pull/1207)
- Add documentation for explaining more tags over releases [#1209](https://github.com/everycure-org/matrix/pull/1209)
### Other Changes
- Fix clinical trial preprocessing nodes [#1039](https://github.com/everycure-org/matrix/pull/1039)
- Fix ec medical nodes in preprocessing [#1052](https://github.com/everycure-org/matrix/pull/1052)
- Fix schema check in preprocessing pipeline [#1082](https://github.com/everycure-org/matrix/pull/1082)
- Setup drugmech ingestion [#1041](https://github.com/everycure-org/matrix/pull/1041)
- Better cli for quickly adding users to multiple teams [#1040](https://github.com/everycure-org/matrix/pull/1040)
- Fix normalizer always returning `normalization_success=True` [#1060](https://github.com/everycure-org/matrix/pull/1060)
- Bump disease list and fix release list name  [#1072](https://github.com/everycure-org/matrix/pull/1072)
- Feat/log datasets used to mlflow [#1048](https://github.com/everycure-org/matrix/pull/1048)
- Fix mlflow metric tracking [#1075](https://github.com/everycure-org/matrix/pull/1075)
- Update onboarding docs to include container registry auth [#1081](https://github.com/everycure-org/matrix/pull/1081)
- Evidence.dev code & deployment & CI [#1085](https://github.com/everycure-org/matrix/pull/1085)
- Feat/add custom argo prometheus metric on failed workflow status [#1098](https://github.com/everycure-org/matrix/pull/1098)
- pinned torch and re-generate requirements on mac [#1109](https://github.com/everycure-org/matrix/pull/1109)
- Hotfix - change version for GT in ingestion catalog [#1116](https://github.com/everycure-org/matrix/pull/1116)
- Disable stability metrics (temporarily) [#1126](https://github.com/everycure-org/matrix/pull/1126)
- Move de-duplication to integration from preprocessing [#1118](https://github.com/everycure-org/matrix/pull/1118)
- increasing the timeout to handle api overloading [#1146](https://github.com/everycure-org/matrix/pull/1146)
- [mini] status badges in readme [#1145](https://github.com/everycure-org/matrix/pull/1145)
- Format kedro experiment docs [#1159](https://github.com/everycure-org/matrix/pull/1159)
- Improved sankey on evdev dashboard [#1153](https://github.com/everycure-org/matrix/pull/1153)
- Fix modelling bug - modelling cloud catalog  [#1165](https://github.com/everycure-org/matrix/pull/1165)
- Update test configuration for topological embeddings to reduce integration test duration [#1161](https://github.com/everycure-org/matrix/pull/1161)
- add min max to reported aggregations [#1152](https://github.com/everycure-org/matrix/pull/1152)
- Use OAuth secret from git crypt. Add docs [#1168](https://github.com/everycure-org/matrix/pull/1168)
- only log mlflow dataset if it hasn't been logged before. [#1180](https://github.com/everycure-org/matrix/pull/1180)
- Hotfix - fix make fetch_secrets missing variable and twice defined [#1172](https://github.com/everycure-org/matrix/pull/1172)
- Resource allocation changes for embeddings pipeline [#1179](https://github.com/everycure-org/matrix/pull/1179)
- Fix/embeddins resources [#1170](https://github.com/everycure-org/matrix/pull/1170)
- Fix catalog in ingestion [#1176](https://github.com/everycure-org/matrix/pull/1176)
- Revert window size to 10 for Node2Vec Embeddings [#1184](https://github.com/everycure-org/matrix/pull/1184)
- add rank columns [#1186](https://github.com/everycure-org/matrix/pull/1186)
- Test deploy evidence.dev 0.3.3 [#1190](https://github.com/everycure-org/matrix/pull/1190)
- Feat/archive mlflow runs [#1181](https://github.com/everycure-org/matrix/pull/1181)
- Debug/expand mlflow hook logging [#1204](https://github.com/everycure-org/matrix/pull/1204)
- Quality control data for Evidence.dev [#1076](https://github.com/everycure-org/matrix/pull/1076)
- Reduce resource requirements for edge and node ingestion into Neo4j. [#1195](https://github.com/everycure-org/matrix/pull/1195)
- include drug and disease in release info [#1221](https://github.com/everycure-org/matrix/pull/1221)