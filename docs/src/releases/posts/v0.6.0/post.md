---
title: v0.6.0
draft: false
date: 2025-05-02
categories:
  - Release
authors:
  - pascalwhoop
  - emil-k
  - matwasilewski
  - Siyan-Luo
  - alexeistepa
  - leelancashire
  - lvijnck
  - drhodesbrc
  - oliverw1
  - kevinschaper
  - JacquesVergine
  - eKathleenCarter
  - eding36
  - amyford
  - redst4r
  - piotrkan
  - Dashing-Nelson
---

### Breaking Changes 🛠
- Remove duplicate tables (nodes & edges) in BigQuery [#1424](https://github.com/everycure-org/matrix/pull/1424)

### Exciting New Features 🎉
- Feature/create k8s backups [#1306](https://github.com/everycure-org/matrix/pull/1306)
- Enable Neo4J endpoint for all releases [#803](https://github.com/everycure-org/matrix/pull/803)
- Create a new git-crypt key to store infra-related secrets for production [#1355](https://github.com/everycure-org/matrix/pull/1355)
- Create run dashboard [#1377](https://github.com/everycure-org/matrix/pull/1377)
- Improve Integration and Filtering Pipelines with Normalization Summaries and Enhanced Edge Tracking [#1379](https://github.com/everycure-org/matrix/pull/1379)
- DS Workbenches on Vertex AI for ML researchers [#1102](https://github.com/everycure-org/matrix/pull/1102)
- Ingest & integrated Embiology KG in production environment [#1406](https://github.com/everycure-org/matrix/pull/1406)

### Experiments 🧪
- Diseases split [#1410](https://github.com/everycure-org/matrix/pull/1410)
- Drug split [#1420](https://github.com/everycure-org/matrix/pull/1420)
- Matrix transformation reports [report](https://github.com/everycure-org/lab-notebooks/blob/main/alexei/8_matrix_normalisation_refined_analysis/report/matrix_transformation_refined.md)
- Run existing modelling pipeline on RTX 2.10.0 (bump from 2.7.3) [report](https://github.com/everycure-org/lab-notebooks/blob/main/amy/rtx-2.10.0-report.ipynb)
- UAB PubMed Embeddings Drug Repurposing Experiment [report](https://github.com/everycure-org/lab-notebooks/blob/1029a7b70fb921690ed5842adbeb8fe2313ffd9a/uab-pubmed-embeddings/experiment-april-2025/UAB_PubMed_Embeddings_Drug_Repurposing_Experiment_Report.ipynb)
- Exploring a matrix transformation for contraindications [report](https://github.com/everycure-org/lab-notebooks/blob/main/alexei/a_contraindications_research/report/transformation_approach.md)
- Add diseases split and matrix transformation reports [#1410](https://github.com/everycure-org/matrix/pull/1410)
- Adding Agent Type Score and Combined Evidence Score to KG Dashboard [#1405](https://github.com/everycure-org/matrix/pull/1405)
- Adding Normalization Reports to KG Dashboard [#1409](https://github.com/everycure-org/matrix/pull/1409)

### Technical Enhancements 🧰
- Use DNS module's variables as outputs, not data, to create a dependency. [#1254](https://github.com/everycure-org/matrix/pull/1254)
- Extend engineering permissions [#749](https://github.com/everycure-org/matrix/pull/749)
- [AIP-169]: deleting workflows and templates older than 30d  [#1265](https://github.com/everycure-org/matrix/pull/1265)
- Add Kubernetes Cluster Restore Plan [#1343](https://github.com/everycure-org/matrix/pull/1343)
- Improve Integration and Filtering Pipelines with Normalization Summaries and Enhanced Edge Tracking [#1379](https://github.com/everycure-org/matrix/pull/1379)
- Roles modification to test Gemini call [#774](https://github.com/everycure-org/matrix/pull/774)
- Revert the changes on permission [#779](https://github.com/everycure-org/matrix/pull/779)
- Add Grafana and Prometheus [#821](https://github.com/everycure-org/matrix/pull/821)
- Securing the external HTTP routes with AIP [#1361](https://github.com/everycure-org/matrix/pull/1361)
- do not schedule non gpu pods on gpu nodes [#1384](https://github.com/everycure-org/matrix/pull/1384)
- slightly better naming for release runs [#1421](https://github.com/everycure-org/matrix/pull/1421)
- Add data cleaning & preprocessing for Embiology KG [#1431](https://github.com/everycure-org/matrix/pull/1431)
- add primary source to edge, fixes #888 [#1357](https://github.com/everycure-org/matrix/pull/1357)

### Documentation ✏️
- Add filtering pipeline docs [#1435](https://github.com/everycure-org/matrix/pull/1435)
- History rewrite ADR [#1356](https://github.com/everycure-org/matrix/pull/1356)

### Bugfixes 🐛
- Avoid overwriting raw data with fabricator pipeline [#554](https://github.com/everycure-org/matrix/pull/554)
- Bugfix/gpu resources [#621](https://github.com/everycure-org/matrix/pull/621)
- neo4j wrong config map for advertised URL [#1364](https://github.com/everycure-org/matrix/pull/1364)
- Hotfix: Fix schema fo `:Label` [#1390](https://github.com/everycure-org/matrix/pull/1390)
- Feat/fix column typing causing errors in spoke normalization nodes [#1392](https://github.com/everycure-org/matrix/pull/1392)
- Fix submodules and make lock [#1429](https://github.com/everycure-org/matrix/pull/1429)
- Change embiology version to string to avoid encoding octals [#1440](https://github.com/everycure-org/matrix/pull/1440)
- Bugfix/add trigger_release label to argo event source [#935](https://github.com/everycure-org/matrix/pull/935)
- Cron for neo4j restarting to avoid outdated certificates [#1280](https://github.com/everycure-org/matrix/pull/1280)
- Added PAT Token to github action [#1444](https://github.com/everycure-org/matrix/pull/1444)

### Other Changes
- Public data release bucket infra code [#1074](https://github.com/everycure-org/matrix/pull/1074)
- Remove git-crypt for almost everyone except admins [#1053](https://github.com/everycure-org/matrix/pull/1053)
- Fix the label selector in the workflow-controller Service. [#1056](https://github.com/everycure-org/matrix/pull/1056)
- Bugfix/gpu fix 2 [#635](https://github.com/everycure-org/matrix/pull/635)
- Add IAM as terraform module for code centric IAM management of the project [#628](https://github.com/everycure-org/matrix/pull/628)
- Add score API key [#1163](https://github.com/everycure-org/matrix/pull/1163)
- Add MoA visualizer [#712](https://github.com/everycure-org/matrix/pull/712)
- Improvement/make grafana pod stateful [#1226](https://github.com/everycure-org/matrix/pull/1226)
- Add DM ability to admin the cluster [#721](https://github.com/everycure-org/matrix/pull/721)
- increase component reusability across dev and prod [#1259](https://github.com/everycure-org/matrix/pull/1259)
- Big memory /cost optimized nodes  [#767](https://github.com/everycure-org/matrix/pull/767)
- Fix the bug where the presync actions get stuck in an infinite delete-recreate loop. [#1292](https://github.com/everycure-org/matrix/pull/1292)
- Make mlflow's postgres password available via an additional key. [#1293](https://github.com/everycure-org/matrix/pull/1293)
- remove gateway-infra namespace requirement [#1295](https://github.com/everycure-org/matrix/pull/1295)
- Fix data-release app's tester workflow. [#1294](https://github.com/everycure-org/matrix/pull/1294)
- debug: allow the tech team to impersonate service accounts [#768](https://github.com/everycure-org/matrix/pull/768)
- Add argo deployment of kg-dashboard pointing at development branch [#782](https://github.com/everycure-org/matrix/pull/782)
- Improvement/argo cd dev app cleanup [#1298](https://github.com/everycure-org/matrix/pull/1298)
- fix https redirect for api [#1299](https://github.com/everycure-org/matrix/pull/1299)
- delete moa argocd app [#1302](https://github.com/everycure-org/matrix/pull/1302)
- delete pubmedbert argocd app [#1301](https://github.com/everycure-org/matrix/pull/1301)
- Add accidental deletion safety mechanisms into argocd apps [#1318](https://github.com/everycure-org/matrix/pull/1318)
- Parametrise argocd apps to enable multi-env deployments. [#1319](https://github.com/everycure-org/matrix/pull/1319)
- fix for wrong role for SSH login [#1321](https://github.com/everycure-org/matrix/pull/1321)
- filter for infra branch for paths filter [#1322](https://github.com/everycure-org/matrix/pull/1322)
- Schedules only workflow jobs on big node types [#1324](https://github.com/everycure-org/matrix/pull/1324)
- Feature/cross account permissions to dev bucket from the prod project. [#1331](https://github.com/everycure-org/matrix/pull/1331)
- enable multi-env runs + toggle private datasets capability [#1326](https://github.com/everycure-org/matrix/pull/1326)
- prune = true in app of apps [#1332](https://github.com/everycure-org/matrix/pull/1332)
- feat/trigger release from gh action [#819](https://github.com/everycure-org/matrix/pull/819)
- fix broken path filter in infra deploy [#1335](https://github.com/everycure-org/matrix/pull/1335)
- Adding Trust Score to Evidence.dev calculated from Knowledge Levels [#1348](https://github.com/everycure-org/matrix/pull/1348)
- move app version on page [#831](https://github.com/everycure-org/matrix/pull/831)
- Revert "Enable Neo4J endpoint for all releases" [#841](https://github.com/everycure-org/matrix/pull/841)
- Feat/neo4j endpoint [#842](https://github.com/everycure-org/matrix/pull/842)
- de-duplicate data-release yaml files [#843](https://github.com/everycure-org/matrix/pull/843)
- auto-encrypt credential files that might be dropped by mistake in parent folder [#1363](https://github.com/everycure-org/matrix/pull/1363)
- Fixes retention to 180d + use SSD for grafana + gives people access to submit workflows [#856](https://github.com/everycure-org/matrix/pull/856)
- Feature/adapt ci for multi env deployments [#1371](https://github.com/everycure-org/matrix/pull/1371)
- Merge/main to infra to main [#854](https://github.com/everycure-org/matrix/pull/854)
- Set fixed depth overrides in association summary sankey chart [#1376](https://github.com/everycure-org/matrix/pull/1376)
- Tighten requirements for release tag [#1372](https://github.com/everycure-org/matrix/pull/1372)
- Fix/sample run [#1381](https://github.com/everycure-org/matrix/pull/1381)
- try out more memory for the OOM spoke node - normalize-spoke-edges [#1391](https://github.com/everycure-org/matrix/pull/1391)
- Hotfix: update labels attribute with null array [#1393](https://github.com/everycure-org/matrix/pull/1393)
- mini improvement in run names [#1402](https://github.com/everycure-org/matrix/pull/1402)
- Fix remove spoke from settings [#1414](https://github.com/everycure-org/matrix/pull/1414)
- increase k8s memory allocation for filtering pipeline [#1413](https://github.com/everycure-org/matrix/pull/1413)
- Add the 'in' operator filtering on pipeline name in argo. [#920](https://github.com/everycure-org/matrix/pull/920)
- Fix submodules in github actions [#1437](https://github.com/everycure-org/matrix/pull/1437)
- add report [#1441](https://github.com/everycure-org/matrix/pull/1441)
- Hotfix / Update matplotlib writers to datasets [#1438](https://github.com/everycure-org/matrix/pull/1438)
- Adds 3 new git-crypt secret keys  [#947](https://github.com/everycure-org/matrix/pull/947)
- Enhancement/trigger test data release [#979](https://github.com/everycure-org/matrix/pull/979)
- [Hotfix] SSL cert not auto updating for dev cluster [#976](https://github.com/everycure-org/matrix/pull/976)
- Enable permission to submit jobs to all members of matrix org [#981](https://github.com/everycure-org/matrix/pull/981)
- Take out project-id as a variable in terraform [#987](https://github.com/everycure-org/matrix/pull/987)
- Improvement/aip 204 env parametrize the dns tf module [#1252](https://github.com/everycure-org/matrix/pull/1252)
- add matrix ui argo cred [#1307](https://github.com/everycure-org/matrix/pull/1307)
- Improvement/aip 204 env parametrize the dns tf module [#1252](https://github.com/everycure-org/matrix/pull/1252)
- Grant workflow identity mgmt permissions to tech team [#760](https://github.com/everycure-org/matrix/pull/760)
- Create a new git-crypt key to store infra-related secrets for production [#1355](https://github.com/everycure-org/matrix/pull/1355)
- Securing the external HTTP routes with AIP [#1361](https://github.com/everycure-org/matrix/pull/1361)
- Pull data fabricator out of repository [#1325](https://github.com/everycure-org/matrix/pull/1325)
- Add ROBOKOP as exception for upstream_data_source_filtering [#1404](https://github.com/everycure-org/matrix/pull/1404)
- Securing the external HTTP routes with AIP [#1361](https://github.com/everycure-org/matrix/pull/1361)
- refactor dashboard prefix page to use a table rather than an endless bar chart [#1411](https://github.com/everycure-org/matrix/pull/1411)