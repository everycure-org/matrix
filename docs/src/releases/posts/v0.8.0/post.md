---
draft: false
date: 2025-07-01
categories:
  - Release
authors:
  - pascalwhoop
  - amyford
  - Dashing-Nelson
  - JacquesVergine
  - jdr0887
  - eKathleenCarter
  - app/github-actions
  - app/dependabot
  - piotrkan
---

### Exciting New Features 🎉
- Added impersonation of Spark Service Account through hooks. [#1575](https://github.com/everycure-org/matrix/pull/1575)
- Write matrix outputs to BQ [#1620](https://github.com/everycure-org/matrix/pull/1620)
- Cleanup script for raw data cleanup [#1519](https://github.com/everycure-org/matrix/pull/1519)
- Feat/add transformer versions [#1551](https://github.com/everycure-org/matrix/pull/1551)
- Pipeline for each evaluation of matrix [#1559](https://github.com/everycure-org/matrix/pull/1559)
- Change infra branches to main [#1595](https://github.com/everycure-org/matrix/pull/1595)
- Improve model prediction performance with Spark Pandas UDFs [#1540](https://github.com/everycure-org/matrix/pull/1540)
- Add drug and disease neighbours histograms [#1613](https://github.com/everycure-org/matrix/pull/1613)
### Experiments 🧪
### Bugfixes 🐛
- Filtering paths [#1567](https://github.com/everycure-org/matrix/pull/1567)
- Fix KG dashboard deploy action name in release action [#1570](https://github.com/everycure-org/matrix/pull/1570)
- Remove KG Dashboard deployment action default release version [#1603](https://github.com/everycure-org/matrix/pull/1603)
- Fixed broken css on evidence dashboard [#1605](https://github.com/everycure-org/matrix/pull/1605)
- Fix KG dashboard deployment action environment variable [#1614](https://github.com/everycure-org/matrix/pull/1614)
- Added missing permission for production GCP CloudBuild SA  [#1616](https://github.com/everycure-org/matrix/pull/1616)
- Fix KG Dashboard link in release PR [#1624](https://github.com/everycure-org/matrix/pull/1624)
- Fix EC clinical trials transformers to use select_cols [#1625](https://github.com/everycure-org/matrix/pull/1625)
- Update rtxkg2 transformer code [#1566](https://github.com/everycure-org/matrix/pull/1566)
- Fix drug and disease ranks [#1572](https://github.com/everycure-org/matrix/pull/1572)
- Resolve Critical Vulnerabilities in Packages and their sub-dependencies as of 13th June 2025 [#1588](https://github.com/everycure-org/matrix/pull/1588)
### Technical Enhancements 🧰
- Add uniform rank based FF transform [#1550](https://github.com/everycure-org/matrix/pull/1550)
- Added tolerations to main pod so that large instances could be tolerated [#1552](https://github.com/everycure-org/matrix/pull/1552)
- Allow sampling pipeline release_version parameter to be null [#1599](https://github.com/everycure-org/matrix/pull/1599)
- Checked and Modified nodes that don't need GPU [#1563](https://github.com/everycure-org/matrix/pull/1563)
- Grant Orchard Production Project to access Dev Bucket [#1593](https://github.com/everycure-org/matrix/pull/1593)
- Change Initial Desired state and Idle Timeout for Workbench [#1635](https://github.com/everycure-org/matrix/pull/1635)
- Checked and Modified nodes that don't need GPU [#1563](https://github.com/everycure-org/matrix/pull/1563)
- changed create-release-pr.yml github oidc from rw to ro [#1578](https://github.com/everycure-org/matrix/pull/1578)
- Infra into main [#1583](https://github.com/everycure-org/matrix/pull/1583)
- Resolve Critical Vulnerabilities in Packages and their sub-dependencies as of 13th June 2025 [#1588](https://github.com/everycure-org/matrix/pull/1588)
- Grant Orchard Production Project to access Dev Bucket [#1593](https://github.com/everycure-org/matrix/pull/1593)
- Production Infra Branch Merge into Main branch [#1596](https://github.com/everycure-org/matrix/pull/1596)
- Infrastructure/deploy main changes [#1607](https://github.com/everycure-org/matrix/pull/1607)
- Add prod data release zone bucket & infra [#1612](https://github.com/everycure-org/matrix/pull/1612)
- Added missing permission for production GCP CloudBuild SA  [#1616](https://github.com/everycure-org/matrix/pull/1616)
- Add KG Dashboard docker configuration [#1609](https://github.com/everycure-org/matrix/pull/1609)
- [KG Dashboard] Refactor project-id to environment variable [#1608](https://github.com/everycure-org/matrix/pull/1608)
- Refactor KG Dashboard deploy action parameters to environment [#1610](https://github.com/everycure-org/matrix/pull/1610)
### Documentation ✏️
- Docs/modelling [#1622](https://github.com/everycure-org/matrix/pull/1622)
- Refactor Pipeline documentation to individual sections [#1611](https://github.com/everycure-org/matrix/pull/1611)
- Create LICENSE [#1543](https://github.com/everycure-org/matrix/pull/1543)
### Other Changes
- Do not refresh Credentials for GCP SA if it is a Github Action [#1594](https://github.com/everycure-org/matrix/pull/1594)
- Removed orgPolicyAdmin from build [#1617](https://github.com/everycure-org/matrix/pull/1617)
- Change permissions to `read only` for create-release-pr.yml github oidc [#1578](https://github.com/everycure-org/matrix/pull/1578)
- Production Infra Branch Merge into Main branch [#1596](https://github.com/everycure-org/matrix/pull/1596)
- Upgraded gunicorn to version 23.0.0 [#1601](https://github.com/everycure-org/matrix/pull/1601)
- Infrastructure/deploy main changes [#1607](https://github.com/everycure-org/matrix/pull/1607)
- Bump the npm_and_yarn group across 2 directories with 5 updates [#1597](https://github.com/everycure-org/matrix/pull/1597)
- Bump the pip group across 1 directory with 3 updates [#1598](https://github.com/everycure-org/matrix/pull/1598)
- Change Initial Desired state and Idle Timeout for Workbench [#1635](https://github.com/everycure-org/matrix/pull/1635)
