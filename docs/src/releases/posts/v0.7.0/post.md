---
draft: false
date: 2025-06-09
categories:
  - Release
authors:
  - amyford
  - eKathleenCarter
  - JacquesVergine
  - Dashing-Nelson
  - alexeistepa
  - lvijnck
  - MariaHei
  - kevinschaper
  - piotrkan
  - emil-k
- leelancashire
---

### Exciting New Features 🎉
- Integrate off label dataset in the data pipeline [#1505](https://github.com/everycure-org/matrix/pull/1505)
- Implement off-label evaluation metric [#1509](https://github.com/everycure-org/matrix/pull/1509)
- Add core entities QC page to the KG dashboard [#1528](https://github.com/everycure-org/matrix/pull/1528) 
- Matrix output transformation pipeline [#1492](https://github.com/everycure-org/matrix/pull/1492)
- Use core entities for drug and disease list [#1485](https://github.com/everycure-org/matrix/pull/1485)
- Add median edge number for drug and disease nodes to KG Dashboard [#1562](https://github.com/everycure-org/matrix/pull/1562)
- Add Metrics section to KG Dashboard [#1564](https://github.com/everycure-org/matrix/pull/1564)
### Experiments 🧪
### Bugfixes 🐛
- Fix production ADR metadata [#1538](https://github.com/everycure-org/matrix/pull/1538)
- Added code to get token from google via SA File [#1460](https://github.com/everycure-org/matrix/pull/1460)
- Fix data fabricator's generate unique id function [#1463](https://github.com/everycure-org/matrix/pull/1463)
- [Hotfix] Add GCP_TOKEN to create-sample-release action [#1497](https://github.com/everycure-org/matrix/pull/1497)
- Fix headless flag removing manual release prompt [#1495](https://github.com/everycure-org/matrix/pull/1495)
- Generate reports once - not per fold [#1498](https://github.com/everycure-org/matrix/pull/1498)
- Fix sentinel node for patch and minor releases [#1526](https://github.com/everycure-org/matrix/pull/1526)
- Fix sampling pipeline break after rewrite push [#1457](https://github.com/everycure-org/matrix/pull/1457)
- Added missing variables to github actions [#1494](https://github.com/everycure-org/matrix/pull/1494)
- Clean up extra duplicate counts on merged_kg dashboard page [#1486](https://github.com/everycure-org/matrix/pull/1486)
- (Dashboard) Fix duplicated edge types (and remove unused queries) [#1465](https://github.com/everycure-org/matrix/pull/1465)
- bugfix - make sentinel data release the last node. [#1517](https://github.com/everycure-org/matrix/pull/1517)
- Hotfix: Revert "Model prediction on spark dataframe instead of pandas dataframe" [#1507](https://github.com/everycure-org/matrix/pull/1507)
- Hotfix, update disease and drug list transformer [#1518](https://github.com/everycure-org/matrix/pull/1518)
- Update core entities' version to v0.1.1 [#1525](https://github.com/everycure-org/matrix/pull/1525)
- [EC-237] Docker build fails after fabrication extraction [#1456](https://github.com/everycure-org/matrix/pull/1456)
### Technical Enhancements 🧰
- Upgraded the storage size from 128G to 512G [#1546](https://github.com/everycure-org/matrix/pull/1546)
- Add 2 new filter functions [#1454](https://github.com/everycure-org/matrix/pull/1454)
- Refactor embiology preprocessing [#1462](https://github.com/everycure-org/matrix/pull/1462)
- Upped the memory for running pipeline make_predictions_and_sort_fold [#1490](https://github.com/everycure-org/matrix/pull/1490)
- Changed spark.driver.maxResultSize from 12g to 18g [#1487](https://github.com/everycure-org/matrix/pull/1487)
- swap summary page to be the dashboard homepage, rename home page to association summary [#1466](https://github.com/everycure-org/matrix/pull/1466)
- Add --confirm-release flag to prevent manual releases [#1489](https://github.com/everycure-org/matrix/pull/1489)
### Documentation ✏️
- Update release docs to reflect that releases should only be run by automation [#1455](https://github.com/everycure-org/matrix/pull/1455)
- Fix production ADR metadata [#1538](https://github.com/everycure-org/matrix/pull/1538)
- Update filtering docs with specific dedup examples [#1453](https://github.com/everycure-org/matrix/pull/1453)
- Infra doc/production access documentation [#1544](https://github.com/everycure-org/matrix/pull/1544)
- Add KG dashboard link in documentation [#1483](https://github.com/everycure-org/matrix/pull/1483)
- Add link to KG dashboard in KG release PR [#1527](https://github.com/everycure-org/matrix/pull/1527)

### Other Changes
- Add Maria to workbench access [#1536](https://github.com/everycure-org/matrix/pull/1536)
- Create workbench for Jane [#1458](https://github.com/everycure-org/matrix/pull/1458)
- Added getting IAP token from actions [#1469](https://github.com/everycure-org/matrix/pull/1469)
- add unit tests whether the sentinel data release node runs last [#1523](https://github.com/everycure-org/matrix/pull/1523)
- Add echo in Kedro release action  [#1568](https://github.com/everycure-org/matrix/pull/1568)
- Sample less edges in sampling environment [#1510](https://github.com/everycure-org/matrix/pull/1510)
- Remove unused imports via ruff [#1531](https://github.com/everycure-org/matrix/pull/1531)
- Rename KG dashboards files and actions [#1547](https://github.com/everycure-org/matrix/pull/1547)
- Change MatplotlibWriter to MatplotlibDataset [#1472](https://github.com/everycure-org/matrix/pull/1472)
- Remove unused tags from integration test and nodes [#1535](https://github.com/everycure-org/matrix/pull/1535)
