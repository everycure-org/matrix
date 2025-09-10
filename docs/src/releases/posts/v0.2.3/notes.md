---
draft: false
date: 2025-01-15
categories:
  - Release
---
## Breaking Changes 🛠

No breaking changes were introduced in this release.

## Exciting New Features 🎉

- Implement k-fold cross-validation (#683):  This significantly improves model evaluation robustness by generating multiple train-test splits.
- Add parameter for ensemble model aggregation function (#905): Allows flexible aggregation methods for ensemble models, enhancing model performance and customization.


## Experiments 🧪

No new experiments were reported in this release.

## Bugfixes 🐛

- Fix protocol for wipe_neo script (#899): Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- Bugfix/release make kedro hooks covered by integration tests (#900): Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- Fixes the SchemaError: `ingest_nodes`: non-nullable column `name` contains null (#943): Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- `fix: branch off from the commit that triggered the release` (#950): Addresses an issue where release branches were created from the wrong commit, ensuring consistency in release management.


## Technical Enhancements 🧰

- Refactor to remove the refit library (#811): Streamlined the codebase by removing the `refit` library, simplifying dependencies and maintenance.
- Simplify Neo4J SSL setup (#878): Improved the Neo4j SSL configuration for easier setup and better security.
- Update neo4j connection string in template (#880): Updated the Neo4j connection string in templates for improved consistency and clarity.
- Replace `argo_node` function in `pipeline.py` with `ArgoNode` class (#885): Improved code structure and organization by refactoring the `argo_node` function into the ArgoNode class.
- Use `pyspark.sql` consistently (#923): Improved code consistency and clarity by using `pyspark.sql` consistently throughout the codebase.
- Feat/rename `object` kw (#922): Renamed the `object` keyword to improve readability and avoid conflicts.
- Feat/refactor test suite (#930): Improved the structure and maintainability of the testing suite.
- Feat/add import sorting (#931): Added import sorting to improve code readability and consistency.
- Feat/add trigger release label to argo (#936): Enhanced Argo workflows by adding labels to indicate data release triggers.
- Modelling cleanup - unify splits (#907): Improved the modelling process by unifying split generation, improving code clarity and consistency.


## Documentation ✏️

- Add unit tests and very minor improvement to `apply_transform` (#808): Improved test coverage for the `apply_transform` function and added minor improvements to improve maintainability.
- Deploy Grafana and Prometheus for improved cluster and experiment runs observability (#834): Improved cluster and experiment monitoring.
- Save `not treat` and `unknown` scores for full matrix (#853): Added columns to store 'not treat' and 'unknown' scores in the full matrix.
- Update common_errors.md (#925): Updated the common errors document with solutions for new problems.
- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Onboarding fixes (#902): Improved onboarding documentation and materials.
- Upgrade the required java version 11 > 17 in the docs and in the docker image (#903): Updated the required Java version to 17, resolving potential compatibility issues.
- Add libomp library to installation documentation (#934): Updated installation instructions to include the `libomp` library.
- Onboarding fixes + Add key Jacques (#883): Improved onboarding process and added key for Jacques.
- Update docs on how to create a release (#940): Updated the documentation on how to create a release.
- Fix minor typos in Kedro Extensions' documentation (#913): Corrected minor typos in the Kedro Extensions documentation.
- Update virtual environment onboarding documentation (#906): Updated the virtual environment setup in the onboarding instructions.
- Add key Matej (#886): Added Matej's key to improve collaboration.
- Create `marcello-deluca.asc` (#892): Added key for Marcello Deluca.
- Add git-crypt key for Kushal (#904): Added Kushal's key to the git-crypt configuration.
- Renaming a confusing flag column name (#893): Improved clarity by renaming a confusing flag column name.
- Add parameter for ensemble model aggregation function (#905): Improved documentation for how to add an ensemble function.
- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Simplify Neo4J SSL setup (#878): Updated the documentation on how to configure Neo4J SSL.


## Newly onboarded colleagues 🚤

- Add key Jacques (#883): Onboarded Jacques and added his key to the git-crypt configuration.
- Add key Matej (#886): Onboarded Matej and added his key to the git-crypt configuration.
- Add git-crypt key for Kushal (#904): Onboarded Kushal and added his key to the git-crypt configuration.


## Other Changes

- Feat/render release info docs (#858): Added feature to render release info in the documentation.
