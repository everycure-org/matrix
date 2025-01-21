---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - JacquesVergine
  - marcello-deluca
  - lvijnck
  - emil-k
  - alexeistepa
  - jdr0887
  - matwasilewski
  - Siyan-Luo
  - oliverw1
  - MariaHei
  - app/github-actions
  - elliottsharp
  - pascalwhoop
  - piotrkan
  - matej-macak
---
## Breaking Changes 🛠

No breaking changes were introduced in this release.

## Exciting New Features 🎉

- Implement k-fold cross-validation (#683): This significantly improves model evaluation robustness by generating multiple train-test splits.
- Add parameter for ensemble model aggregation function (#905): Allows flexible aggregation methods for ensemble models, enhancing model performance and customization.
- Automate kedro submit release run periodically (#877):  Introduces a scheduled job to automatically submit the Kedro pipeline for periodic data releases.


## Experiments 🧪

No new experiments were reported in this release.

## Bugfixes 🐛

- Fix protocol for `wipe_neo` script (#899): Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- Bugfix/release make kedro hooks covered by integration tests (#900): Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- Fixes the SchemaError: `ingest_nodes`: non-nullable column `name` contains null (#943): Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- `fix: branch off from the commit that triggered the release` (#950): Addresses an issue where release branches were created from the wrong commit, ensuring consistency in release management.
- Make CI checks pass on main again (#967): Addresses issues causing CI failures on the `main` branch.
- Fix/get git sha using the correct command (#974):  Fixes the way the git sha is fetched, using the correct command `git rev-parse HEAD`.
- Bugfix/allocate resources to main container in neo4j template (#977): Allocates resources to the main container in the Neo4j template.
- Fix: missing access to GH token in workflow (#990): Fixes an issue preventing access to the GitHub token in the workflow.


## Technical Enhancements 🧰

- Refactor to remove the `refit` library (#811): Streamlined the codebase by removing the `refit` library, simplifying dependencies and maintenance.
- Simplify Neo4J SSL setup (#878): Improved the Neo4j SSL configuration for easier setup and better security.
- Update Neo4j connection string in template (#880): Updated the Neo4j connection string in templates for improved consistency and clarity.
- Replace `argo_node` function in `pipeline.py` with `ArgoNode` class (#885): Improved code structure and organization by refactoring the `argo_node` function into the `ArgoNode` class.
- Use `pyspark.sql` consistently (#923): Improved code consistency and clarity by using `pyspark.sql` consistently throughout the codebase.
- Feat/rename `object` kw (#922): Renamed the `object` keyword to improve readability and avoid conflicts.
- Feat/refactor test suite (#930): Improved the structure and maintainability of the testing suite.
- Feat/add import sorting (#931): Added import sorting to improve code readability and consistency.
- Feat/add trigger release label to argo (#936): Enhanced Argo workflows by adding labels to indicate data release triggers.
- Modelling cleanup - unify splits (#907): Improved the modelling process by unifying split generation, improving code clarity and consistency.
- Allow overriding dynamic pipeline options + resolver to load settings in catalog (#901): Improved flexibility by enabling the override of dynamic pipeline options and adding a resolver to load settings from the catalog.
- Only allow one model per modelling pipeline (#924): Improves clarity and prevents potential conflicts by restricting modelling pipelines to a single model.
- Have Test DataRelease on Main (#989): Enables testing for data release pipeline directly from main branch.
- Check if branch starts with 'release' when triggering a data release (#921): Added check to ensure that data release is only triggered from branches starting with 'release/'.
- Check if tag exists before submit (#983): Added check to prevent the submission of a release if the Git tag already exists.
- Disable SSL for local Neo4J (#972): Disables SSL for local Neo4J instances to simplify local development setup.


## Documentation ✏️

- Add unit tests and very minor improvement to `apply_transform` (#808): Improved test coverage for the `apply_transform` function and added minor improvements to improve maintainability.
- Add Kedro resource documentation (#919): Added documentation for Kedro resources.
- Update common_errors.md (#925): Updated the common errors document with solutions for new problems.
- Onboarding fixes (#902): Improved onboarding documentation and materials.
- Upgrade the required java version 11 > 17 in the docs and in the docker image (#903): Updated the required Java version to 17, resolving potential compatibility issues.
- Add libomp library to installation documentation (#934): Updated installation instructions to include the `libomp` library.
- Update docs on how to create a release (#940): Updated the documentation on how to create a release.
- Fix minor typos in Kedro Extensions' documentation (#913): Corrected minor typos in the Kedro Extensions documentation.
- Update virtual environment onboarding documentation (#906): Updated the virtual environment setup in the onboarding instructions.
- Update index.md (#956): updated the `index.md` with updated instructions
- Simplify Neo4J SSL setup (#878): Updated the documentation on how to configure Neo4J SSL.
- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Have matrix CLI include mkdocs frontmatter for release notes and articles (#959): Matrix CLI now includes mkdocs frontmatter for release notes and articles.


## Newly onboarded colleagues 🚤

- Add key Jacques (#883): Onboarded Jacques and added his key to the git-crypt configuration.
- Add key Matej (#886): Onboarded Matej and added his key to the git-crypt configuration.
- Add git-crypt key for Kushal (#904): Onboarded Kushal and added his key to the git-crypt configuration.
- Add git crypt public key for MariaHei (#944): Onboarded MariaHei and added her key to the git-crypt configuration.


## Other Changes

- Feat/render release info docs (#858): Added feature to render release info in the documentation.
- Incrementing spoke version (#914): Updated the SPOKE version.
- `Debug: List the directory to debug the not found error (#988): Added debug step to identify the root cause of a "file not found" error.
- Feat/allow intermediate non-latest releases (#957): Allow the creation of intermediate non-latest releases.
- Only allow one model per modelling pipeline (#924): Improves clarity and prevents potential conflicts by restricting modelling pipelines to a single model.