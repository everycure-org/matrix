---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - JacquesVergine
  - emil-k
  - marcello-deluca
  - lvijnck
  - alexeistepa
  - matwasilewski
  - jdr0887
  - Siyan-Luo
  - MariaHei
  - oliverw1
  - elliottsharp
  - app/github-actions
  - pascalwhoop
  - piotrkan
  - matej-macak
---
## Breaking Changes 🛠

No breaking changes were introduced in this release.

## Exciting New Features 🎉

- Implement k-fold cross-validation (#683): This significantly improves model evaluation robustness by generating multiple train-test splits.
- Add parameter for ensemble model aggregation function (#905): Allows flexible aggregation methods for ensemble models, enhancing model performance and customization.
- Automate kedro submit release run periodically (#877):  Adds a GitHub Actions workflow to periodically submit Kedro pipelines for automated releases.
- Deploy Grafana and Prometheus for improved cluster and experiment runs observability (#834): Improved monitoring for the cluster and experiment runs.

## Experiments 🧪

No new experiments were reported in this release.

## Bugfixes 🐛

- Fix protocol for `wipe_neo` script (#899): Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- Bugfix/release make kedro hooks covered by integration tests (#900): Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- Fixes the SchemaError: `ingest_nodes`: non-nullable column `name` contains null (#943): Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- `fix: branch off from the commit that triggered the release` (#950): Addresses an issue where release branches were created from the wrong commit, ensuring consistency in release management.
- Add trigger release label to argo (#936): Addresses an issue where data releases were not triggered.
- Check if tag exists before submit (#983): Prevents accidental overwrites of existing releases.
- Fix/get git sha using the correct command (#974): Uses the correct git command to get the git sha, resolving issues with inconsistent sha values.
- Fix: missing access to GH token in workflow (#990): Resolves an issue where the GitHub Actions workflow could not access the GH token.


## Technical Enhancements 🧰

- Refactor to remove the `refit` library (#811): Streamlined the codebase by removing the `refit` library, simplifying dependencies and maintenance.
- Simplify Neo4J SSL setup (#878): Improved the Neo4j SSL configuration for easier setup and better security.
- Update neo4j connection string in template (#880): Updated the Neo4j connection string in templates for improved consistency and clarity.
- Replace `argo_node` function in `pipeline.py` with `ArgoNode` class (#885): Improved code structure and organization by refactoring the `argo_node` function into the `ArgoNode` class.
- Use `pyspark.sql` consistently (#923): Improved code consistency and clarity by using `pyspark.sql` consistently throughout the codebase.
- Rename `object` keyword (#922): Renamed the `object` keyword to improve readability and avoid conflicts.
- Refactor test suite (#930): Improved the structure and maintainability of the testing suite.
- Add import sorting (#931): Added import sorting to improve code readability and consistency.
- Modelling cleanup - unify splits (#907): Improved the modelling process by unifying split generation, improving code clarity and consistency.
- Allow overriding dynamic pipeline options (#901): Allows for overriding dynamic pipeline options through environment variables.
- Only allow one model per modelling pipeline (#924): Ensures only one model is trained per modelling pipeline run.
- Allocate resources to main container in neo4j template (#977): Allocates resources to the main container within Neo4j templates.
- Have Test DataRelease on Main (#989): Enables faster testing of data releases.
- Have matrix CLI include mkdocs frontmatter for release notes and articles (#959): Enhances the Markdown generation in the CLI to provide improved metadata.


## Documentation ✏️

- Add unit tests and very minor improvement to `apply_transform` (#808): Improved test coverage for the `apply_transform` function and added minor improvements to improve maintainability.
- Save `not treat` and `unknown` scores for full matrix (#853): Added columns to store 'not treat' and 'unknown' scores in the full matrix.
- Update `common_errors.md` (#925): Updated the common errors document with solutions for new problems.
- Add Kedro resource documentation (#919): Added documentation for Kedro resources.
- Onboarding fixes (#902): Improved onboarding documentation and materials.
- Update virtual environment onboarding documentation (#906): Updated the virtual environment setup in the onboarding instructions.
- Update docs on how to create a release (#940): Updated the documentation on how to create a release.
- Fix minor typos in Kedro Extensions' documentation (#913): Corrected minor typos in the Kedro Extensions documentation.
- Update required java version 11 > 17 in the docs and in the docker image (#903): Updated the required Java version to 17, resolving potential compatibility issues.
- Add libomp library to installation documentation (#934): Updated installation instructions to include the `libomp` library.
- Add key Jacques (#883): Added key for Jacques for improved onboarding collaboration.
- Upgrade to SPOKE v5.2 (#914): Upgraded the SPOKE version to V5.2.
- Update index.md (#956): Updated onboarding documentation.


## Newly onboarded colleagues 🚤

- Add key Jacques (#883): Onboarded Jacques and added his key to the git-crypt configuration.
- Add key Matej (#886): Onboarded Matej and added his key to the git-crypt configuration.
- Add git-crypt key for Kushal (#904): Onboarded Kushal and added his key to the git-crypt configuration.
- Add git crypt public key for MariaHei (#944): Onboarded MariaHei and added her key to the git-crypt configuration.


## Other Changes

- Feat/render release info docs (#858): Added feature to render release info in the documentation.
- Renaming a confusing flag column name (#893): Improved clarity by renaming a confusing flag column name.
- Only allow one model per modelling pipeline (#924): Improved pipeline management.
- Allow overriding dynamic pipeline options (#901):  Provides greater flexibility in configuring dynamic pipelines.
- Check if branch starts with 'release' when triggering a data release. (#921): Added validation for branch names.
- Make CI checks pass on main again (#967): Resolves the CI issues.
- Disable SSL for local Neo4J (#972): Disables SSL for the local Neo4J instance for easier local development.
- Add parameter for ensemble model aggregation function (#905): Improved documentation for how to add an ensemble function.
- Incrementing SPOKE version (#914)
- Improve Git checks for Kedro submit (#961): Improves the pre-commit checks to prevent invalid submissions.
- Add integration tests for generating the changelog (#968): Adds integration tests to ensure that the changelog is generated correctly.
- Allow intermediate non-latest releases (#957): Enables the creation of intermediate releases without flagging them as latest.
- Have Test DataRelease on Main (#989): Makes the testing of data release faster and easier.