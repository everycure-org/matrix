---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - lvijnck
  - emil-k
  - JacquesVergine
  - marcello-deluca
  - alexeistepa
  - matwasilewski
  - jdr0887
  - Siyan-Luo
  - app/github-actions
  - oliverw1
  - MariaHei
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
- Automate kedro submit release run periodically (#877):  Adds a scheduled GitHub Action to periodically submit the Kedro pipeline for automating releases. This helps ensure regular updates and reduces manual effort.


## Experiments 🧪

No new experiments were reported in this release.

## Bugfixes 🐛

- Fix protocol for `wipe_neo` script (#899): Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- Bugfix/release make Kedro hooks covered by integration tests (#900): Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- Fixes the SchemaError: `ingest_nodes`: non-nullable column `name` contains null (#943): Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- `fix: branch off from the commit that triggered the release` (#950): Addresses an issue where release branches were created from the wrong commit, ensuring consistency in release management.
- Fix/get git sha using the correct command (#974): Correctly retrieves the git SHA using the correct command (`git rev-parse HEAD`), ensuring accurate version tracking and release identification.
- Bugfix/allocate resources to main container in Neo4j template (#977): Ensures the main container in the Neo4j template receives the appropriate resources, improving performance and stability.
- Add git-crypt key for Kushal (#904)
- Add git crypt public key for MariaHei (#944)
- Make CI checks pass on main again (#967)
- Fix: Missing access to GH token in workflow (#990): Resolves an issue where the GitHub Action was not accessing the necessary token due to a configuration error. This ensures the action can now create GitHub releases.
- Fix: Jinja template placeholders had a different name (#986): Corrects a naming discrepancy between variables used in the Jinja templating engine and their actual values. This ensures that the correct information is now passed to the templating process.
- Fix: Missing access to GH token in workflow (#990): Fixes an issue causing the github action to not function as intended because it could not access the GH token.


## Technical Enhancements 🧰

- Refactor to remove the `refit` library (#811): Streamlined the codebase by removing the `refit` library, simplifying dependencies and maintenance.
- Simplify Neo4j SSL setup (#878): Improved the Neo4j SSL configuration for easier setup and better security.
- Update Neo4j connection string in template (#880): Updated the Neo4j connection string in templates for improved consistency and clarity.
- Replace `argo_node` function in `pipeline.py` with `ArgoNode` class (#885): Improved code structure and organization by refactoring the `argo_node` function into the `ArgoNode` class.
- Use `pyspark.sql` consistently (#923): Improved code consistency and clarity by using `pyspark.sql` consistently throughout the codebase.
- Feat/rename `object` kw (#922): Renamed the `object` keyword to improve readability and avoid conflicts.
- Feat/refactor test suite (#930): Improved the structure and maintainability of the testing suite.
- Feat/add import sorting (#931): Added import sorting to improve code readability and consistency.
- Modelling cleanup - unify splits (#907): Improved the modelling process by unifying split generation, improving code clarity and consistency.
- Allow overriding dynamic pipeline options + resolver to load settings in catalog (#901): Enhances flexibility by enabling the overriding of dynamic pipeline options through environment variables. Introduces a resolver function to seamlessly load settings from the catalog, improving configuration manageability.
- Feat: Allow intermediate non-latest releases (#957): This enhancement allows the creation of non-latest releases in the Github workflow. 
- Check if branch starts with 'release' when triggering a data release (#921): Added validation to ensure that only branches starting with 'release' can trigger data releases, preventing accidental releases from unintended branches. 
- Only allow one model per modelling pipeline (#924): Improved the pipeline by ensuring only one model is processed within the modelling stage, preventing confusion and unexpected behavior.


## Documentation ✏️

- Add unit tests and very minor improvement to `apply_transform` (#808): Improved test coverage for the `apply_transform` function and added minor improvements to improve maintainability.
- Add Kedro resource documentation (#919): Added documentation on how to define Kedro resources.
- Onboarding fixes (#902): Improved onboarding documentation and materials.
- Update common_errors.md (#925): Updated the common errors document with solutions for new problems.
- Upgrade the required java version 11 > 17 in the docs and in the docker image (#903): Updated the required Java version to 17, resolving potential compatibility issues.
- Add `libomp` library to installation documentation (#934): Updated installation instructions to include the `libomp` library.
- Update docs on how to create a release (#940): Updated the documentation on how to create a release.
- Fix minor typos in Kedro Extensions' documentation (#913): Corrected minor typos in the Kedro Extensions documentation.
- Update virtual environment onboarding documentation (#906): Updated the virtual environment setup in the onboarding instructions.
- Deploy Grafana and Prometheus for improved cluster and experiment runs observability (#834): Improved cluster and experiment monitoring.
- Update index.md (#956): Updated the main onboarding documentation page, ensuring better navigation and improved content clarity.
- Have Matrix CLI include mkdocs frontmatter for release notes and articles (#959): Enhanced the functionality of the Matrix CLI, embedding mkdocs frontmatter for efficient integration with documentation articles and release notes.
- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Simplify Neo4J SSL setup (#878): Updated the documentation on how to configure Neo4J SSL.
- Rename a confusing flag column name (#893): Updated the documentation to ensure the column names are meaningful.


## Newly onboarded colleagues 🚤

- Add key Jacques (#883): Onboarded Jacques and added his key to the `git-crypt` configuration.
- Add key Matej (#886): Onboarded Matej and added his key to the `git-crypt` configuration.
- Add key for Marcello Deluca (#892): Onboarded Marcello Deluca and added his key to the `git-crypt` configuration.


## Other Changes

- Feat/render release info docs (#858): Added feature to render release info in the documentation.
- Save `not treat` and `unknown` scores for full matrix (#853): Added columns to store 'not treat' and 'unknown' scores in the full matrix.
- Have Test DataRelease on Main (#989): Added a test for the data release workflow to run on the main branch. This ensures that the workflow is properly tested on the main branch.
- Disable SSL for local Neo4J (#972): This changes makes it easier to develop with Neo4J locally, without having to manage any certificates.
- Only allow one model per modelling pipeline (#924): This change ensures that the modelling pipeline only trains one model at a time, improving the maintainability and clarity of the pipeline.
- Incrementing Spoke Version (#914): Updated the version number for the SPOKE knowledge graph.
- Have matrix CLI include mkdocs frontmatter for release notes and articles (#959): Added functionality to the matrix CLI which allows it to generate markdown files ready to be used on mkdocs website.
- Improve handling of intermediate releases (#957): Improved the handling of intermediate releases, allowing more flexible release management.
- Improve handling of empty parameter files (#968): Added validation to prevent issues caused by empty parameters files in the pipeline. 
- Add Integration Tests for Generating the Changelog (#968): This enhancement includes integration testing for the changelog generation process, enhancing reliability and preventing regressions.