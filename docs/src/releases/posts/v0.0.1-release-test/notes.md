---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - lvijnck
  - marcello-deluca
  - JacquesVergine
  - emil-k
  - alexeistepa
  - matwasilewski
  - jdr0887
  - Siyan-Luo
  - MariaHei
  - app/github-actions
  - oliverw1
  - elliottsharp
  - pascalwhoop
  - piotrkan
  - matej-macak
---
## Breaking Changes 🛠

No breaking changes were introduced in this release.

## Exciting New Features 🎉

- Automated periodic release submissions (#877): A scheduled GitHub Action now periodically submits the Kedro pipeline, automating the release process and reducing manual effort.
- K-fold cross-validation (#683): This significantly improves model evaluation robustness by generating multiple train-test splits.
- Flexible aggregation function for ensemble models (#905): Allows flexible aggregation methods for ensemble models, enhancing model performance and customization.
- Intermediate release support (#957): The release workflow now supports intermediate releases, providing greater flexibility in release management and enabling more granular control over the release process.
- Matrix CLI enhancements (#959): The Matrix CLI now includes mkdocs frontmatter for release notes and articles, streamlining documentation integration and enhancing workflow efficiency.


## Experiments 🧪

No new experiments were reported in this release.

## Bugfixes 🐛

- Fix protocol for `wipe_neo` script (#899): Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- Bugfix/release make Kedro hooks covered by integration tests (#900): Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- Fixes the SchemaError: `ingest_nodes`: non-nullable column `name` contains null (#943): Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- `fix: branch off from the commit that triggered the release` (#950): Addresses an issue where release branches were created from the wrong commit, ensuring consistency in release management.
- Fix/get git sha using the correct command (#974): Correctly retrieves the git SHA using the correct command (`git rev-parse HEAD`), ensuring accurate version tracking and release identification.
- Bugfix/allocate resources to main container in Neo4j template (#977): Ensures the main container in the Neo4j template receives the appropriate resources, improving performance and stability.
- Fix: Missing access to GH token in workflow (#990): Resolves an issue where the GitHub Action was not accessing the necessary token due to a configuration error. This ensures the action can now create GitHub releases.
- Fix: Jinja template placeholders had a different name (#986): Corrects a naming discrepancy between variables used in the Jinja templating engine and their actual values. This ensures that the correct information is now passed to the templating process.

## Technical Enhancements 🧰

- Refactor to remove the `refit` library (#811): Streamlined the codebase by removing the `refit` library, simplifying dependencies and maintenance.
- Simplify Neo4j SSL setup (#878): Improved the Neo4j SSL configuration for easier setup and better security.
- Update Neo4j connection string in template (#880): Updated the Neo4j connection string in templates for improved consistency and clarity.
- Replace `argo_node` function in `pipeline.py` with `ArgoNode` class (#885): Improved code structure and organization by refactoring the `argo_node` function into the `ArgoNode` class.
- Use `pyspark.sql` consistently (#923): Improved code consistency and clarity by using `pyspark.sql` consistently throughout the codebase.
- Modelling cleanup - unify splits (#907): Improved the modelling process by unifying split generation, improving code clarity and consistency.
- Allow overriding dynamic pipeline options + resolver to load settings in catalog (#901): Enhances flexibility by enabling the overriding of dynamic pipeline options through environment variables. Introduces a resolver function to seamlessly load settings from the catalog, improving configuration manageability.
- Consistent use of pyspark.sql (#923): Enhanced code clarity and maintainability.
- Refactored test suite (#930): Improved the structure and maintainability of the testing suite.
- Added import sorting (#931): Added import sorting to improve code readability and consistency.
- Check if branch starts with 'release' when triggering a data release (#921): Added validation to ensure that only branches starting with 'release' can trigger data releases, preventing accidental releases from unintended branches. 
- Only allow one model per modelling pipeline (#924): Improved the pipeline by ensuring only one model is processed within the modelling stage, preventing confusion and unexpected behavior.
- Improved handling of intermediate releases (#957): Improved the handling of intermediate releases, allowing more flexible release management.
- Improved handling of empty parameter files (#968): Added validation to prevent issues caused by empty parameters files in the pipeline. 
- Improved handling of intermediate releases (#957): Improved the handling of intermediate releases, allowing more flexible release management.
- Added integration testing for changelog generation (#968): Improved reliability and prevented regressions.
- Refactored the code to remove the `refit` library (#811).  Simplified dependencies and improved maintainability.
- The `argo_node` function was replaced with the `ArgoNode` class (#885).
- The code now consistently uses `pyspark.sql` (#923), improving clarity and consistency.
- The test suite was improved (#930)
- Added import sorting (#931) for improved code readability.


## Documentation ✏️

- Add unit tests and very minor improvement to `apply_transform` (#808): Improved test coverage for the `apply_transform` function and added minor improvements to improve maintainability.
- Onboarding fixes (#902): Improved onboarding documentation and materials.
- Update common_errors.md (#925): Updated the common errors document with solutions for new problems.
- Upgrade the required java version 11 > 17 in the docs and in the docker image (#903): Updated the required Java version to 17, resolving potential compatibility issues.
- Add `libomp` library to installation documentation (#934): Updated installation instructions to include the `libomp` library.
- Update docs on how to create a release (#940): Updated the documentation on how to create a release.
- Fix minor typos in Kedro Extensions' documentation (#913): Corrected minor typos in the Kedro Extensions documentation.
- Update virtual environment onboarding documentation (#906): Updated the virtual environment setup in the onboarding instructions.
- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Simplify Neo4J SSL setup (#878): Updated the documentation on how to configure Neo4J SSL.
- Update index.md (#956): Updated the main onboarding documentation page, ensuring better navigation and improved content clarity.
- Have Matrix CLI include mkdocs frontmatter for release notes and articles (#959): Enhanced the functionality of the Matrix CLI, embedding mkdocs frontmatter for efficient integration with documentation articles and release notes.
- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Simplify Neo4J SSL setup (#878): Updated the documentation on how to configure Neo4J SSL.
- Renaming a confusing flag column name (#893): Updated the documentation to ensure the column names are meaningful.


## Newly onboarded colleagues 🚤

- Add key Jacques (#883): Onboarded Jacques and added his key to the `git-crypt` configuration.
- Add key Matej (#886): Onboarded Matej and added his key to the `git-crypt` configuration.
- Add git-crypt key for Kushal (#904): Onboarded Kushal and added his key to the `git-crypt` configuration.
- Add git crypt public key for MariaHei (#944)


## Other Changes

- Feat/render release info docs (#858): Added feature to render release info in the documentation.
- Save `not treat` and `unknown` scores for full matrix (#853): Added columns to store 'not treat' and 'unknown' scores in the full matrix.
- Have Test DataRelease on Main (#989): Added a test for the data release workflow to run on the main branch. This ensures that the workflow is properly tested on the main branch.
- Disable SSL for local Neo4J (#972): This changes makes it easier to develop with Neo4J locally, without having to manage any certificates.
- Only allow one model per modelling pipeline (#924): This change ensures that the modelling pipeline only trains one model at a time, improving the maintainability and clarity of the pipeline.
- Incrementing Spoke Version (#914): Updated the version number for the SPOKE knowledge graph.
- Have matrix CLI include mkdocs frontmatter for release notes and articles (#959): Added functionality to the matrix CLI which allows it to generate markdown files ready to be used on mkdocs website.
- Improved handling of intermediate releases (#957): Improved the handling of intermediate releases, allowing more flexible release management.
- Addressed an issue with release branches being created from the incorrect commit (#950).
- Improved the handling of empty parameter files (#968), preventing pipeline errors.
- Added several new authors to the documentation (.authors.yml).
- Added a test for the data release workflow to run on the main branch (#989)
- Improved the handling of empty parameter files (#968).
- Added validation to prevent accidental releases from unintended branches (#921).
- Improved the pipeline by ensuring only one model is processed within the modelling stage (#924).
- Added a test for data release workflow on the main branch (#989).
- Corrected a naming discrepancy in Jinja template placeholders (#986).
- Improved handling of intermediate releases (#957).
- Added integration testing for changelog generation (#968).
