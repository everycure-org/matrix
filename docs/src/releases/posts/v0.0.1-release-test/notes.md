---
draft: false
date: 
categories:
  - Release
authors:
  - JacquesVergine
  - emil-k
  - marcello-deluca
  - lvijnck
  - alexeistepa
  - matwasilewski
  - Siyan-Luo
  - jdr0887
  - MariaHei
  - oliverw1
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
- Add support for intermediate non-latest releases (#957): Allows the creation of non-latest releases, improving flexibility in the release management process.


## Experiments 🧪

No new experiments were reported in this release.


## Bugfixes 🐛

- Fix protocol for `wipe_neo` script (#899): Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- Bugfix/release make kedro hooks covered by integration tests (#900): Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- Fixes the SchemaError: `ingest_nodes`: non-nullable column `name` contains null (#943): Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- `fix: branch off from the commit that triggered the release` (#950): Addresses an issue where release branches were created from the wrong commit, ensuring consistency in release management.
- Make CI checks pass on main again (#967): Resolved issues that were causing CI checks to fail on the `main` branch.


## Technical Enhancements 🧰

- Refactor to remove the `refit` library (#811): Streamlined the codebase by removing the `refit` library, simplifying dependencies and maintenance.
- Simplify Neo4J SSL setup (#878): Improved the Neo4j SSL configuration for easier setup and better security.
- Update Neo4j connection string in template (#880): Updated the Neo4j connection string in templates for improved consistency and clarity.
- Replace `argo_node` function in `pipeline.py` with `ArgoNode` class (#885): Improved code structure and organization by refactoring the `argo_node` function into the `ArgoNode` class.
- Use `pyspark.sql` consistently (#923): Improved code consistency and clarity by using `pyspark.sql` consistently throughout the codebase.
- Rename the `object` keyword (#922): Renamed the `object` keyword to improve readability and avoid conflicts.
- Refactor test suite (#930): Improved the structure and maintainability of the testing suite.
- Add import sorting (#931): Added import sorting to improve code readability and consistency.
- Add trigger release label to Argo (#936): Enhanced Argo workflows by adding labels to indicate data release triggers.
- Modelling cleanup - unify splits (#907): Improved the modelling process by unifying split generation, improving code clarity and consistency.
- Allow overriding dynamic pipeline options + resolver to load settings in catalog (#901): Enhanced flexibility by allowing the dynamic pipeline options to be overridden through a custom resolver.
- Expand git checks for Kedro submit (#961): Enhanced the git checks for the Kedro submit command.
- Only allow one model per modelling pipeline (#924): Ensured that only one model is processed per modelling pipeline.
- Disable SSL for local Neo4J (#972): Simplified local development by disabling SSL for the local Neo4j instance.


## Documentation ✏️

- Update common errors (#925): Updated the common errors document with solutions for new problems.
- Add Kedro resource documentation (#919): Added documentation for Kedro resources.
- Update onboarding documentation (#902, #906, #956): Improved onboarding documentation and materials with fixes for virtual environment setup and updated index.
- Upgrade the required Java version 11 > 17 in the docs and in the docker image (#903): Updated the required Java version to 17, resolving potential compatibility issues.
- Add libomp library to installation documentation (#934): Updated installation instructions to include the `libomp` library.
- Update docs on how to create a release (#940): Updated the documentation on how to create a release.
- Fix minor typos in Kedro Extensions' documentation (#913): Corrected minor typos in the Kedro Extensions documentation.
- Simplify Neo4J SSL setup (#878): Updated the documentation on how to configure Neo4J SSL.
- Add parameter for ensemble model aggregation function (#905): Improved documentation for how to add an ensemble function.
- Update index.md (#956): Added key for MariaHei.
- Add Kedro resource documentation (#919): Added documentation for Kedro resources.
- Have matrix CLI include mkdocs frontmatter for release notes and articles (#959): Improved the structure of release notes by automatically adding mkdocs frontmatter.


## Newly onboarded colleagues 🚤

- Add key Jacques (#883): Onboarded Jacques and added his key to the git-crypt configuration.
- Add key Matej (#886): Onboarded Matej and added his key to the git-crypt configuration.
- Add git-crypt key for Kushal (#904): Onboarded Kushal and added his key to the git-crypt configuration.
- Add git crypt public key for MariaHei (#944): Onboarded MariaHei and added her key to the git-crypt configuration.


## Other Changes

- Feat/render release info docs (#858): Added feature to render release info in the documentation.
- Add parameter for ensemble model aggregation function (#905): Improved documentation for how to add an ensemble function.
- Save `not treat` and `unknown` scores for full matrix (#853): Added columns to store 'not treat' and 'unknown' scores in the full matrix.
- Rename a confusing flag column name (#893): Improved clarity by renaming a confusing flag column name.
- incrementing spoke version (#914): Updated SPOKE version.
- Check if branch starts with 'release' when triggering a data release (#921): Improved data release triggers by ensuring the branch name starts with 'release'.
- Only allow one model per modelling pipeline (#924): Ensured that only one model is processed per modelling pipeline.
- Fix/get git sha using the correct command (#974): Fixed the command used to get the git SHA.
