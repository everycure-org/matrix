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
- Automate kedro submit release run periodically (#877): Automated data releases are now run periodically.
- Add trigger release label to argo (#936): This enhanced Argo workflows by adding labels to indicate data release triggers.

## Experiments 🧪

No new experiments were reported in this release.

## Bugfixes 🐛

- Fix protocol for `wipe_neo` script (#899): Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- `Bugfix/release` make kedro hooks covered by integration tests (#900): Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- Fixes the SchemaError: `ingest_nodes`: non-nullable column `name` contains null (#943): Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- `fix: branch off from the commit that triggered the release` (#950): Addresses an issue where release branches were created from the wrong commit, ensuring consistency in release management.
- Make CI checks pass on main again (#967): Resolved issues that prevented CI checks from passing on the main branch.
- Fix/get git sha using the correct command (#974): The git SHA is now fetched using the correct command, improving reliability.
- Bugfix/allocate resources to main container in neo4j template (#977): Fixed resource allocation in the Neo4j Docker Compose template, enhancing performance.
- Fix: missing access to GH token in workflow (#990): Resolved an issue where the GitHub Actions workflow lacked access to the GitHub token.


## Technical Enhancements 🧰

- Refactor to remove the `refit` library (#811): Streamlined the codebase by removing the `refit` library, simplifying dependencies and maintenance.
- Setup batch pipeline + extract normalization + simplify preprocessing to remove normalization (#766): Improved the batch pipeline and simplified preprocessing by removing normalization.
- Simplify Neo4J SSL setup (#878): Improved the Neo4j SSL configuration for easier setup and better security.
- Update neo4j connection string in template (#880): Updated the Neo4j connection string in templates for improved consistency and clarity.
- Replace `argo_node` function in `pipeline.py` with `ArgoNode` class (#885): Improved code structure and organization by refactoring the `argo_node` function into the `ArgoNode` class.
- Use `pyspark.sql` consistently (#923): Improved code consistency and clarity by using `pyspark.sql` consistently throughout the codebase.
- Feat/rename `object` kw (#922): Renamed the `object` keyword to improve readability and avoid conflicts.
- Feat/refactor test suite (#930): Improved the structure and maintainability of the testing suite.
- Feat/add import sorting (#931): Added import sorting to improve code readability and consistency.
- Allow overriding dynamic pipeline options + resolver to load settings in catalog (#901): Enhanced flexibility by allowing the overriding of dynamic pipeline options and loading settings from the catalog.
- Modelling cleanup - unify splits (#907): Improved the modelling process by unifying split generation, improving code clarity and consistency.
- Only allow one model per modelling pipeline (#924): Improved consistency by ensuring only one model is processed per modelling pipeline.
- Disable SSL for local Neo4J (#972): Disabled SSL for local Neo4j deployments to streamline local development.
- Add Integration Tests for Generating the Changelog (#968): This improved the reliability of the changelog generation by adding integration tests.
- Check if branch starts with 'release' when triggering a data release (#921): This improved release management by ensuring that the branch starts with 'release' when triggering a data release.
- Implement new NodeNormalizer with multiple calls to the API: (#972) Improved data cleaning and normalization with NodeNormalizer.
- Refactor normalization logic within integration: (#972) Improved code organization and modularity in the integration pipeline.
- Refactor code to run with latest Kedro version: (#972) Improved compatibility with the latest Kedro version.
- Refactor release logic: (#972) Improved code organization and modularity around release.


## Documentation ✏️

- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Onboarding fixes (#902): Improved onboarding documentation and materials.
- Update common_errors.md (#925): Updated the common errors document with solutions for new problems.
- Add libomp library to installation documentation (#934): Updated installation instructions to include the `libomp` library.
- Update virtual environment onboarding documentation (#906): Updated the virtual environment setup in the onboarding instructions.
- Update docs on how to create a release (#940): Updated the documentation on how to create a release.
- Fix minor typos in Kedro Extensions' documentation (#913): Corrected minor typos in the Kedro Extensions documentation.
- Update index.md (#956): Updated the project's index file to improve readability and provide additional information.
- Upgrade the required java version 11 > 17 in the docs and in the docker image (#903): Updated the required Java version to 17, resolving potential compatibility issues.
- Add key Jacques (#883): Improved onboarding process and added key for Jacques.
- Add key Matej (#886): Improved onboarding process and added key for Matej.
- Add git-crypt key for Kushal (#904): Improved onboarding process and added key for Kushal.
- Add git crypt public key for MariaHei (#944): Added a public key for MariaHei to improve collaboration.
- Simplify Neo4J SSL setup (#878): Updated the documentation on how to configure Neo4J SSL.
- Add kedro resource documentation (#919): Added documentation for Kedro resources.
- Add parameter for ensemble model aggregation function (#905): Improved documentation for how to add an ensemble function.
- Have matrix CLI include mkdocs frontmatter for release notes and articles (#959): This enhanced the release notes and articles by including mkdocs frontmatter.


## Newly onboarded colleagues 🚤

- Add key Jacques (#883): Onboarded Jacques and added his key to the git-crypt configuration.
- Add key Matej (#886): Onboarded Matej and added his key to the git-crypt configuration.
- Add git-crypt key for Kushal (#904): Onboarded Kushal and added his key to the git-crypt configuration.
- Add git crypt public key for MariaHei (#944): Onboarded MariaHei and added her key to the git-crypt configuration.


## Other Changes

- Feat/render release info docs (#858): Added feature to render release info in the documentation.
- Incrementing spoke version (#914): Updated the SPOKE version in the data sources.
- Have Test DataRelease on Main (#989): Simplified testing by adding a test data release workflow to the main branch.
- Only allow one model per modelling pipeline (#924): Improved pipeline consistency.
- Feat/expand git checks for kedro submit (#961): Enhances Git checks before data release submission.

