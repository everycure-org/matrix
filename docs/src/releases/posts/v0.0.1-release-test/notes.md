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
  - jdr0887
  - alexeistepa
  - matwasilewski
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

- **k-fold Cross-Validation (#683):** Improves model evaluation robustness.
- **Flexible Ensemble Aggregation (#905):** Allows flexible aggregation methods for ensemble models, enhancing model performance and customization.
- **Intermediate Releases (#957):** Allows for non-latest releases to be published on GitHub.


## Experiments 🧪

No new experiments were reported in this release.

## Bugfixes 🐛

- **Fix protocol for `wipe_neo` script (#899):** Corrected the protocol used in the `wipe_neo` script for improved Neo4j interaction.
- **Improved Kedro Hooks Test Coverage (#900):** Improved test coverage for Kedro hooks, enhancing release stability and reliability.
- **Schema Error Fix in `ingest_nodes` (#943):** Resolved a schema error in the `ingest_nodes` function by handling null values in the `name` column.
- **Release Branching Fix (#950):** Addresses an issue where release branches were created from the wrong commit.


## Technical Enhancements 🧰

- **Removal of `refit` Library (#811):** Streamlines the codebase by removing the `refit` library.
- **Simplified Neo4j SSL Setup (#878):** Improves Neo4j SSL configuration.
- **`argo_node` Refactoring (#885):** Improves code structure by refactoring the `argo_node` function into the ArgoNode class.
- **Consistent `pyspark.sql` Usage (#923):** Improves code consistency and clarity by using `pyspark.sql` consistently.
- **Refactored Test Suite (#930):** Improved the structure and maintainability of the testing suite.
- **Import Sorting (#931):** Added import sorting to improve code readability and consistency.
- **Release Trigger Label to Argo (#936):** Enhanced Argo workflows by adding labels to indicate data release triggers.
- **Modelling Cleanup - Unify Splits (#907):** Improved the modelling process by unifying split generation.
- **Batch Pipeline (#766):** Introduced a new pipeline for batch processing of data.


## Documentation ✏️

- **Updated Common Errors (#925):** Added solutions for new problems.
- **Added Kedro Resource Documentation (#919):** Added documentation for Kedro resources.
- **Onboarding Improvements (#902, #906, #883, #886, #904, #892, #886):** Improved onboarding documentation and materials including Git Crypt setup.
- **Updated Java Version Documentation (#903):** Updated the required Java version to 17.
- **Added `libomp` Library to Installation Documentation (#934):** Updated installation instructions to include the `libomp` library.
- **Updated Release Creation Documentation (#940):** Updated the documentation on how to create a release.
- **Minor Typos in Kedro Extensions' Documentation (#913):** Corrected minor typos in Kedro Extensions documentation.
- **Simplified Neo4J SSL Setup Documentation (#878):** Updated the documentation on how to configure Neo4J SSL.


## Newly onboarded colleagues 🚤

- **Jacques (#883):** Onboarded and added key to git-crypt.
- **Matej (#886):** Onboarded and added key to git-crypt.
- **Kushal (#904):** Onboarded and added key to git-crypt.
- **MariaHei (#944):** Onboarded and added key to git-crypt.


## Other Changes

- **Rendering Release Info in Docs (#858):** Added feature to render release info in the documentation.
- **Overriding Dynamic Pipeline Options (#901):** Allows overriding dynamic pipeline options and loading settings in the catalog.
- **Improved Git Checks for Kedro Submit (#961):** Added more checks to ensure the git repository is in a clean state before submitting a kedro pipeline.
- **Matrix CLI Frontmatter for Release Notes/Articles (#959):** Enabled the Matrix CLI to include mkdocs frontmatter for release notes and articles.
- **Incrementing Spoke Version (#914):** Updated the Spoke version.
- **Disabling SSL for Local Neo4J (#972):** Disabled SSL for local Neo4J to simplify local development.
- **Correct Git SHA Command (#974):** Fixed the command to get the git SHA, improving reliability.

