---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - JacquesVergine
  - emil-k
  - lvijnck
  - marcello-deluca
  - alexeistepa
  - jdr0887
  - matwasilewski
  - Siyan-Luo
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

- **K-fold Cross-Validation (#683):** Improves model evaluation robustness.
- **Flexible Ensemble Aggregation (#905):** Allows flexible aggregation methods for ensemble models.
- **Periodic Kedro Pipeline Submission (#877):** Automates data release runs.


## Experiments 🧪

No new experiments were reported in this release.


## Bugfixes 🐛

- **`wipe_neo` Script Protocol Fix (#899):** Corrected the protocol used in the `wipe_neo` script.
- **Improved Kedro Hooks Test Coverage (#900):** Enhanced test coverage for Kedro hooks.
- **Schema Error Fix in `ingest_nodes` (#943):** Resolved a schema error in the `ingest_nodes` function.
- **Release Branching Fix (#950):** Addresses an issue where release branches were created from the wrong commit.
- **CI Fixes (#967):** Makes CI checks pass on main again.
- **GH Action Fix (#990):** Adds missing access to GH token in workflow.
- **Git SHA Fix (#974):** Uses the correct command to get the git sha.


## Technical Enhancements 🧰

- **Removal of `refit` Library (#811):** Streamlined the codebase by removing the `refit` library.
- **Simplified Neo4j SSL Setup (#878):** Improved Neo4j SSL configuration.
- **`argo_node` Refactoring (#885):** Improved code structure by refactoring the `argo_node` function.
- **Consistent `pyspark.sql` Usage (#923):** Improved code consistency by using `pyspark.sql` consistently.
- **Refactored Test Suite (#930):** Improved the structure and maintainability of the testing suite.
- **Added Import Sorting (#931):** Improved code readability and consistency.
- **Added Release Trigger Label to Argo (#936):** Added labels to indicate data release triggers.
- **Improved Modelling process (#907):** Improved modelling process by unifying split generation.
- **Removal of Batch-ing in inference pipeline (#909):**  Simplifies the inference pipeline.
- **Improved Handling of Intermediate Releases (#957):** Allows for the creation of intermediate non-latest releases.
- **Resource allocation in Neo4j template (#977):** Allocates resources to the main container in Neo4j template.
- **Removal of unnecessary reliance on the normalizer's outputs in the preprocessing step (#766):** Removed the reliance on the normalizer's output.
- **Introducing the `inject` decorator (#901):**  Simplifies the code and makes the code more modular.
- **Separate nodes for each source (#901):**  Makes the code more modular.
- **Unification of splits generation (#907):**  Ensures all splits have the same schema.
- **Only one model per modelling pipeline (#924):**  Simplifies the modelling pipeline by enforcing only one model per pipeline.
- **Removal of schema validation in the evaluation pipeline (#901):** Improves performance and reduces complexity of the evaluation step.
- **Using only the full dataset for training (#901):** Improves performance and training stability.
- **Use of PySpark for all data manipulation (#901):** Improves performance and ensures consistency.


## Documentation ✏️

- **Improved Onboarding Materials (#902, #883, #886, #904):** Updated onboarding documentation.
- **Updated Java Version Documentation (#903):** Updated the required Java version in the documentation.
- **Updated `libomp` Library Installation Instructions (#934):** Added instructions for installing the `libomp` library.
- **Updated Release Creation Documentation (#940):** Updated the documentation for creating a release.
- **Minor Typos Fixed in Kedro Extensions Documentation (#913):** Corrected typos in the Kedro Extensions documentation.
- **Updated Virtual Environment Setup Documentation (#906):** Updated the virtual environment setup instructions.
- **Added Documentation for Kedro Resources (#919):** Added documentation for Kedro resources.
- **Updated Neo4j SSL Configuration Documentation (#878):** Improved Neo4j SSL configuration documentation.
- **Updated Documentation on Ensemble Aggregation Function (#905):** Added documentation for how to use the new flexible ensemble aggregation function.
- **Improved Common Errors Documentation (#925):** Added solutions to new problems.
- **Update `index.md` (#956):** Fixed a typo in index.md.


## Newly onboarded colleagues 🚤

- **Jacques (#883):** Onboarded and added to the git-crypt configuration.
- **Matej (#886):** Onboarded and added to the git-crypt configuration.
- **Kushal (#904):** Onboarded and added to the git-crypt configuration.
- **MariaHei (#944):** Onboarded and added to the git-crypt configuration.


## Other Changes

- **Rendering Release Info in Docs (#858):** Added a feature to render release info in the documentation.
- **Added Matrix CLI Frontmatter (#959):** Added support for mkdocs frontmatter for release notes and articles in Matrix CLI.
- **Improved Git Checks for Kedro Submit (#961):** Enhanced git checks for the kedro submit command.
- **Added Integration Tests for Changelog Generation (#968):** Added integration tests for generating the changelog.
- **Disable SSL for local Neo4j (#972):** Added configuration to disable SSL for local Neo4j.
- **Improved handling of intermediate releases (#957):** Allows to create intermediate releases without flagging them as latest.
- **Added support for intermediate releases (#951):** Allows to create intermediate releases without flagging them as latest.
- **Expanded Git Checks (#961):** Expands the git checks to include whether the branch starts with 'release'.
- **Improved parameter handling (#901):** Introduced `inject` decorator for cleaner and modular handling of objects.
- **Improved parameter handling (#901):** Added nodes for each source to improve modularity and parallelization.
- **Improved performance (#901):** Removed unnecessary schema validations, and uses only full training dataset.
- **Improved performance (#901):** Uses PySpark for all data manipulation.
- **Updated `index.md` (#956):** Updated `index.md` file with the latest information.

