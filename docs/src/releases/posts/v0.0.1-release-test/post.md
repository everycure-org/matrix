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
# Matrix Platform `v0.2.7`:  K-fold Cross-Validation, Flexible Ensemble Aggregation, and Pipeline Enhancements

This release of the Matrix Platform introduces significant improvements to model evaluation, ensemble flexibility, and pipeline efficiency.  Key highlights include k-fold cross-validation for robust model assessment, flexible ensemble aggregation functions for customized model performance, and substantial technical enhancements and refactoring for improved code maintainability.

<!-- more -->

## Enhanced Model Evaluation and Flexibility

* **K-fold Cross-Validation (#683):**  Robustness in model evaluation is significantly enhanced with the implementation of k-fold cross-validation. This generates multiple train-test splits, offering a more comprehensive evaluation of model performance and generalization ability. This feature facilitates more reliable model comparisons and configuration choices.  The evaluation catalog now incorporates fold information (`fold_{fold}`) directly into the dataset paths, and results are stored per fold in a dedicated `reporting` layer.

* **Flexible Ensemble Aggregation (#905):**  A new parameter, `evaluation.reported_aggregations`, allows users to specify the aggregation functions used for ensemble models. This provides greater control over how individual model predictions are combined, enabling improved performance and customization tailored to specific use cases.  Default aggregation functions include `numpy.mean`, `numpy.std`, `numpy.median`, `numpy.min`, and `numpy.max`.

## Pipeline and Infrastructure Improvements

* **Intermediate Releases (#957):** This release introduces the ability to publish non-latest releases on GitHub, enabling more granular version control and facilitating comparisons between different release versions.  This is reflected in changes to the `create-post-pr-release.yml` workflow, which now determines the intended release version from the pull request branch name and handles the `--latest` flag for GitHub releases based on version ordering.

* **Batch Pipeline (#766):** A new pipeline for batch processing of data has been introduced, further enhancing the platform's ability to handle large datasets efficiently.

* **Modelling Pipeline Refactoring:** The modelling pipeline now includes fold information in the dataset paths, aligning with the k-fold cross-validation changes.  Data transformations and model training are now performed per fold, and sharding is decoupled from model and fold.  Additional parameters for drug and disease types were added to the `prefilter_nodes` function for more specific node filtering.

* **Evaluation Pipeline Refactoring:** The evaluation pipeline has been restructured to generate pipelines for individual folds and models using the new `_create_evaluation_fold_pipeline` function.  The `consolidate_evaluation_reports` function now accepts keyword arguments, and a new function, `reduce_aggregated_results`, simplifies aggregated results for MLFlow reporting.

## Technical Enhancements and Refactoring

* **Removal of `refit` Library (#811):**  Removing the `refit` library streamlines the codebase and simplifies dependency management, making maintenance more straightforward.

* **Simplified Neo4j SSL Setup (#878):**  The Neo4j SSL configuration has been simplified, making setup easier and improving security.  Documentation for this setup has also been updated.

* **`argo_node` Refactoring (#885):**  The `argo_node` function has been refactored into the `ArgoNode` class, enhancing code structure and organization.

* **Consistent `pyspark.sql` Usage (#923):** Standardizing the use of `pyspark.sql` improves code consistency and readability.

* **Refactored Test Suite (#930):** The test suite has been refactored for improved structure and maintainability, making it easier to ensure code quality.

* **Import Sorting (#931):**  Consistent import sorting further improves code readability and maintainability.

* **Release Trigger Label to Argo (#936):** Argo workflows now include labels to indicate data release triggers, improving monitoring and tracking.

* **Modelling Cleanup - Unify Splits (#907):** The modelling process has been streamlined by unifying split generation.

* **Overriding Dynamic Pipeline Options (#901):** The platform now allows overriding dynamic pipeline options and loading settings in the catalog, increasing flexibility.

* **Improved Git Checks for Kedro Submit (#961):**  Additional checks ensure the git repository is in a clean state before submitting a Kedro pipeline, preventing unintended changes from being included in pipeline runs.

* **Matrix CLI Frontmatter for Release Notes/Articles (#959):** The Matrix CLI now includes mkdocs frontmatter for release notes and articles, simplifying documentation generation.

* **Incrementing Spoke Version (#914):** The Spoke version has been updated.

* **Disabling SSL for Local Neo4J (#972):**  Disabling SSL for local Neo4j development simplifies local setup and testing.

* **Correct Git SHA Command (#974):**  A fix to the command for retrieving the git SHA improves reliability.


## Bug Fixes

* **Fix protocol for `wipe_neo` script (#899):**  The protocol used in the `wipe_neo` script has been corrected for improved Neo4j interaction.

* **Improved Kedro Hooks Test Coverage (#900):** Increased test coverage for Kedro hooks enhances release stability and reliability.

* **Schema Error Fix in `ingest_nodes` (#943):** A schema error in the `ingest_nodes` function, caused by null values in the `name` column, has been resolved.

* **Release Branching Fix (#950):** An issue where release branches were created from the incorrect commit has been addressed, ensuring consistency in release management.


## Documentation and Onboarding

* **Updated Common Errors (#925):**  The documentation for common errors has been updated with solutions for newly encountered problems.

* **Added Kedro Resource Documentation (#919):** Documentation for Kedro resources has been added.

* **Onboarding Improvements (#902, #906, #883, #886, #904, #892, #886):**  Several improvements to onboarding documentation and materials, including Git Crypt setup, have been implemented.  A key change is that new users now share their GPG key via a Pull Request (PR) instead of directly in the onboarding issue, improving security and streamlining key management.

* **Updated Java Version Documentation (#903):** The required Java version has been updated to 17.

* **Added `libomp` Library to Installation Documentation (#934):**  Installation instructions now include the `libomp` library.

* **Updated Release Creation Documentation (#940):**  The documentation on creating a release has been updated.

* **Minor Typos in Kedro Extensions' Documentation (#913):** Minor typos in the Kedro Extensions documentation have been corrected.

* **Simplified Neo4J SSL Setup Documentation (#878):**  Documentation for Neo4J SSL setup has been simplified.

## New Contributors

* **Jacques (#883):** Onboarded and added key to git-crypt.
* **Matej (#886):** Onboarded and added key to git-crypt.
* **Kushal (#904):** Onboarded and added key to git-crypt.
* **MariaHei (#944):** Onboarded and added key to git-crypt.


This release marks a significant step forward in the platform's capabilities, with enhanced model evaluation, flexible ensemble aggregation, and numerous technical improvements.  The focus on robust model assessment, customizable performance, and improved code maintainability strengthens the foundation for future development and wider adoption.
