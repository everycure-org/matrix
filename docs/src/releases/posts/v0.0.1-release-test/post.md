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

# `v0.2.7`:  Enhanced Release Automation, K-Fold Cross-Validation, and Flexible Ensemble Aggregation

This release of the Matrix Platform introduces significant improvements to release automation, model evaluation, and pipeline flexibility. Key enhancements include automated periodic release submissions, k-fold cross-validation for robust model evaluation, and a flexible aggregation function for ensemble models.

<!-- more -->

## Automated Release Management and Pipeline Enhancements

* **Automated Release Runs (#877):** A scheduled GitHub Action now periodically submits the Kedro pipeline, automating the release process and reducing manual effort.  This ensures regular updates and facilitates continuous integration and delivery.
* **Intermediate Release Support (#957):**  The release workflow now supports intermediate releases, providing greater flexibility in release management and enabling more granular control over the release process.  This allows for testing and validation of releases before they are marked as "latest."
* **Dynamic Pipeline Options and Resolver (#901):**  Pipeline options can now be overridden using environment variables, enhancing flexibility and enabling customization of pipeline execution. The introduction of a resolver function allows loading settings from the catalog, improving configuration manageability.


## Robust Model Evaluation and Flexible Ensembling

* **K-fold Cross-Validation (#683):** Model evaluation robustness is significantly improved through the implementation of k-fold cross-validation.  This generates multiple train-test splits, enabling a more comprehensive assessment of model performance and generalization ability.
* **Ensemble Model Aggregation Function (#905):**  The addition of a parameter for the ensemble model aggregation function allows users to specify flexible aggregation methods. This enhances model performance and customization by enabling different strategies for combining predictions from individual models.


## Infrastructure and Tooling Improvements

* **Matrix CLI Enhancements (#959):** The Matrix CLI now includes mkdocs frontmatter for release notes and articles, streamlining documentation integration and enhancing workflow efficiency.
* **`create-post-pr-release` Workflow Update:** The release creation workflow has been updated to dynamically determine the release version from the PR branch name (using the `release/*` pattern). This automation ensures accurate release tagging and simplifies the release process. The workflow now also determines whether the release should be tagged as "latest."
* **`matrix-cli` Dependencies Update:** The `matrix-cli` now utilizes Jinja2 for templating, PyYAML for YAML interaction, questionary for user prompts, tabulate for tabular output, tenacity for retry logic, and Typer for CLI construction. These additions enhance the CLI's functionality and improve user experience.
* **Modelling Catalog Restructure:** The modelling catalog has been restructured to incorporate fold information in filepaths, allowing for separation of data by fold and shard.  A new "combined predictions" dataset facilitates analysis across folds, supporting cross-validation and ensemble methods.  The removal of model-specific paths for some datasets streamlines the pipeline.
* **Modelling Pipeline Restructure:**  The modelling pipeline now incorporates folds directly into its structure, enabling parallel processing of different data splits. The `_create_model_pipeline` function dynamically creates pipelines for each fold.  The pipeline now also combines predictions across folds, suggesting an ensemble approach or aggregated analysis.
* **Evaluation Catalog Updates:** The evaluation catalog now includes fold-specific filepaths for pairs and results, enabling separate evaluation data for each fold.  Aggregated and reduced aggregated results are also included, along with a master report, for a more comprehensive evaluation analysis.

## Bug Fixes and Stability Enhancements

Several bug fixes address critical issues and enhance platform stability:

* **`wipe_neo` Script Protocol Fix (#899):**  Corrected the protocol used in the `wipe_neo` script to improve Neo4j interaction.
* **Kedro Hooks Test Coverage (#900):** Improved test coverage for Kedro hooks enhances release stability.
* **Schema Error Fix in `ingest_nodes` (#943):**  Resolved a schema error related to null values in the `name` column of the `ingest_nodes` function.
* **Release Branching Fix (#950):** Addressed an issue with release branches being created from the incorrect commit.
* **Git SHA Retrieval Fix (#974):** Corrected the command used to retrieve the Git SHA for accurate version tracking.
* **Neo4j Resource Allocation Fix (#977):** Ensured the main container in the Neo4j template receives appropriate resources.
* **GitHub Token Access Fix (#990):**  Resolved an issue with GitHub Actions not accessing the required token for creating releases.
* **Jinja Template Placeholder Fix (#986):**  Corrected a naming discrepancy in Jinja template placeholders.
* **Empty Parameter File Handling (#968):** Added validation to handle empty parameter files, preventing pipeline errors.


## Documentation and Onboarding Improvements

Several documentation updates enhance user experience:

* **Kedro Resource Documentation (#919):** Added documentation on Kedro resources.
* **Onboarding Fixes (#902):** Improved onboarding documentation.
* **Common Errors Documentation Update (#925):** Updated the `common_errors.md` document.
* **Java Version Update in Documentation (#903):**  Updated the required Java version to 17 in the documentation.
* **`libomp` Library Added to Installation Documentation (#934):** Included the `libomp` library in the installation instructions.
* **Release Creation Documentation Update (#940):** Updated the documentation on how to create a release.
* **Kedro Extensions Documentation Typos Fixed (#913):**  Corrected typos in the Kedro Extensions documentation.
* **Virtual Environment Documentation Update (#906):** Updated the virtual environment setup instructions.
* **Neo4J SSL Setup Documentation Update (#878):**  Updated documentation for Neo4J SSL setup.
* **Confusing Flag Column Name Renamed (#893):** Renamed a confusing flag column name for clarity.
* **Main Onboarding Page Update (#956):** Updated the `index.md` onboarding page.
* **GPG Key Sharing via PR in Onboarding (#902, #944, #904):**  Updated onboarding documentation to instruct new users to share their GPG public key via a Pull Request (PR) for improved security.


## Security Enhancements

* **Onboarding GPG Key Submission via PR:** New users now submit their GPG public keys through Pull Requests, improving security practices and enhancing control over key management.


This release improves the stability, automation, and flexibility of the Matrix Platform. The automated release process, combined with k-fold cross-validation and flexible ensemble aggregation, enables more robust model evaluation and streamlined workflows. The numerous bug fixes, documentation improvements, and infrastructure enhancements contribute to a more reliable and developer-friendly platform.
