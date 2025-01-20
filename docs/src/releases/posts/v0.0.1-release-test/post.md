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
# `v0.2.7`: Robust Modelling with K-Fold Cross-Validation and Enhanced Pipeline Flexibility

This release significantly enhances the Matrix Platform's modelling capabilities and pipeline flexibility.  The introduction of k-fold cross-validation strengthens model evaluation, while improvements to ensemble model aggregation and support for intermediate releases improve customization and control over the release process.  Furthermore, various bug fixes, technical enhancements, and documentation updates contribute to a more stable and user-friendly platform.

<!-- more -->

## Enhanced Modelling and Evaluation

This release introduces major improvements to the modelling pipeline:

* **K-Fold Cross-Validation (#683):**  Model evaluation robustness is significantly improved with the implementation of k-fold cross-validation.  Generating multiple train-test splits provides a more comprehensive assessment of model performance and generalization ability, leading to more reliable model selection and comparisons.  The modelling catalog now reflects this change, incorporating "folds" into the filepath structure.

* **Flexible Ensemble Model Aggregation (#905):**  Users can now specify the aggregation function for ensemble models, providing greater control over how individual model predictions are combined.  This enhances model performance and customization.  Documentation has been updated to reflect this new functionality.

## Pipeline and Infrastructure Enhancements

Several enhancements streamline the pipelines and improve infrastructure:

* **Intermediate Non-Latest Release Support (#957):**  The platform now supports creating non-latest releases, providing greater flexibility in the release management process.

* **Removal of `refit` Library (#811):**  Removing the `refit` library streamlines the codebase and simplifies dependencies, reducing maintenance overhead.

* **Simplified Neo4j SSL Setup (#878):**  Neo4j SSL configuration is now simpler and more secure, improving the setup process for developers and users.  The documentation has been updated accordingly.

* **Updated Neo4j Connection String (#880):**  Updating the Neo4j connection string in templates improves consistency and clarity.

* **`ArgoNode` Class Introduction (#885):**  Refactoring the `argo_node` function into the `ArgoNode` class enhances code structure and organization.

* **Consistent `pyspark.sql` Usage (#923):**  Consistent usage of `pyspark.sql` improves code clarity and maintainability.

* **`object` Keyword Renaming (#922):**  Renaming the `object` keyword improves readability and prevents potential conflicts.

* **Modelling Pipeline Refactoring for K-Fold Cross-Validation:** The modelling pipeline has been significantly refactored to accommodate k-fold cross-validation, including changes to data structures and core functions like `create_model_input_nodes`, `apply_transformers`, and `train_model`.  This refactoring allows for parallel training and tuning of models for different folds and shards.

* **Evaluation Pipeline Update for K-Fold Cross-Validation:**  The evaluation pipeline now calculates and aggregates metrics for each fold, ensuring accurate performance assessment with cross-validation.

## Bug Fixes and Stability Improvements

Several bug fixes enhance the platform's stability and reliability:

* **`wipe_neo` Script Protocol Fix (#899):**  Correcting the protocol improves Neo4j interaction.

* **Kedro Hooks Test Coverage (#900):**  Improved test coverage for Kedro hooks enhances stability and reliability.

* **`ingest_nodes` Schema Error Fix (#943):**  Handling null values in the `name` column resolves a schema error.

* **Release Branching Fix (#950):**  Ensuring release branches are created from the correct commit improves consistency in release management.

* **CI Check Fixes (#967):**  Resolving CI check failures on the `main` branch improves development workflow.

## Documentation and Onboarding Enhancements

Several documentation updates improve onboarding and user experience:

* **Updated Common Errors Documentation (#925):**  Adding solutions for new problems helps users troubleshoot common issues.

* **Kedro Resource Documentation (#919):**  New documentation explains Kedro resources.

* **Onboarding Documentation Updates (#902, #906, #956):**  Improvements to the onboarding documentation, including virtual environment setup, streamline the onboarding process.

* **Updated Java Version Requirement (#903):**  Upgrading the required Java version to 17 ensures compatibility.

* **Added `libomp` Library to Installation Documentation (#934):**  Including the `libomp` library ensures all dependencies are installed.

* **Updated Release Creation Documentation (#940):**  Clearer instructions simplify release creation.

* **Typo Fixes in Kedro Extensions Documentation (#913):**  Correcting typos improves clarity.

* **Updated Onboarding Guide for GPG Key Sharing:** The onboarding guide now instructs new users to share GPG keys via Pull Requests for improved security and key management.  The git-crypt documentation has been updated accordingly.

* **Automated Mkdocs Frontmatter for Release Notes (#959):**  Adding mkdocs frontmatter automatically to release notes and articles improves structure and consistency.


## Other Notable Changes

* **Matrix CLI and Release Automation:**  The `matrix-cli` now leverages Jinja2 templates for improved prompt generation and release note management, and uses the `date` module for timestamps.  The trigger for the "Create Pull Request to Verify AI Summary of Release Notes" workflow has been updated to `debug/release*`.

* **Miscellaneous Technical Enhancements:**  These include refactoring the test suite (#930), adding import sorting (#931), adding release trigger labels to Argo (#936), unifying splits in modelling (#907), allowing dynamic pipeline option overrides (#901), expanding git checks for Kedro submit (#961), limiting one model per modelling pipeline (#924), disabling SSL for local Neo4j (#972), saving 'not treat' and 'unknown' scores (#853), renaming a confusing flag column (#893), incrementing the spoke version (#914), checking branch names for release triggers (#921), and fixing the git SHA retrieval command (#974).

* **New Team Members:**  Several new colleagues have been onboarded, including Jacques (#883), Matej (#886), Kushal (#904), and MariaHei (#944).

This release marks a significant step forward in enhancing the Matrix Platform's modelling capabilities, pipeline flexibility, and overall user experience. The implementation of k-fold cross-validation, coupled with numerous technical improvements, bug fixes, and documentation updates, contributes to a more robust, reliable, and user-friendly platform for drug repurposing research.
