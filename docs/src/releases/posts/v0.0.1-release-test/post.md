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
# `v0.2.7`: Automated Releases, K-Fold Cross-Validation, and Enhanced Pipeline Flexibility

This release of the Matrix Platform introduces significant automation, enhanced model evaluation capabilities, and increased flexibility in pipeline configuration and execution.  We've automated the periodic submission of the Kedro pipeline, implemented k-fold cross-validation for robust model evaluation, enabled flexible aggregation functions for ensemble models, and introduced support for intermediate releases.  These improvements streamline the release process, enhance model robustness, and provide greater control over pipeline execution.

<!-- more -->

## Automated Release Process

A key improvement in this release is the automation of the Kedro pipeline submission process (#877).  A scheduled GitHub Action now periodically submits the pipeline, significantly reducing manual effort and ensuring regular releases.

## Enhanced Model Evaluation and Flexibility

This release brings major enhancements to model evaluation and customization:

* **K-fold Cross-Validation (#683):** Model evaluation robustness is significantly improved through the implementation of k-fold cross-validation. This generates multiple train-test splits, providing a more comprehensive performance assessment and reducing the risk of overfitting.

* **Flexible Aggregation Function for Ensemble Models (#905):** Users can now specify custom aggregation methods for ensemble models, allowing for fine-grained control over how individual model predictions are combined. This enables greater flexibility in optimizing ensemble performance and tailoring the aggregation strategy to specific requirements.

## Improved Release Management and Pipeline Control

Several enhancements improve release management and offer greater control over the release process:

* **Intermediate Release Support (#957):** The release workflow now accommodates intermediate releases, providing greater flexibility and granularity in release management. This enables more frequent releases and allows for more controlled testing and validation of new features.

* **Allow Overriding Dynamic Pipeline Options and Resolver to Load Settings in Catalog (#901):**  Pipeline flexibility is enhanced by allowing users to override dynamic pipeline options via environment variables.  A new resolver function seamlessly loads settings from the catalog, improving configuration management and simplifying parameterization.

## Bug Fixes and Stability Improvements

Several bug fixes address critical issues and enhance platform stability:

* **`wipe_neo` Script Protocol Fix (#899):** The protocol used in the `wipe_neo` script is corrected, ensuring proper interaction with Neo4j and preventing potential issues.

* **Kedro Hooks Test Coverage Improvement (#900):** Increased test coverage for Kedro hooks enhances release stability and reliability by catching potential issues earlier in the development cycle.

* **Schema Error Fix in `ingest_nodes` (#943):**  Addressing the schema error related to null values in the `name` column of the `ingest_nodes` function ensures data integrity.

* **Release Branching Fix (#950):** Correcting the issue with release branches originating from the wrong commit ensures consistency and prevents potential errors during release creation.

* **Git SHA Retrieval Fix (#974):**  Ensuring the Git SHA is retrieved using the correct command (`git rev-parse HEAD`) guarantees accurate version tracking and release identification.

* **Neo4j Template Resource Allocation Fix (#977):** Allocating appropriate resources to the main container in the Neo4j template improves performance and stability.

* **GitHub Action Token Access Fix (#990):**  Resolving the GitHub Action's token access issue allows the action to create GitHub releases correctly.

* **Jinja Template Placeholder Fix (#986):** Correcting the naming discrepancy in Jinja template placeholders ensures the proper information is passed to the templating process.

* **Improved Handling of Empty Parameter Files (#968):**  Preventing pipeline errors due to empty parameter files ensures smooth pipeline execution.

## Technical Enhancements and Refactoring

This release includes various technical improvements and refactoring efforts:

* **Removal of `refit` Library (#811):** Streamlining the codebase by removing the `refit` library simplifies dependencies and reduces maintenance overhead.

* **Simplified Neo4j SSL Setup (#878):**  Improving the Neo4j SSL configuration simplifies setup and enhances security.

* **Updated Neo4j Connection String in Template (#880):** Updating the Neo4j connection string ensures consistency and clarity across templates.

* **`argo_node` Refactoring to `ArgoNode` Class (#885):** Replacing the `argo_node` function with the `ArgoNode` class enhances code structure, organization, readability, and maintainability.

* **Consistent Use of `pyspark.sql` (#923):** Standardizing the use of `pyspark.sql` throughout the codebase improves code consistency, clarity, and maintainability.

* **Modelling Cleanup - Unify Splits (#907):**  Streamlining the modelling process through unified split generation enhances code clarity and consistency.

* **Refactored Test Suite (#930):**  Improving the structure and maintainability of the test suite makes it easier to ensure code quality and identify potential issues.

* **Added Import Sorting (#931):** Consistent import sorting enhances code readability and maintainability.


## Documentation Improvements

Several updates enhance the documentation and onboarding experience:

* **Onboarding Guide Update (#902):**  Improving the onboarding documentation streamlines the onboarding process for new contributors.

* **Common Errors Documentation Update (#925):**  Adding solutions for recently encountered problems to the common errors documentation helps users troubleshoot issues more effectively.

* **Java Version Documentation Update (#903):** Updating the documentation to reflect the required Java version (17) prevents compatibility issues.

* **`libomp` Library Added to Installation Documentation (#934):** Including the `libomp` library in the installation instructions ensures all dependencies are installed.

* **Release Creation Documentation Update (#940):** Clearer instructions on how to create a release simplify the release process and ensure consistency.

* **Kedro Extensions Documentation Typos Fixed (#913):**  Correcting typos in the Kedro Extensions documentation improves clarity and professionalism.

* **Virtual Environment Onboarding Documentation Update (#906):** Updating the virtual environment setup instructions streamlines the onboarding process.

* **Kedro Resources Documentation (#919):** Adding documentation for Kedro resources assists users in effectively utilizing this feature.

* **Simplified Neo4j SSL Setup Documentation (#878):**  Updating the Neo4j SSL configuration documentation reflects the simplified setup process.

* **Onboarding Documentation Update for GPG Key Sharing (#808, #906):** Updating the onboarding documentation to instruct new users to share their GPG key via a Pull Request improves security practices.  The path for GPG key sharing is also corrected to `.git-crypt/keys/public_keys`.

* **Matrix CLI Documentation Enhancement (#959):**  Documenting the enhanced functionality of the Matrix CLI, including embedding mkdocs frontmatter, provides users with clear instructions on utilizing these features.

## Newly Onboarded Colleagues

We welcome the following new colleagues to the team:

* Jacques (#883)
* Matej (#886)
* Kushal (#904)
* MariaHei (#944)

## Other Changes

* **Rendering Release Information in Documentation (#858):**  Adding a feature to render release information in the documentation improves accessibility for users.

* **Saving `not treat` and `unknown` Scores (#853):** Including columns to store "not treat" and "unknown" scores in the full matrix enhances data completeness and analysis capabilities.

* **Data Release Workflow Test on Main Branch (#989):** Adding a test for the data release workflow on the main branch ensures the workflow is thoroughly tested.

* **Disabling SSL for Local Neo4j (#972):**  Simplifying local development with Neo4j by disabling SSL removes the need for certificate management.

* **Enforcing Single Model per Modelling Pipeline (#924):**  Ensuring only one model is processed within the modelling stage prevents confusion and unexpected behavior.

* **Incrementing Spoke Version (#914):** Updating the SPOKE knowledge graph version ensures consistency.

* **Matrix CLI Mkdocs Frontmatter Integration (#959):**  Adding functionality to the Matrix CLI to generate markdown files with mkdocs frontmatter streamlines documentation integration.

* **Validating Release Branch Names (#921):** Adding validation to ensure that only branches starting with "release" trigger data releases prevents accidental releases from unintended branches.


This release marks a substantial step forward in automating the release process, enhancing model evaluation, and improving pipeline flexibility. The addition of k-fold cross-validation and flexible ensemble aggregation strengthens the platform's analytical capabilities. The numerous bug fixes, technical enhancements, and documentation updates contribute to a more robust, reliable, and user-friendly platform.
