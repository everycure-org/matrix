---
draft: false
date: 2025-01-21
categories:
  - Release
authors:
  - Siyan-Luo
  - piotrkan
  - oliverw1
  - emil-k
  - lvijnck
---
# `v0.2.7`:  Dependency Updates, Pipeline Refinements, and Schema Validation Enhancements

This release of the Matrix Platform focuses on technical improvements and dependency updates, enhancing stability, maintainability, and data processing efficiency.  Key changes include significant dependency updates, refactoring of the integration pipeline, a more flexible schema validation mechanism, and improvements to data handling and logging.

<!-- more -->

## Technical Enhancements and Dependency Updates

This release includes numerous updates to core dependencies and underlying infrastructure:

* **Dependency Updates (#982):**  A comprehensive update of 23 Python packages, including `aiobotocore`, `alembic`, `botocore`, `cloudpickle`, `databricks-sdk`, `debugpy`, `google-api-python-client`, `more-itertools`, `openai`, `orjson`, `pydantic`, `prompt-toolkit`, `rdflib`, `referencing`, `scipy`, `strawberry-graphql`, `transformers`, `uvloop`, `watchfiles`, `websockets`, and `wrapt`. These updates ensure compatibility, improve performance, and address potential security vulnerabilities.

* **Integration Pipeline Refactoring (#982):**  The integration pipeline has been refactored to use a more modular approach with transformers. This improvement enhances code organization, maintainability, and testability. Transformers have been moved to a dedicated directory, and import paths have been updated accordingly. A new abstract `Normalizer` class and a `NopNormalizer` have been introduced, providing a no-operation normalization option and improving flexibility.  The `biolink_deduplicate_edges` function has been simplified, removing the dependency on `biolink_predicates`.

* **Schema Validation Overhaul (#938):**  The dependency on Pandera for schema validation has been removed and replaced with an internal, class-agnostic schema validation module (`matrix.utils.pa_utils`). This new module provides a `check_output` decorator and `DataFrameSchema` data class, offering greater flexibility and improved compatibility.

* **Data Fabricator Adjustments (#982):**  The parameters for the data fabricator in `fabricator/parameters.yml` have been adjusted, modifying the number of rows generated for several datasets.  This change impacts the size and characteristics of fabricated data used in testing and local development.

* **Removal of `biolink.predicates` Dataset (#982):** The `integration.raw.biolink.predicates` dataset has been removed from the integration catalog.

* **Kedro Pipeline Submission Workflow Update (#1000):** The scheduled workflow for Kedro pipeline submission has been renamed to `Periodically run kedro kg-release` and its cron schedule has been adjusted.  Extraneous git commands within the workflow have been removed, streamlining the process.

## Bug Fixes

Two notable bugs were addressed in this release:

* **Docker Build Failure in GitHub Actions (#1001):** A fix resolves a previous issue where the Docker build was failing in GitHub Actions, improving CI/CD reliability.

* **Release Sort Order in Documentation (#1002):** The sort order of releases in the release history documentation has been corrected to display releases in reverse chronological order (newest first).

## Other Improvements

Several other enhancements improve code quality and maintainability:

* **Improved Logging (#982):** Logging has been improved in various parts of the codebase, providing better insights into pipeline execution and facilitating debugging.
* **Enhanced Error Handling (#982):**  More robust error handling mechanisms have been added to improve the stability and resilience of the platform.
* **Increased Unit Test Coverage (#982):** Numerous unit tests have been added to ensure code quality and identify potential issues early.
* **Improved Project Organization (#982):** Several files and folders have been renamed to improve project organization and readability.

## Documentation Updates

The release notes within `releases_aggregated.yaml` have been updated to reflect the changes in this release.

This release contributes to the ongoing evolution of the Matrix Platform, focusing on technical improvements, dependency updates, and enhanced maintainability.  The shift to a more flexible schema validation approach and the refactoring of the integration pipeline demonstrate a commitment to improving code quality and extensibility.  The dependency updates ensure the platform remains current and secure.
