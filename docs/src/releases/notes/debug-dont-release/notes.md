---
draft: false
date: 2025-02-11
categories:
  - Release
authors:
  - eKathleenCarter
  - lvijnck
  - Siyan-Luo
  - emil-k
  - JacquesVergine
  - alexeistepa
  - pascalwhoop
  - piotrkan
  - oliverw1
  - amyford
---
## Breaking Changes 🛠

No breaking changes in this release.


## Exciting New Features 🎉

- Added a new CLI command to add a single user to multiple teams (`matrix gh-users add`).
- Added scheduled sampling pipeline (`scheduled-sampling-pipeline.yml`). This pipeline runs daily at 5am GMT to collect a sample of the data.


## Experiments 🧪

- Added a new experiment to test the use of a scheduled pipeline to collect a sample of the data.


## Bugfixes 🐛

- Fixed an issue where the clinical trial preprocessing nodes were not working correctly.
- Fixed an issue where the normalizer always returned `normalization_success=True`.
- Fixed an issue where the Argo template was not being linted correctly.
- Fixed a deadlock issue in subprocess calls.
- Fixed an issue in the release notes generation, where the scope of information used was incorrect.
- Fixed an issue where the MLflow metric tracking was not working correctly.
- Fixed an issue where the EC medical nodes were not being processed correctly in the preprocessing pipeline.
- Fixed a bug in the GitHub action runner that caused tokens in JSON blobs to be misinterpreted.
- Fixed a bug where the BigQuery table was not correctly updated if it existed.
- Fixed a missing Makefile target.
- Fixed a broken docs link.
- Fixed a schema check in the preprocessing pipeline.
- Fixed a version issue for GT in the ingestion catalog.
- Fixed an error in the integration pipeline due to missing interpolation key.


## Technical Enhancements 🧰

- Improved the CLI for quickly adding users to multiple teams.
- Improved the efficiency of the data ingestion process by utilizing BigQuery.
- Refactored code to improve readability and maintainability.
- Changed the way release notes are generated, and improved the logic for determining whether to generate notes and articles.
- Improved the logic for determining the latest minor release.
- Improved the robustness of subprocess calls by handling deadlocks more efficiently.
- Improved the docker images generation pipeline.
- Improved the way Argo workflows are handled locally.
- Replaced `git-crypt` with a script to handle secrets in the repository.  This is only relevant for system admins, most people now use Secret Manager.
- Changed the default resources for Argo workflows and ArgoNodes.


## Documentation ✏️

- Updated the onboarding documentation to include container registry authentication.
- Added a FAQ section for common errors.
- Added a new runbook for generating notes and articles by AI.
- Added a runbook for setting up Argo workflows locally.
- Added documentation for the Public Data Zone infrastructure.
- Added information about DNS configuration.
- Updated the `git-crypt` documentation to reflect the move to Secret Manager.
- Updated documentation for the Compute Cluster. This is now renamed to Kubernetes Cluster and the content has changed slightly.
- Added documentation for the observability stack.



## Newly onboarded colleagues 🚤

No new colleagues onboarded in this release.


## Other Changes

- Updated dependencies.
- Added unit tests.
- Improved error handling.
- Added several new configuration files to separate concerns.
- Added several new pipelines to handle the different data sources, now supporting a more modular pipeline structure.
- Added several new datasets to allow more flexible data management in pipelines.
- Added new functionality to resolve names to curies for source and target columns in clinical trials data.
- Updated the way data release pipeline handles semmed filtering.


