---
draft: false
date: 2025-02-04
categories:
  - Release
authors:
  - piotrkan
  - lvijnck
  - alexeistepa
  - emil-k
  - JacquesVergine
  - eKathleenCarter
---
## Breaking Changes 🛠

No breaking changes were introduced in this release.


## Exciting New Features 🎉

- Added a new `GitHubReleaseCSVDataset` to simplify ingestion of data from GitHub releases (PR 1050).
- Implemented MLFlow logging of datasets used in the pipeline (PR 1048).


## Experiments 🧪

No experiments were explicitly documented in these PRs.


## Bugfixes 🐛

- Fixed an issue in clinical trial preprocessing nodes (PR 1039).
- Corrected the normalizer to accurately reflect `normalization_success` status (PR 1060).
- Fixed MLflow metric tracking (PR 1075).


## Technical Enhancements 🧰

- Removed hardcoded SILC configuration (PR 973).
- Improved the efficiency and robustness of Robokop data transformation (PR 1032).
- Updated resource defaults in `kedro4argo_node.py` (from GiB to Gb) (PR 1032).
- Various internal code simplifications and improvements across multiple files.


## Documentation ✏️

No specific documentation changes were noted in these PRs.


## Newly onboarded colleagues 🚤

No onboarding-related PRs were included in this set.


## Other Changes

- Added a new `new-eKathleenCarter.asc` (PR 1032) (Assumed to be an author's GPG key).
- Various configuration updates in `parameters.yml`, `globals.yml`, `catalog.yml` files, primarily affecting data sources and pipeline configurations.

