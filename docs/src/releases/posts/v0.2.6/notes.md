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
## Breaking Changes 🛠

None.


## Exciting New Features 🎉

None.


## Experiments 🧪

None.


## Bugfixes 🐛

- Fixed an issue where the docker build was failing in GH Actions (#1001).
- Fixed the sort order of releases in the Release History documentation (#1002).


## Technical Enhancements 🧰

- Updated `aiobotocore` to version 2.18.0, `alembic` to 1.14.1, `botocore` to 1.36.1, `cloudpickle` to 3.1.1, `databricks-sdk` to 0.41.0, `debugpy` to 1.8.12, `google-api-python-client` to 2.159.0, `more-itertools` to 10.6.0, `openai` to 1.59.9, `orjson` to 3.10.15, `pydantic` to 2.10.5, `prompt-toolkit` to 3.0.50, `rdflib` to 7.1.3, `referencing` to 0.36.1, `scipy` to 1.15.1, `strawberry-graphql` to 0.258.0, `transformers` to 4.48.1, `uvloop` to 0.21.0, `watchfiles` to 1.0.4, `websockets` to 14.2, and `wrapt` to 1.17.2 (#982).
- Refactored the integration pipeline to use a more modular approach with transformers. (#982)
- Improved the `biolink_deduplicate_edges` function for more efficient edge deduplication. (#982)
- Updated the Kedro pipeline submission workflow's cron schedule (#1000).
- Removed the `biolink.predicates` dataset from the integration catalog (#982).
- Increased the number of rows generated for several datasets in the `parameters.yml` file. (#982).
- Improved the handling of topological embeddings. (#982)
- Changed the `RTXTransformer` to only handle single semmed edges. (#982)
- Improved the `IngestedNodesSchema` and other schema definitions for better type handling. (#982)
- Refactored the data release pipeline for better modularity and readability. (#982)
- Modified the way array columns are joined in the data release pipeline. (#982)
- Improved error handling in the data release pipeline. (#982)
- Changed the way the `RTXTransformer` handles SemMedDB edges. (#982)
- Removed the dependency on Pandera's schema validation, and implemented an internal class-agnostic schema validation module for more flexibility and better compatibility. (#938)
- Created abstract `Normalizer` class.  (#982)
- Modified the `NCATSNodeNormalizer` to use the abstract `Normalizer` class. (#982)
- Added `NopNormalizer` for no-op normalization. (#982)


## Documentation ✏️

- Updated release notes in `releases_aggregated.yaml` (#1002)


## Newly onboarded colleagues 🚤

None.


## Other Changes

- Renamed the Kedro pipeline submission workflow (#1000).
- Improved logging in several parts of the code. (#982)
- Added numerous unit tests. (#982)
- Added more robust error handling (#982)
- Renamed several files and folders to improve project organization (#982).
