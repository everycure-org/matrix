## Breaking Changes 🛠

No breaking changes were introduced in this release.


## Exciting New Features 🎉

- Implemented k-fold cross-validation in the modelling pipeline.  This allows for more robust model evaluation and reduces the impact of random train-test splits.  The pipeline now generates multiple train-test splits ("folds"), trains a model for each, and aggregates the performance metrics across all folds.  A "full" split (using all data for training) is also created.  The matrix generation and evaluation pipelines now also support k-fold cross-validation.


## Experiments 🧪

- Experiments with matrix transformation and normalization techniques to mitigate biases caused by frequently occurring diseases and drugs.


## Bugfixes 🐛

No specific bug fixes are explicitly mentioned in the PR description or diff.


## Technical Enhancements 🧰

- Refactored the evaluation pipeline to improve efficiency and reporting.  Enhanced reporting now includes accuracy, F1 score, Recall@n, AUROC, Hit@k, and MRR metrics.
- Improved the `changelog_gen.py` script (removed).  Release information is now managed directly in `docs/src/data/releases.yaml`.
- Updated the pipeline configuration files (`catalog.yml` and `parameters.yml`) to accommodate k-fold cross-validation.  This includes changes to dataset paths and parameters to handle multiple folds and aggregated results.
- Added a new hook to extract release information more effectively.


## Documentation ✏️

- Updated the pipeline documentation (`docs/src/pipeline/index.md`) to reflect the changes introduced by k-fold cross-validation.
- Added a release history section (`docs/src/releases/release_history.md`) to display release notes.


## Newly onboarded colleagues 🚤

No onboarding-related changes are mentioned.


## Other Changes

- Modified the GitHub Actions workflow (`create-release-pr.yml`) to handle release versions and generate release notes appropriately.  The branch triggering the workflow was also updated.
- Added `docs/src/data/releases.yaml` to store release information.
