## Breaking Changes 🛠

- No breaking changes were introduced in this PR.


## Exciting New Features 🎉

- Implemented k-fold cross-validation in the modelling pipeline.  This allows for more robust model evaluation and reduces the impact of random train/test splits.  The pipeline now generates multiple train/test splits ("folds"), trains a model for each fold, and aggregates the results for a more reliable performance estimate.


## Experiments 🧪

- Experiments with matrix transformation and normalization techniques to mitigate biases caused by frequently occurring drugs and diseases.


## Bugfixes 🐛

- No bug fixes were explicitly mentioned in this PR.


## Technical Enhancements 🧰

- Refactored the evaluation pipeline to handle k-fold cross-validation results.  This includes changes to the catalog, parameters, nodes, and pipeline definitions to efficiently manage and aggregate metrics across folds.
- Improved the reporting capabilities of the evaluation pipeline, including the addition of more metrics (accuracy, F1 score, Recall@n, AUROC, Hit@k, and MRR) and enhanced reporting formats.
- Added checks to the evaluation pipeline to ensure that the test data does not overlap with the training data.
-  The `changelog_gen.py` script was removed, simplifying the release process. The release information is now directly added to `docs/src/data/releases.yaml`.
- Updated the `ReleaseInfoHooks` to extract relevant information for the release notes.


## Documentation ✏️

- Updated documentation to reflect the changes introduced by k-fold cross-validation, including explanations of the process and the new metrics reported.  Specifically, the pipeline documentation was updated.


## Newly onboarded colleagues 🚤

- No onboarding-related changes were mentioned in this PR.


## Other Changes

- Updated the GitHub workflow (`create-release-pr.yml`) to simplify the release process and to handle the removal of `changelog_gen.py`.
- Modified the data catalog and parameters files (`catalog.yml`, `parameters.yml`) to support k-fold cross-validation.  This involved changes to the filepaths and data structures used to handle multiple folds.

