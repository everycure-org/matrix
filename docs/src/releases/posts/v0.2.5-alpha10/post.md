# `v0.2.6`: Robust Model Evaluation with K-Fold Cross-Validation and Streamlined Release Management

This release of the Matrix Platform introduces k-fold cross-validation into the modeling pipeline, significantly enhancing the robustness of model evaluation.  This release also streamlines the release management process and improves reporting capabilities.

<!-- more -->

## Key Enhancements 🚀

### 1. K-Fold Cross-Validation for Robust Evaluation

The modeling pipeline now incorporates k-fold cross-validation, a technique that partitions the data into multiple "folds" for training and testing. This approach mitigates the impact of random train/test splits and provides a more reliable estimate of model performance. The pipeline now trains a model for each fold and aggregates the results across folds using mean, standard deviation, median, minimum, and maximum. The number of folds is configurable via the `n_splits` parameter in `settings.py`, defaulting to 3.  This change affects the modeling, matrix generation, and evaluation pipelines, and is reflected in the corresponding catalog entries and pipeline definitions.

### 2. Enhanced Reporting Capabilities

The evaluation pipeline's reporting capabilities have been significantly improved, now including a wider range of metrics: accuracy, F1 score, Recall@n, AUROC, Hit@k, and MRR.  Reporting formats have also been enhanced to provide a clearer and more comprehensive overview of model performance.  The `reduce_aggregated_results` function simplifies reporting to MLflow, focusing on specific aggregations controlled by the `evaluation.reported_aggregations` parameter.

### 3. Streamlined Release Management

The release process has been simplified by removing the `changelog_gen.py` script.  Release information is now managed directly within `docs/src/data/releases.yaml`.  The `ReleaseInfoHooks` have been updated to extract relevant information for release notes, and the `create-release-pr.yml` GitHub workflow has been modified to reflect these changes. The workflow now triggers on branches prefixed with `debug/release*` and commits directly to the `docs/src/releases` directory.

### 4.  Technical Enhancements

-  Checks have been added to the evaluation pipeline to ensure that test data does not overlap with training data.
- Data catalog and parameter files (`catalog.yml`, `parameters.yml`) have been updated to support k-fold cross-validation, including changes to filepaths and data structures to handle multiple folds. Filepaths now include `fold_{fold}` placeholders (e.g., `modelling.model_input.fold_{fold}.splits`).
- Matrix generation now occurs for each fold and a full split using all data.


## Technical Deep Dive

Developers should be particularly aware of the following changes:

- **Data Catalog Changes:** The data catalog entries for modeling, matrix generation, and evaluation have been updated to include `fold_{fold}` placeholders in filepaths.
- **Pipeline Definitions:**  The modeling, matrix generation, and evaluation pipeline definitions have been refactored to handle k-fold cross-validation.
- **Release Information Management:**  The process for managing release information has been streamlined with the removal of `changelog_gen.py` and the update to `docs/src/data/releases.yaml`.

## Documentation ✏️

Documentation has been updated to reflect the introduction of k-fold cross-validation, including explanations of the process and the new metrics reported.  These updates are primarily within the pipeline documentation.


## Next Steps 🔮

Future work will continue to focus on improving model performance, expanding the knowledge graph, and enhancing the platform's capabilities.  This includes exploring new modeling techniques, integrating additional data sources, and automating the release pipeline.
