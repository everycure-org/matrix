# `v0.3.0`: Robust Model Evaluation with K-Fold Cross-Validation

This release introduces k-fold cross-validation to the Matrix Pipeline, significantly enhancing model evaluation robustness and reducing the impact of random train-test splits. Alongside cross-validation, this release delivers improvements to the evaluation pipeline, reporting, and overall pipeline structure.

<!-- more -->

## Key Enhancements

### 1. K-Fold Cross-Validation

The modelling, matrix generation, and evaluation pipelines now support k-fold cross-validation. The pipeline generates multiple train-test splits (“folds”), trains a model for each, and aggregates performance metrics across all folds. A “full” split, using all data for training, is also created.  This provides a more comprehensive evaluation of model performance and stability. The number of folds is configurable through `settings.py` and defaults to 3, plus the "full" dataset, resulting in 4 training runs.

### 2. Enhanced Evaluation and Reporting

The evaluation pipeline has been refactored for improved efficiency and reporting.  It now includes a broader range of metrics: accuracy, F1 score, Recall@n, AUROC, Hit@k, and MRR, providing a more detailed performance analysis.  Reporting has been streamlined with a master evaluation report consolidating results across all models, evaluations, and folds.  A reduced aggregation is also provided for more concise MLFlow reporting, controlled by `evaluation.reported_aggregations`.

### 3. Streamlined Release Management

Release information is now managed directly within `docs/src/data/releases.yaml`, replacing the dynamic generation process.  This simplifies release management and improves maintainability. A new release history section (`docs/src/releases/release_history.md`) displays the release notes based on this YAML file.

### 4. Pipeline Configuration Updates

The pipeline configuration files (`catalog.yml` and `parameters.yml`) have been updated to accommodate k-fold cross-validation.  Changes include updated dataset paths and parameters to handle multiple folds and aggregated results.  The inference pipeline now defaults to using the model trained on the "full" dataset (`fold_full`).

## Technical Details

**Cross-Validation Implementation:**

The `modelling` pipeline's `make_folds` function generates folds based on the `n_splits` parameter in `settings.py`. The `combine_data` and `aggregate_metrics` functions handle the combination of predictions and metrics from different folds. `modelling.aggregation_functions` in `parameters.yml` defines the aggregation methods (mean, std, median, min, max). The `matrix_generation` and `evaluation` pipelines are adjusted to handle fold-specific data and results.

**Evaluation Enhancements:**

The `evaluation/nodes.py` introduces functions like `reduce_aggregated_results` to simplify MLFlow reporting and `consolidate_evaluation_reports` for cross-fold aggregation. `evaluation.reported_aggregations` in `parameters.yml` controls which aggregations are reported.

**Reporting Changes:**

The `changelog_gen.py` script has been removed. The `ReleaseInfoHooks` in `hooks.py` now extracts release information from `releases.yaml`.

**Other Changes:**

- The GitHub Actions workflow (`create-release-pr.yml`) now triggers on `debug/release*` branches.
- A new test file (`tests/pipelines/test_modelling.py`) tests the `make_folds` functionality.  
- Experiments were conducted with matrix transformation and normalization techniques to address biases.

## Conclusion

This release focuses on improving model evaluation robustness and reporting, streamlining release management, and enhancing the pipeline structure for handling cross-validation. These improvements contribute to more reliable model development and analysis within the Matrix Pipeline.
