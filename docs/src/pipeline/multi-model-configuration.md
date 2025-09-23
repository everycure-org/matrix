# Multi-Model Configuration Guide

This guide explains how to configure and run multiple modelling pipelines within Matrix. We have introduced packaged model wrappers and ensemble aggregation during matrix generation so that multiple trained models can score in one run.

## Overview

- Models listed in `matrix.settings.DYNAMIC_PIPELINES_MAPPING()["modelling"]` train in parallel across folds and shards.
- Matrix generation wraps each trained model with its preprocessing (`ModelWithPreprocessing`) before scoring.
- A configurable aggregator (`matrix_generation.model_ensemble.agg_func`) combines per-model predictions into one matrix per fold.
- Downstream pipelines (transformations, evaluation, inference) now consume those aggregated predictions by default.

## Dynamic Pipeline Configuration

- The dynamic mapping lives in `pipelines/matrix/src/matrix/settings.py` and can be overridden through `KEDRO_DYNAMIC_PIPELINES_MAPPING_*` environment variables.
- Each entry in the `modelling` list must provide a `model_name` that matches `pipelines/matrix/conf/base/modelling/parameters/<model_name>.yml`.
- `num_shards` controls how many shard-specific estimators are trained per model and per fold; it is shared across models unless overridden via the environment.
- `cross_validation.n_cross_val_folds` defines how many CV folds are generated (an additional fold trains on the full dataset).

## Key Parameter Files

- `pipelines/matrix/conf/base/modelling/parameters/defaults.yml` supplies the base generator, transformers, tuner, and metrics that every model inherits.
- Per-model files such as `pipelines/matrix/conf/base/modelling/parameters/xg_ensemble.yml` override the defaults to define estimators, features, and custom metrics.
- Ensemble aggregation within modelling is configured through `modelling.<model>.model_options.ensemble.agg_func`, controlling how shard outputs are merged.
- Matrix-generation aggregation across models is configured in `pipelines/matrix/conf/base/matrix_generation/parameters.yml` under `matrix_generation.model_ensemble.agg_func`.

## Running Multi-Model Pipelines

- Install dependencies and pull secrets: `make install`, `make fetch_secrets`.
- Train all configured models: `uv run kedro run --pipeline modelling`.
- Generate matrices and predictions (aggregated across models): `uv run kedro run --pipeline matrix_generation`.
- Run optional follow-up stages as needed: `uv run kedro run --pipeline matrix_transformation`, `uv run kedro run --pipeline evaluation`.

## Pipeline Behavior

### Modelling

- Shared nodes build filtered datasets and cross-validation splits once for the entire model roster.
- For every `model_name`, the pipeline fits preprocessing transformers per fold, trains shard-specific estimators, and aggregates them with `ModelWrapper`.
- Outputs stored per model include transformers, fitted models, fold predictions, combined CV predictions, and sanity-check metrics.
- Catalogue entries live under `modelling.fold_{fold}.{model_name}.*` (see catalog.yml for full list).

### Matrix Generation

- `package_model_with_preprocessing` creates a `ModelWithPreprocessing` for every `(fold, model)` pairing and persists it at `matrix_generation.fold_{fold}.{model_name}.wrapper`.
- These wrappers encapsulate the estimator, fitted transformers, and feature list so inference reuses the exact preprocessing stack.
- `matrix_generation.fold_{fold}.wrapper` aggregates the per-model wrappers using `matrix_generation.model_ensemble.agg_func` (default: `numpy.mean`).
- `make_predictions_and_sort` executes once per fold with the aggregated wrapper and emits a single predictions table per fold at `matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions`.

### Downstream Consumption

- Matrix transformations, evaluation, stability, and inference pipelines read the aggregated predictions produced for each fold.
- Reporting nodes reuse the same aggregated outputs for plots and tables; per-model wrappers remain available for bespoke analyses or ablations.
- Inference reuses the full-data fold wrapper (`fold_{n_cross_val_folds}`) when serving predictions.

## Data Layout

```
matrix_generation/
├── model_wrappers/
│   ├── fold_0/
│   │   ├── wrapper.pickle                # aggregated multi-model wrapper
│   │   ├── xg_ensemble/wrapper.pickle    # per-model wrapper with preprocessing
│   │   └── xg_synth/wrapper.pickle
│   └── fold_3/
│       └── ...
└── model_output/
    ├── fold_0/matrix_predictions/
    └── fold_3/matrix_predictions/
```

### Dataset Naming Highlights

- Training artefact: `modelling.fold_{fold}.{model_name}.models.model`.
- Transformers: `modelling.fold_{fold}.{model_name}.model_input.transformers`.
- Per-model wrappers: `matrix_generation.fold_{fold}.{model_name}.wrapper`.
- Aggregated predictions: `matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions`.

> Tip: Update any custom Kedro catalog entries to read the aggregated dataset or the per-model wrappers as required.

## Adding or Removing Models

1. Adjust the `modelling` list inside `matrix.settings.DYNAMIC_PIPELINES_MAPPING` (or set the matching environment variable) to add or drop `{"model_name": "<your_model>"}` entries.
2. Create `pipelines/matrix/conf/base/modelling/parameters/<your_model>.yml` by copying `defaults.yml` and adapting the generator, transformers, tuner, ensemble aggregation, and metrics.
3. Confirm downstream consumers either reference the aggregated predictions or target a specific per-model wrapper.

## Troubleshooting

- `OutputNotUniqueError` for `matrix_generation.fold_{fold}.model_output.sorted_matrix_predictions` indicates that multiple nodes are mapped to the same dataset; ensure only the updated matrix generation node writes to it.
- If predictions ignore preprocessing, verify the fold-specific transformers exist and that `ModelWithPreprocessing` points to the expected feature list.
- Unexpected ensemble behaviour usually traces back to `matrix_generation.model_ensemble.agg_func` or per-model `ensemble.agg_func`; review those settings when scores diverge.
- Alignment issues between modelling and matrix generation typically mean the pipelines were run with different model rosters—rerun modelling before scoring.

## Performance Considerations

- Multi-model execution increases GPU utilisation during modelling; adjust `num_shards` or the model roster when scheduling becomes a bottleneck.
- Matrix generation now invokes every model wrapper during scoring; tune Spark partitioning via `matrix_generation.matrix_generation_options` if runtime grows.
- Storage usage scales with the number of models and folds because wrappers and predictions are persisted per fold.
- Memory requirements rise when generating predictions for many models simultaneously; consider batching targets via `matrix_generation.matrix_generation_options.batch_by`.

## Migration Notes

- Single-model setups continue to work; the aggregated predictions now represent the multi-model ensemble result.
- Update legacy consumers that referenced `matrix_generation.fold_{fold}.{model_name}.model_output.sorted_matrix_predictions` to use the aggregated dataset or the new per-model wrappers.
- Validate downstream analytics with a single fold before full runs, especially when changing aggregation functions or adding new models.
- Keep the modelling and matrix generation pipelines in sync—rerun modelling whenever transformers or estimators change.

## Related Documentation

- [Modelling Pipeline](pipeline_steps/modelling.md)
- [Matrix Generation Pipeline](pipeline_steps/matrix_generation.md)
- [Dynamic Pipelines](../getting_started/deep_dive/kedro_extensions.md#dynamic-pipelines)
- [Custom Modelling Walkthrough](../getting_started/deep_dive/walkthroughs/custom_modelling.ipynb)
