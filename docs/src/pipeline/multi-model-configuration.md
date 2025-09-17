# Multi-Model Configuration Guide

This document explains how to configure and run multiple models in parallel within the Matrix pipeline, with full support for multi-model inference and matrix generation.

## Overview

The Matrix pipeline supports running multiple machine learning models in parallel during training, matrix generation, transformation and evaluation. This capability allows you to:

- Train different model types simultaneously (e.g., xg_ensemble, xg_synth)
- Compare model performance across different configurations
- Generate matrices and predictions from ALL models
- Run inference on multiple models simultaneously
- Leverage parallel compute resources more effectively
- Perform comprehensive model comparison and analysis

## Configuration

### Model Definition

Models are defined in the `DYNAMIC_PIPELINES_MAPPING` configuration in [`matrix.settings.DYNAMIC_PIPELINES_MAPPING`](pipelines/matrix/src/matrix/settings.py), which is built with [`matrix.settings.generate_dynamic_pipeline_mapping`](pipelines/matrix/src/matrix/settings.py) and wrapped by [`matrix.settings.disable_private_datasets`](pipelines/matrix/src/matrix/settings.py):

```python
DYNAMIC_PIPELINES_MAPPING = (
    lambda: disable_private_datasets(
        generate_dynamic_pipeline_mapping(
            {
                # ... other configuration ...
                "modelling": [
                    {
                        "model_name": "xg_ensemble",
                        "model_config": {"num_shards": 1},
                    },
                    {
                        "model_name": "xg_synth",
                        "model_config": {"num_shards": 1},
                    },
                    {
                        "model_name": "lightgbm",
                        "model_config": {"num_shards": 1},
                    },
                ],
                # ... other configuration ...
            }
        )
    )
)
```

### Key Parameters

- **`model_name`**: Unique identifier for the model (must match parameter files)
- **`model_config`**: Model-specific configuration including number of shards

### Quickstart: run multi-model locally

- Install and fetch secrets (see [pipelines/matrix/Makefile](pipelines/matrix/Makefile)):
  - `make install`
  - `make fetch_secrets`
- Run training for all configured models:
  - `uv run kedro run --pipeline modelling`
- Generate matrices/predictions for all configured models:
  - `uv run kedro run --pipeline matrix_generation`
- Optionally run downstream transforms/evaluation similarly:
  - `uv run kedro run --pipeline matrix_transformation`
  - `uv run kedro run --pipeline evaluation`

Note: pipelines use the dynamic mapping in [`pipelines/matrix/src/matrix/settings.py`](pipelines/matrix/src/matrix/settings.py), so the set of models is code-driven.

## How It Works

### 1. Model Training Phase

During the modelling pipeline:

- All configured models train in parallel
- Each model produces its own set of outputs with model-specific naming
- Training artifacts are stored separately for each model

### 2. Matrix Generation Phase

During matrix generation:

- All models produce predictions in parallel
- Each model's predictions are stored with unique dataset names:
  ```
  matrix_generation.fold_{fold}.{model_name}.model_output.sorted_matrix_predictions
  ```
- All models generate their own reporting outputs (plots and tables)

### 3. Downstream Consumption

All pipelines now support multi-model operations:

- **Matrix Transformations**: Applies transformations to ALL models' outputs.
- **Matrix Evaluation**: Evaluates ALL models' outputs individually.
- **Reporting**: Generates separate reports for each model.

## Data Organization

### Output Structure

```
matrix_generation/
├── model_output/
│   ├── fold_0/
│   │   ├── xg_ensemble/          # Model-specific outputs
│   │   │   └── matrix_predictions/
│   │   ├── xg_synth/             # Model-specific outputs
│   │   │   └── matrix_predictions/
│   │   └── lightgbm/             # Model-specific outputs
│   │       └── matrix_predictions/
│   ├── fold_1/
│   │   ├── xg_ensemble/
│   │   ├── xg_synth/
│   │   └── lightgbm/
│   └── ...
└── reporting/                    # Separate reports for each model
    ├── xg_ensemble/
    │   ├── plots/
    │   └── tables/
    ├── xg_synth/
    │   ├── plots/
    │   └── tables/
    └── lightgbm/
        ├── plots/
        └── tables/
```

### Dataset Naming Convention

- **Training outputs**: `{model_name}_modelling.fold_{fold}.models.model`
- **Predictions**: `matrix_generation.fold_{fold}.{model_name}.model_output.sorted_matrix_predictions`
- **Transformers**: `{model_name}_modelling.fold_{fold}.model_input.transformers`

Tip: Ensure your catalog entries and any custom consumers expect the model_name segment in the dataset paths.

## Adding New Models

To add a new model to the multi-model configuration:

1. Add model configuration in [`pipelines/matrix/src/matrix/settings.py`](pipelines/matrix/src/matrix/settings.py):
   ```python
   {
       "model_name": "my_new_model",
       "model_config": {"num_shards": 1},
   }
   ```
2. Create parameter file at `pipelines/matrix/conf/base/modelling/parameters/my_new_model.yml`
3. Implement the model following existing patterns in the modelling pipeline code (see `pipelines/matrix/src/matrix/pipelines/modelling/`).

## Troubleshooting

### OutputNotUniqueError

If you see an error like:

```
OutputNotUniqueError: Output(s) ['matrix_generation.fold_0.model_output.sorted_matrix_predictions']
are returned by more than one nodes. Node outputs must be unique.
```

This indicates that multiple models are trying to write to the same output dataset. This issue has been resolved in the multi-model implementation, but ensure that:

- Each model has a unique `model_name`
- The pipeline is using the model-specific naming convention
- You're using the updated pipeline code that includes model names in output paths

### Model Configuration Issues

Ensure that:

- Each model has a unique and valid `model_name`
- All required parameter files exist for each model
- Model implementations follow the established patterns

## Performance Considerations

- Parallel Execution: Models train and generate predictions in parallel, improving overall pipeline performance
- Resource Usage: Each model consumes compute resources independently
- Storage: Each model's outputs are stored separately, increasing storage requirements
- Memory: Matrix generation phase may require more memory when multiple models run simultaneously

## Best Practices

1. Consistent Naming: Use descriptive and consistent model names
2. Parameter Organization: Keep model parameters in separate files for maintainability
3. Resource Planning: Consider compute and storage requirements when adding models
4. Testing: Use dry-run or run a single fold first to validate configuration before full execution
5. Model Comparison: Leverage multi-model outputs to compare performance and choose the best model for production

## Migration from Single Model

If migrating from a single-model setup:

1. No changes needed—existing models will automatically participate in multi-model inference
2. Ensure all consuming pipelines are updated to use the new naming convention
3. Update any custom catalog entries to support model-specific paths
4. Test with a single fold before full production execution

## Related Documentation

- [Modelling Pipeline](pipeline_steps/modelling.md)
- [Matrix Generation Pipeline](pipeline_steps/matrix_generation.md)
- [Dynamic Pipelines](../getting_started/deep_dive/kedro_extensions.md#dynamic-pipelines)
- [Custom Modelling Walkthrough](../getting_started/deep_dive/walkthroughs/custom_modelling.ipynb)
