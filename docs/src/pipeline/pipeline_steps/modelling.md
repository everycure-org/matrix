### Modelling

The modelling pipeline trains prediction models using drug-disease pairs and knowledge graph embeddings to predict treats relationships between drugs and diseases.

## Overview

The modelling pipeline implements a robust cross-validation strategy with ensemble learning to predict whether a drug treats a disease. The pipeline classifies three categories of drug-disease relationships:
- **Treat**: Known positive relationships where a drug treats a disease
- **Not Treat**: Known negative relationships where a drug does not treat a disease  
- **Unknown**: Pairs where the relationship is unknown (used for prediction)

## Cross-Validation Strategy

The pipeline uses a sophisticated cross-validation approach with 4 folds:

- **Folds 0-2**: Training folds used for model evaluation and hyperparameter tuning
  - Each fold contains different train/test splits from the ground truth data
  - Used to assess model performance and prevent overfitting
- **Fold 3**: Full training data fold used for final model training
  - Contains all available training data (no test split)
  - This is the model whose results are shared and used for inference

This approach ensures robust evaluation while maximizing the use of available training data for the final model.

## Data Preparation

### Ground Truth Construction

The pipeline starts by mapping ground truth data to the knowledge graph:

1. **Filter Valid Pairs**: Extracts drug-disease pairs from the knowledge graph, filtering by specified drug and disease types
2. **Attach Embeddings**: Enriches pairs with node embeddings from the embedding pipeline
3. **Prefilter Nodes**: Creates a curated set of drug and disease nodes for model training

### Split Generation

The `make_folds` function creates cross-validation splits:

- Generates train/test splits for folds 0-2 using the specified splitter strategy
    - Train 90%
    - Test 10%
- Creates a full training dataset for fold 3 (no test split)
- Each split maintains the distribution of positive and negative examples

### Negative Sampling Strategy

To address class imbalance and reduce bias, the pipeline implements a negative sampling approach:

1. **Sharding**: Creates 3 shards, each with a different randomly sampled set of negative examples

## Model Training Process

### Per-Shard Training

For each fold and shard combination, the pipeline:

1. **Data Transformation**: Applies fitted transformers to the enriched splits
2. **Hyperparameter Tuning**: Uses the training data to tune model parameters
3. **Model Training**: Trains an XGBoost classifier on the transformed data

### Ensemble Creation

After training individual shard models, the pipeline creates ensemble models:

1. **Shard Averaging**: Combines predictions from all shards within each fold using the specified aggregation function
2. **Fold Model**: Creates a single XGBoost classifier object representing the ensemble for each fold
3. **Prediction Generation**: Applies the ensemble model to generate predictions on the test data

## Pipeline Architecture

The modelling pipeline consists of several key components:

### Shared Pipeline
- **Filter Valid Pairs**: Extracts and validates drug-disease relationships
- **Attach Embeddings**: Enriches data with node embeddings
- **Prefilter Nodes**: Creates curated node sets
- **Make Folds**: Generates cross-validation splits

### Model-Specific Pipeline
- **Enrich Splits**: Creates enriched datasets for each shard
- **Per-Fold Training**: Trains models for each fold and shard combination
- **Ensemble Creation**: Combines shard models into fold ensembles
- **Performance Evaluation**: Evaluates model performance across folds

## Model Configuration

The pipeline supports configurable model parameters through the `modelling` configuration:

- **Model Options**: Defines the model type, features, and training parameters
- **Transformers**: Specifies data preprocessing steps
- **Ensemble Configuration**: Defines how to combine shard predictions
- **Metrics**: Specifies evaluation metrics for model performance

## Outputs

The pipeline produces several key outputs:

- **Trained Models**: XGBoost classifiers for each fold
- **Predictions**: Model predictions on test data for each fold
- **Performance Metrics**: Evaluation metrics across all folds
- **Tuning Reports**: Hyperparameter tuning convergence plots
- **Combined Predictions**: Aggregated predictions from all evaluation folds

## Usage

The modelling pipeline is designed to be flexible and configurable:

```python
# Example configuration
modelling:
  model_name: "xgboost_classifier"
  model_config:
    num_shards: 3
  model_options:
    features: ["embedding_1", "embedding_2", ...]
    target_col_name: "relationship_type"
    ensemble:
      agg_func: "mean"
```


