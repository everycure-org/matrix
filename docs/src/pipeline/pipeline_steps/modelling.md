The modelling pipeline trains prediction models using drug-disease pairs and knowledge graph embeddings to predict treats relationships between drugs and diseases.

## Overview

The modelling pipeline implements a robust cross-validation strategy with ensemble learning to predict whether a drug treats a disease. The pipeline classifies three categories of drug-disease relationships:

- **Treat**: relationships where a drug treats a disease
- **Not Treat**: negative relationships where a drug does not treat a disease. These are often contraindications
- **Unknown**: pairs where the relationship is unknown 

For every drug disease pair, we compute scores for all 3 classes.
<!-- Do all add up to 1? Do we have some existing docs from experiments for this? -->

## Process

The modelling pipeline follows a systematic approach to train robust prediction models:

1. **Cross-Validation Setup**: Create 4 folds total:
    - Folds 0, 1, 2: Training folds with different test/train splits (90%/10%) for known positives and negatives
    - Fold 3: Full training data fold (no test split) for final model production

2. **Shard Generation**: For each fold, create 3 shards with different randomly sampled negative examples to address class imbalance and reduce sampling bias

3. **Model Training**: Train individual XGBoost classifiers on each fold-shard combination using the transformed feature data

4. **Ensemble Creation**: Combine predictions from all shards within each fold using aggregation functions to produce one ensemble model per fold

5. **Production Model**: Use the Fold 3 ensemble model (trained on full dataset) for final matrix generation and inference




![](../../assets/img/modelling_pipeline_docs.drawio.png)



### 1. Cross-Validation Setup

The pipeline uses a cross-validation approach with multiple folds:

- **Folds 0, 1 and 2**: Training folds used for model evaluation and hyperparameter tuning
    - Each fold contains different train/test splits from the ground truth data
    - Used to assess model performance and prevent overfitting
- **Fold 3**: Full training data fold used for final model training
    - Contains all available training data (no test split)

!!! info
    `fold_3` is the model used to generate results that are shared with the Medical Team

#### Ground Truth Data

The pipeline pulls in ground truth data from the release containing known positive and negative examples.

Please see data documentation for more information.

<!-- Will add link when it exists -->


#### Split Generation

The `make_folds` function creates cross-validation splits using [StratifiedShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html) from scikit-learn:

- Generates train/test splits for folds 0-2 using the specified splitter strategy
    - Train 90%
    - Test 10%
- Creates a full training dataset for fold 3 (no test split)
- Each split maintains the distribution of positive and negative examples


### 2. Shard Generation

To address class imbalance and reduce bias, the pipeline implements a negative sampling approach:

- Generates the "unknown" drug-disease pairs by randomly sampling from the space of possible drug disease combinations
- We choose any drug-disease pairs at random, because they are overwhelmingly likely to be negatives
- Creates 3 shards, each with a different randomly sampled set of negative examples
- Multiple shards help reduce the impact of random sampling bias on model performance


### 3. Model Training

For each shard in each fold, the pipeline trains an [XGBoost](https://xgboost.readthedocs.io/en/stable/) classifier on the transformed feature data.


### 4. Ensemble Creation

After training individual shard models, the pipeline creates ensemble models:

- Combines predictions from all shards within each fold using the specified aggregation function
- Creates a single XGBoost classifier object representing the ensemble for each fold
- Applies the ensemble model to generate predictions on the test data


