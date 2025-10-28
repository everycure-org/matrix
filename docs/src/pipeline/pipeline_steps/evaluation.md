The evaluation pipeline computes various metrics in order to assess the performance of the models trained in the previous stages. This now supports a broader suite of metrics with enhanced reporting capabilities, including accuracy, F1 Score, Recall@n, AUROC, Hit@k, and MRR. Evaluations are performed using drug-disease pair scores from the matrix generation step, ensuring computational efficiency by avoiding repeated inference.

Computed metrics generally fall into three categories:

1. **Full-matrix ranking metrics**: These metrics focus on how well the model ranks the full set of drug-disease pairs comprising the matrix (e.g. AUROC, MRR).
2. **Disease-specific ranking metrics**: These metrics assess ranking precision for each specific disease (e.g. Hit@k).
3. **Ground truth classification metrics**: These metrics examine the model's ability to distinguish between known positive and known negative drug-disease pairs (e.g. accuracy, F1 score).

The evaluation pipeline utilises k-fold cross-validation, where by the metrics are computed for each fold, and then aggregated across folds for a more robust estimate of the model's performance.

More details on the metrics computed in each category can be found in the [evaluation deep-dive](../data_science/evaluation_deep_dive.md)



## Implementation details

The evaluation pipeline takes as input the full matrix dataset of drug-disease pairs, along with with their treat scores. 
For each evaluation metric, this large input dataset is processed by:
- Restricting to only the pairs required for the evaluation metric
- Labelling any ground truth pairs
The processed dataset is then used to compute the evaluation metric. 

This process is summarised by the following high-level diagram:

![Evaluation Pipeline](../../assets/deep_dive/evaluation_simple.drawio.svg)

Notably, inference is not performed within the evaluation pipeline since the treat scores are provided in the input matrix.

> *Example.* (ground truth classification metric) The full matrix dataframe is restricted to ground truth positives, which are labelled by `y=1`, and ground truth negatives which are labelled by `y=0`.   

The following diagram gives a more detailed overview of how the evaluation pipeline fits into the wider MATRIX system. 

![Evaluation Pipeline Full](../../assets/deep_dive/evaluation_full_matrix_docs.drawio.svg)


### Parameters configuration file 

The evaluation metrics are configured in the file `parameters.yml`. Let us explain the structure of this file through two examples. 
#### Example 1 (classification metrics, clinical trials version)

```yaml
# Threshold-based classification metrics for ground truth data (clinical trials version)
evaluation.simple_classification_trials:
  evaluation_options:
    generator:
      object: matrix.datasets.pair_generator.GroundTruthTestPairs
      positive_columns: 
        - "trial_sig_better"
        - "trial_non_sig_better"
      negative_columns:
        - "trial_sig_worse"
        - "trial_non_sig_worse"
    evaluation:
      object: matrix.pipelines.evaluation.evaluation.DiscreteMetrics
      metrics:
        - object: sklearn.metrics.accuracy_score
        - object: sklearn.metrics.f1_score
      score_col_name: *score-col
      threshold: 0.5
```
- `generator` defines the dataset that is required for the metrics. 
- `GroundTruthTestPairs` is represents a dataset consisting ground truth positives and negatives.
- `positive_columns` and `negative_columns` define the ground truth sets by specifying columns in the matrix dataframe. 
- `evaluation` defines the evaluation metrics used. 

#### Example 2 (full matrix ranking metrics, standard version)

```yaml
# Full matrix ranking 
evaluation.full_matrix:
  evaluation_options:
    generator:
      object: matrix.datasets.pair_generator.FullMatrixPositives
      positive_columns: 
        - "is_known_positive"
    evaluation:
      object: matrix.pipelines.evaluation.evaluation.FullMatrixRanking 
      rank_func_lst: 
        - object:  matrix.pipelines.evaluation.named_metric_functions.RecallAtN 
          n: 1000
        - object:  matrix.pipelines.evaluation.named_metric_functions.RecallAtN
          n: 10000
        - object:  matrix.pipelines.evaluation.named_metric_functions.RecallAtN
          n: 100000
        - object:  matrix.pipelines.evaluation.named_metric_functions.RecallAtN
          n: 1000000
      quantile_func_lst: 
        - object: matrix.pipelines.evaluation.named_metric_functions.AUROC
``` 
- `FullMatrixPositives` is the object defining the dataset containing the necessary information for the computation of the full matrix ranking metrics. This dataset consists of the ground truth positive drug-disease pairs with columns giving:
    - The matrix rank of each pair
    - The matrix quantile rank of each pair
- This is enough for the computation of the metrics since:
    - Recall@n may be computed using the ranks of the ground truth positive pairs
    - AUROC may be computed using the quantile ranks of the ground truth positive pairs  
- `positive_columns` defines the ground truth positive set by specifying columns in the matrix dataframe. 
- `rank_func_lst` specifies the list of metrics which require the rank for computation.
- `quantile_func_lst` specifies the list of metrics which require the quantile rank for computation.
 