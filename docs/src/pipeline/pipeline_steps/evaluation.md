### Evaluation

The evaluation pipeline computes various metrics in order to assess the performance of the models trained in the previous stages. This now supports a broader suite of metrics with enhanced reporting capabilities, including accuracy, F1 Score, Recall@n, AUROC, Hit@k, and MRR. Evaluations are performed using drug-disease pair scores from the matrix generation step, ensuring computational efficiency by avoiding repeated inference.

Computed metrics generally fall into three categories:

1. **Full-matrix ranking metrics**: These metrics focus on how well the model ranks the full set of drug-disease pairs comprising the matrix (e.g. AUROC, MRR).
2. **Disease-specific ranking metrics**: These metrics assess ranking precision for each specific disease (e.g. Hit@k).
3. **Ground truth classification metrics**: These metrics examine the model's ability to distinguish between known positive and known negative drug-disease pairs (e.g. accuracy, F1 score).

The evaluation pipeline utilises k-fold cross-validation, where by the metrics are computed for each fold, and then aggregated across folds for a more robust estimate of the model's performance.

More details on the metrics computed in each category can be found in the [evaluation deep-dive](../data_science/evaluation_deep_dive.md)