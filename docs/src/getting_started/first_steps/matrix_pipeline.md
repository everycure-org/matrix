# Matrix Pipeline

The Matrix pipeline is Every Cure's core data science pipeline for drug repurposing predictions. At a high level, it:

1. Ingests and integrates knowledge graph data from multiple biomedical sources

2. Computes embeddings for the graph entities using:
   - Node attribute embeddings from language models
   - Topological embeddings capturing graph structure

3. Trains ML models to predict drug-disease treatment relationships using:
   - Ground truth positive/negative drug-disease pairs
   - Cross-validation with stratified splits
   - Synthesized additional training data for better generalization
   - Multiple model architectures (XGBoost, Random Forest, Ensembles)

4. Generates an "all vs all" matrix of predictions for every drug-disease combination

5. Applies transformations to handle "frequent flyer" effects where certain drugs/diseases dominate predictions

6. Evaluates performance using multiple metrics:
   - Full matrix ranking (AUROC, MRR)
   - Disease-specific ranking (Hit@k)
   - Classification metrics (Accuracy, F1)

The pipeline enables systematic drug repurposing by scoring the potential of drugs to treat diseases based on patterns learned from biomedical knowledge graphs and known drug-disease relationships. For an in-depth understanding of each pipeline stage, see the [pipeline documentation](../pipeline/index.md).

All of these steps are reflected in our kedro pipeline which you will learn more about in our tech-stack section.

[Learn about our tech stack! :material-skip-next:](./tech_stack.md){ .md-button .md-button--primary }
