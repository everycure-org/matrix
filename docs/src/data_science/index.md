---
title: Understanding the Data Science Pipeline That Powers Every Cure's Drug Pipeline
---

At Every Cure, we are on a mission to unlock the hidden potential of existing drugs to save lives. We believe that looking at all the existing drugs available in pharmacies today, we can find numerous new uses for these drugs to help people live better lives.

The answers are hidden in the data somewhere. We *just* have to find them. For that, we follow a fairly simple process of ingesting numerous data sources, integrating them to train models on the data that predict new high-potential drug-disease combinations.

## Leveraging Prior Knowledge

### Knowledge Graphs

Our drug repurposing pipeline begins with the construction and utilization of comprehensive knowledge graphs. These graphs integrate diverse biomedical data sources, including drug-target interactions, protein-protein interactions, disease pathways, and clinical outcomes. By representing this complex web of relationships in a graph structure, we can capture the intricate connections between drugs, diseases, and biological entities.

We use Neo4j as our graph database, allowing for efficient storage and querying of these large-scale knowledge graphs. This foundation enables us to leverage existing biomedical knowledge and discover potential new drug-disease associations that might not be apparent through traditional analysis methods.

## Generation of Input Data

### Node Embeddings

To transform our knowledge graph into a format suitable for machine learning models, we employ various node embedding techniques. These methods map each node (representing drugs, diseases, proteins, etc.) to a dense vector in a high-dimensional space, preserving the graph's structural and semantic information.

We experiment with several embedding algorithms, including:

- Node2Vec for random walk-based embeddings
- PubMedBert and OpenAI for LLM-based embeddings

These embeddings capture complex relationships within the graph, providing rich features for our downstream prediction tasks.

### Dimensionality Reduction

Given the high-dimensional nature of our node embeddings, we often apply dimensionality reduction techniques to make the data more manageable and to potentially uncover latent structures. Common methods we employ include:

- Principal Component Analysis (PCA) for linear dimensionality reduction
- t-SNE and UMAP for non-linear dimensionality reduction and visualization

These techniques help us to identify clusters of similar entities and reduce computational complexity for subsequent modeling steps.

### Topological Embeddings

In addition to traditional node embeddings, we explore topological embedding methods that capture higher-order structures within our knowledge graphs. These approaches, inspired by topological data analysis, include:

- GraphSAGE for inductive representation learning

Topological embeddings provide additional information to our node embeddings, uncovering complex patterns in drug-disease relationships that are not captured when not considering graph structure.

## Predicting Treatment Scores

### Predictions

With our enriched feature set, we train various machine learning models to predict treatment scores for drug-disease pairs. Our modeling approaches include:

- Random Forest.....
- Gradient Boosting models (XGBoost) for their ability to handle complex feature interactions
- Deep Neural Networks
- Ensemble methods that combine multiple model types to improve robustness and performance

We use MLflow to track experiments, comparing different model architectures and hyperparameters to identify the most promising approaches.

## Evaluation

### Model Evaluation

Rigorous evaluation is crucial to ensure the reliability and usefulness of our predictions. Our evaluation strategy includes:

- Cross-validation techniques to assess model generalization
- Metrics such as AUROC, to quantify predictive performance
- Comparison against baseline models and random predictions to measure improvement
- External validation using held-out datasets and literature-based validation of top predictions




## Conclusion

Through this comprehensive pipeline, from knowledge graph construction to final model evaluation, we aim to systematically uncover new potential uses for existing drugs, accelerating the drug repurposing process and ultimately improving patient outcomes.
