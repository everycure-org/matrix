# Pipeline

## Overview

Roughly speaking, our pipeline consists of five logical stages, i.e., ingestion, integration, embedding, modelling, and evaluation.

![](../assets/img/status.drawio.svg)

## Stages

### Ingestion

The ingestion pipeline aims to ingest all the downstream data in BigQuery, our data warehouse of choice. Data from different sources is assigned metadata for lineage tracking.

> To date our pipeline only ingests data from the RTX-KG2 source.

### Integration

The integration stage aims to produce our internal knowledge-graph, in [biolink](https://biolink.github.io/biolink-model/) format. As we ingest data from different sources, entity resolution becomes a prevalent topic. The integration step consolidates entities across sources to avoid data duplication in the knowledge graph.

> To data, this step is missing as we're only ingesting data from a single source.

### Embeddings

Embeddings are vectorized representations of the entities in our knowledge graph. These are currently computed using two steps:

1. GenAI model is used to compute individual node embeddings
2. GraphSage embedding algorithm is ran on the node embeddings to produce topological embeddings

> Our graph database, i.e., Neo4J comes with out-of-the-box functionality to compute both node and topological embeddings in-situ. The Kedro pipeline orchestrates the computation of these.


### Evaluation

The evaluation pipeline computes various metrics in order to assess the performance of the models trained in the previous stages. 

Currently, we have the following evaluation methods. 

1. *Threshold-based classification metrics for ground truth data*. Measures how well the model classifies ground truth positive and negatives using threshold-based metrics such as accuracy and F1-score.
2. *Threshold-independent metrics for ground truth data*. Measures how well the model classifies ground truth positive and negatives using threshold-independent metrics such as AUROC.
3. *All vs. all ranking with all drugs x test diseases matrix.*. Gives information on all drugs vs all disease ranking performance of models by using threshold-independent metrics such as AUROC and synthesised negatives. The construction of the synthesised negatives are based on a matrix of drug-disease pairs for a given list of all drugs and the list of disease appearing in the ground-truth positive test set. 
4. *Disease-specific ranking*. Measures the performance of the model at ranking drugs for a fixed disease using metrics such as Hit@k and mean reciprocal rank (MRR). 


