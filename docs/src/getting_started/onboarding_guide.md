# Onboarding

Welcome to the Matrix onboarding guide! This document provide an introduction to the codebase, and guide you through the process of setting up your local environment.

# Preliminaries

The Matrix project contains a pipeline and auxiliary services designed to identify drug-disease pair candidates for drug-repurposing.

Moreover, the codebase contains the defintion of the infrastructure to run the pipeline on Google Cloud.

## Kedro

> 🔎 Kedro is an open-source framework to write modular data science code. We recommend checking out the [introduction video](https://docs.kedro.org/en/stable/introduction/index.html).

We're using [Kedro](https://kedro.org/) as our data pipelining framework. Kedro is a rather light framework, that comes with the following [key concepts](https://docs.kedro.org/en/stable/get_started/kedro_concepts.html#):

1. __Project template__: Standard directory structure to streamline project layout, i.e., configuration, data, and pipelines.
2. __Data catalog__: A lightweight abstraction for datasets, abstracting references to the file system, in a compact configuration file.
3. __Pipelines__: A `pipeline` object abstraction, that leverages `nodes` that plug into `datasets` as defined by the data catalog[^1].
4. __Vizualiation__: Out-of-the-box pipeline visualization based directly on the source code.

[^1]: Kedro allows for fine-grained control over pipeline execution, through the [kedro run](https://docs.kedro.org/en/stable/nodes_and_pipelines/run_a_pipeline.html) command.

### Data layer convention

Data used by our pipeline is registered in the data catalog. To add additional structure to the catalog items, we organise our data according to the following convention:

1. __Raw__: Data as received directly from the source, no pre-processing performed.
2. __Intermediate__: Data with simple cleaning steps applied, e.g., correct typing and column names.
3. __Primary__: Golden datasets, usually obtained by merging _intermediate_ datasets.
4. __Feature__: Primary dataset enriched with features inferred from the data, e.g., enriching an `age` column given a `date-of-birth` column.
5. __Model input__: Dataset transformed for usage by a model.
6. __Models__: Materialized models, often in the form of a pickle.
7. __Model output__: Dataset containing column where model predictions are ran.
8. __Reporting__: Any datasets that provide reporting, e.g., convergence plots.  

![](../assets/img/convention.png)

### Data fabrication

Our pipeline operates on large datasets, as a result the pipeline may take several hours the complete. Large iteration time leads to decreased developer productivity. For this reason, we've established a data fabricator to enable test runs on synthetic data.

To seamlessly run the same codebase on both the fabricated and the production data, we leverage [Kedro configuration environments](https://docs.kedro.org/en/stable/configuration/configuration_basics.html#configuration-environments).


![](../assets/img/fabrication.drawio.svg)


## Pipeline

### Overview

Roughly speaking, our pipeline consists of five logical stages, i.e., ingestion, integration, embedding, modelling, and evaluation.

![](../assets/img/status.drawio.svg)

#### Ingestion

The ingestion pipeline aims to ingest all the downstream data in BigQuery, our data warehouse of choice. Data from different sources is assigned metadata for lineage tracking.

> To date our pipeline only ingests data from the RTX-KG2 source.

#### Integration

The integration stage aims to produce our internal knowledge-graph, in [biolink](https://biolink.github.io/biolink-model/) format. As we ingest data from different sources, entity resolution becomes a prevalent topic. The integration step consolidates entities across sources to avoid data duplication in the knowledge graph.

> To data, this step is missing as we're only ingesting data from a single source.

#### Embeddings

Embeddings are vectorized representations of the entities in our knowledge graph. These are currently computed using two steps:

1. GenAI model is used to compute individual node embeddings
2. GraphSage embedding algorithm is ran on the node embeddings to produce topological embeddings

> Our graph database, i.e., Neo4J comes with out-of-the-box functionality to compute both node and topological embeddings in-situ. The Kedro pipeline orchestrates the computation of these.


#### Evaluation

The evaluation pipeline computes various metrics in order to assess the performance of the models trained in the previous stages. 

Currently, we have the following evaluation methods. 

1. *Threshold-based classification metrics for ground truth data*. Measures how well the model classifies ground truth positive and negatives using threshold-based metrics such as accuracy and F1-score.
2. *Threshold-independent metrics for ground truth data*. Measures how well the model classifies ground truth positive and negatives using threshold-independent metrics such as AUROC.
3. *All vs. all ranking with all drugs x test diseases matrix.*. Gives information on all drugs vs all disease ranking performance of models by using threshold-independent metrics such as AUROC and synthesised negatives. The construction of the synthesised negatives are based on a matrix of drug-disease pairs for a given list of all drugs and the list of disease appearing in the ground-truth positive test set. 
4. *Disease-specific ranking*. Measures the performance of the model at ranking drugs for a fixed disease using metrics such as Hit@k and mean reciprocal rank (MRR). 


