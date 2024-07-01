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