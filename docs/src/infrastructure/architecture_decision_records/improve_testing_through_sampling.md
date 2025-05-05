---
title: Improve testing through sampling
--- 

## Status

Accepted

## Context

### Problems

1. How can we catch more errors before reaching production, in local testing and CI?
2. How can we allow data scientists to run the modelling stage on subsets of the production data?

### User context 

Running the full pipeline in production currently takes 12 hours given the volume of data it processes. We are looking for options to allow data scientists and engineers to iterate faster on their code or modelling changes. 

The current infrastructure uses a Data Fabricator class to fabricate synthetic data following the same schema as production data. This synthetic data is then used in the test pipeline that can be run locally and used in the CI.

The Data Fabricator allows for capturing a wide range or errors, but it is not perfect. At various occasions it fails to detect errors due to drifts between:

1. Synthetic and real data 
2. Local and cloud environments setup

After discussion with the team, these are the most common issues that seem to arise in production without being caught in the test pipeline:

| Issue | Importance | Description | Why is it not covered in tests | Mitigation |
| ------------- | ------------- | ------------- | ------------- | ------------- |
| 1. Normalization API | High | The normalization API can fail to normalize some ids and return a null. A few nulls can be handled, but too many and the modelling stage will fail. This usually happens 10 hours or so after the start of the pipeline, and can be caused by the parameters used to configure the API. | Data drift: the test pipeline uses a mock of the normalization API. | None |  
| 2. Gap between fabricated data and production data | High | The biggest issue in that space is related to non-nullable columns in fabricated data, while being nullable in production. | Data drift: fabricated data doesn't represent production data accurately. | Manually update fabricator parameters |
| 3. Pre-processing pipeline | Medium | The pre-processing pipeline can fail with some manually typed data sources. | Environment drift: the test pipeline doesn't run pre-processing. | None |
| 4. Kedro local Memory dataset | Medium | The production pipeline can fail as it cannot reach a memory dataset computed in a previous stage. | Environment drift: the test pipeline is sharing memory between stages, but the production environment is not. | None |
| 5. Incorrect source files | Medium | Either the files can be outdated, or their file paths can be incorrect when running the production pipeline. | Environment drift: the test pipeline doesn't use the same files as production. | None |
| 6. Out Of Memory | Low | Memory errors will make the production pipeline fail. They often arise in Neo4J. Restarting the pipeline usually solves the issue. | Environment drift: the test pipeline does cover out of memory failures. | None |

There is already work in progress to take care of the environment drift between fabricated data and production, which should address issues 4 and 5.

### Technical context

We are using the Kedro framework which easily allows to run specific nodes of the pipelines in different environments.

The data sources consists of knowledge graphs, tabular data or key value pairs. When testing with a subset of a knowledge graphs, as they comprise nodes connected by edges, we would like the subset to be representative of the whole graph.

We would like to abstract the sampling of the data to allow for different strategies depending on the user's needs. Ideally, we would want data scientists to define their own strategies.

We would like to persist as little sampled data as possible to limit the risk of dataset invalidation issues.

We would like to keep any sample pipeline runtime below 30 minutes to allow for fast feedback and low cloud costs.

## Decision

### Mitigation: improving the data fabricator

1. Update data fabricator parameters to better represent production data (especially nullable columns). 
2. Write a script to semi-automatically update the fabricator's parameters based on production data.

### Short term: run the pipeline on a sample of production data 

#### What is the sampling logic?

We would like to get a sample of the full knowledge graph. We would like this sample to keep the same proportions of ground truth pairs for the ML models to work.

Ultimately, the sampling logic will be abstracted out as a parameter to allow users to create their own custom logics.

#### How does sampling integrates with the pipeline?

We want to sample the output of the integration pipeline, for it to be used in following stages - everything from embeddings.

__Outputs of the sampling pipeline__
 We will only sample and persist the nodes and edges of the knowledge graph as it is the biggest data source. This way we limit the number of persisted files to maintain, and rely on inner joins to propagate the sample to the other data sources.

__Inputs of the sampling pipeline__
The inputs of the sampling pipeline are the integrated knowledge graph nodes and edges, as well as ground truth pairs.

#### How to set this up in Kedro?

A sample can be created by running the command `kedro run -e sample -p create_sample`, this command will populate the sample environment with the sampled data.

To run the pipelines with sampled data, we can run the command `kedro run -e sample -p test_sample` which runs the the pipeline from the embeddings stage.

## Consequences

We can now iterate faster on code and model changes and catch more errors before reaching production. This increases productivity among engineers and data scientists.
