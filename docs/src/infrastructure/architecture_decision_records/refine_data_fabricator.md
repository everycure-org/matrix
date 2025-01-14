---
title: Refine Data Fabricator
--- 

## Status

Under review

## Context

### Problems

1. How can we catch more errors before reaching production, in local testing and CI?
2. How can we allow data scientists to run the modelling stage on subsets of the production data?
3. How can we raise production errors early in the pipeline?

### User context 

Running the full pipeline in production currently takes 12 hours given the volume of data it processes. We are looking for options to allow data scientists and engineers to iterate faster on their code or modelling changes. 

The current infrastructure uses a Data Fabricator class to fabricate synthetic data following the same schema as production data. This synthetic data is then used in the test pipeline that can be run locally and used in the CI.

The Data Fabricator allows for capturing a wide range or errors, but it is not perfect. At various occasions it fails to detect errors due to drifts between:

1. Synthetic and real data 
2. Local and cloud environments setup

After discussing with Piotr and Laurens, these are the most common issues that seem to arise in production without being caught in the test pipeline:

| Issue | Importance | Description | Why is it not covered in tests |
| ------------- | ------------- | ------------- | ------------- |
| Normalization API | High | The normalization API can fail to normalize some ids and return a null. A few nulls can be handled, but too many and the modelling stage will fail. This usually happens 10 hours or so after the start of the pipeline, and can be caused by the parameters used to configure the API. | The test pipeline uses a mock of the normalization API. |
| Gap between fabricated data and production data | High | The biggest issue in that space is related to non-nullable columns in fabricated data, while being nullable in production. | Data fabricator parameters do not represent production data accurately. |
| Pre-processing pipeline | Medium | The pre-processing pipeline can fail with some manually typed data sources. | The test pipeline doesn't run pre-processing. |
| Kedro local Memory dataset | Medium | The production pipeline can fail as it cannot reach a memory dataset computed in a previous stage. | The test pipeline is sharing memory between stages, but the production environment is not. |
| Incorrect source files | Medium | Either the files can be outdated, or their file paths can be incorrect when running the production pipeline. | The test pipeline doesn't use the same files as production. |
| Out Of Memory | Low | Memory errors will make the production pipeline fail. They often arise in Neo4J. Restarting the pipeline usually solves the issue. | The test pipeline does cover out of memory failures. |

### Technical context

We are using the Kedro framework which easily allows to run specific nodes of the pipelines in different environments.

The data sources consists of knowledge graphs, tabular data or key value pairs. When testing with a subset of a knowledge graphs, as they comprise nodes connected by edges, we would like the subset to preserve the internal structure of it to be representative.

We would like to abstract the sampling of the data to allow for different strategies depending on the user's needs. Ideally, we would want data scientists to define their own strategies.

We would like to avoid persisting sampled data to stay away from dataset invalidation issues.

We would like to keep any test pipeline runtime below 30 minutes to allow for fast feedback and low cloud costs.

We would like to have the ability to sample production data at two key points a) before the DE stage and b) before the modelling stage.

## Decision

### Mitigation

1. Update data fabricator parameters to better represent production data (especially nullable columns). 
2. Write a script to semi-automatically update the fabricator's parameters based on production data.

### Short term

1. Run the pipeline locally using a sample of production data.
    * Following Laurens previous work, the simplest first step would be to add a sampling stage in the pipeline that can be enabled or disabled via a kedro flag.
    * Following Pascal's previous work, the sampling strategy needs to generate a representative sample of the production knowledge graph containing ground truth pairs.
2. Abstract out the sampling strategy so it can be passed as a parameter to the pipeline.
3. Raise errors currently happening in the modelling stage earlier in the pipeline via sanity checks in the Data Engineering stage.

### Long term 

1. Run the test pipeline in a pre-prod environment using a sample of production data.
3. Include the pre-processing stage in the test pipeline by fabricating data before it.
4. Do a sanity check on memory datasets used between stages, that would fail in prod.

## Consequences

These changes will allow users to fail faster on their code or modelling changes and will catch bugs before they reach production, hence improving productivity and faster feedback loops.
