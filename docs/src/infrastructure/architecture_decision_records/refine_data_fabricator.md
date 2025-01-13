---
title: Refine Data Fabricator
--- 

## Status

Under review

## Context

### User context 

Running the full pipeline in production currently takes 12 hours given the volume of data it processes. We are looking for options to allow data scientists and engineers to iterate faster on their code or modelling changes. 

The current infrastructure uses a Data Fabricator class to fabricate synthetic data following the same schema as production data. This synthetic data is then used in a test pipeline that can be run locally and used in integration tests.

The Data Fabricator allows for capturing a wide range or errors, but it is not perfect. At various occasions it fails to detect errors due to the drift between synthetic and real data. 

After discussing with Piotr, these are the most common issues that seem to arise in production without being caught in the test pipeline:

| Issue | Importance | Description | Why is it not covered in tests |
| ------------- | ------------- | ------------- | ------------- |
| Normalization API | High | The normalization API can fail to normalize some ids an return a null. A few nulls can be handled, but too many and the modelling stage can fail after 10 hours or so after the start of the pipeline. This can be caused by the parameters used to configure the API. | The test pipeline uses a mock of the normalization API. |
| Pre-processing pipeline | Medium | The pre-processing pipeline can fail with some manually typed data sources. | The test pipeline doesn't run pre-processing. |    
| Kedro local Memory dataset | Medium | The production pipeline can fail as it cannot reach a memory dataset computed in a previous stage. | The test pipeline is sharing memory between stages, but the production environment is not. |
| Files in the wrong location | Medium | File path can be incorrect when running the production pipeline. | The test pipeline doesn't use the same file paths as production. |
| Out Of Memory | Low | Memory errors will make the production pipeline fail. They often arise in Neo4J. Restarting the pipeline usually solves the issue. | The test pipeline does not use enough data to run out of memory. |


### Technical context

We are using the Kedro framework which easily allows to run specific nodes of the pipelines in different environments.

The data sources consists of knowledge graphs, tabular data or key value pairs. When working with knowledge graphs, as they comprise nodes connected by edges, we would like to ensure that the subset of a graph preserves the internal structure of it to be representative.

We would like to abstract the sampling of the data to allow for different strategies depending on the user's needs. Ideally, we would want data scientists to define their own strategies.

We would like to avoid storing subsampled data to stay away from dataset invalidation issues.

We would like to keep any test pipeline runtime below 30 minutes to allow for fast feedback and low cloud costs.

We would like to have the ability to sample production data at any stage of the pipeline, allowing for faster iteration times.

### Problems

1. How can we catch more errors before reaching production, in local testing and CI?
2. How can we allow data scientists to run pipelines on subsets of the production data?
3. How can we push production errors to happen as early as possible?

## Decision

### Mitigation

TBD

### Short term

* Use fabricated data to run tests locally and catch most of the issues easily, but accept that it doesn't look like cloud data.
* Leverage Kedro's flexibility to allow users to run parts of the pipeline locally using a sample of the cloud data to debug their issue locally.

The easiest sampling strategy is to pick random rows from the input using a threshold, however we want the results or errors to be reproducible, therefore we should avoid random sampling strategies, unless they can be made deterministic via a seed.

In the medium term, we could develop flexible sampling strategies to allow for different user scenarios, for example sampling precisely to replicate a bug, or to test a model against specific data.

Sampling strategies could be configured via a flag in `kedro run`, it would then pull the right data from the cloud environment and be processed locally.


### Long term

We could develop curated persisted datasets of different sizes, extracted from the production data. They would provide an intermediate step between fabricated data and production one, and allow to measure and compare models performance. These datasets would be based on the medical and data science teams' knowledge.

## Consequences

This change should allow users to iterate faster on their code or modelling changes, hence a gain in productivity, happiness and less cloud costs for debugging.
