---
title: Refine Data Fabricator
--- 

## Status

_Under review_

## Context

Running the full pipeline in production takes currently 12 hours given the volume of data it processes. We are looking for options to allow data scientists and engineers to iterate faster on their code or modelling changes. 

The choice that was previously made involves a Data Fabricator class that will fabricate data that follows the same schema as the cloud data. This synthetic data is then used in the test pipeline.

The main pain point users currently have is to see their changes running succesfully in the test pipeline, but failing in the cloud one. Users are failing to replicate the error locally, and have to rely on the cloud pipeline to debug the issue. Looking further, it might also be useful for data scientists to test their models on a specific subset of the full data locally, to check that they work as expected.

The Kedro framework easily allows to run specific nodes of the pipelines in different environments.


## Decision

__In the short term__

* Use fabricated data to run tests locally and catch most of the issues easily, but accept that it doesn't look like cloud data.
* Leverage Kedro's flexibility to allow users to run parts of the pipeline locally using a sample of the cloud data to debug their issue locally.

The easiest sampling strategy is to pick random rows from the input using a threshold, however we want the results or errors to be reproducible, therefore we should avoid random sampling strategies, unless they can be made deterministic via a seed.

In the medium term, we could develop flexible sampling strategies to allow for different user scenarios, for example sampling precisely to replicate a bug, or to test a model against specific data.

Sampling strategies could be configured via a flag in `kedro run`, it would then pull the right data from the cloud environment and be processed locally.


__In the long term__

We could develop curated persisted datasets of different sizes, extracted from the production data. They would provide an intermediate step between fabricated data and production one, and allow to measure and compare models performance. These datasets would be based on the medical and data science teams' knowledge.

## Consequences

This change should allow users to iterate faster on their code or modelling changes, hence a gain in productivity, happiness and less cloud costs for debugging.
