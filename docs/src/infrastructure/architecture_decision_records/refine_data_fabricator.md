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


* Use fabricated data to run tests locally and catch most of the issues easily, but accept that it does't look like cloud data.
* Leverage Kedro's flexibility to allow users to run parts of the pipeline locally using a sample of the cloud data to debug their issue locally.

Looking at the sampling in detail: sampling strategies should be flexible to allow for different user scenarios, for example replicating a bug, performance testing or model testing. This could be configurabled via a flag in Kedro, that would then pull the right data from the cloud environment.

## Consequences

This change should allow users to iterate faster on their code or modelling changes, hence a gain in productivity, happiness and less cloud costs for debugging.