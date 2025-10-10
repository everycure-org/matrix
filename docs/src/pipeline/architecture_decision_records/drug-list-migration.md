# ADR: CI Optimization with GitHub Actions Self-Hosted Runners

**Status**: Accepted  
**Date**: 2025-10-10  
**Deciders**: Team Data
**Technical Story**: [EC-DATA603] - Matrix Forward Compatibility with New Drug List (Part of Drug List Migration Project)

## Context and Problem Statement

We are 


## Decision

We decided to make as little changes to the matrix pipeline as possible. 

## Impact

The following parts of the pipeline were impacted:

### Ingestion & Drug Transformation

![ingestion_changes](/Users/piotrkaniewski/work/matrix/docs/src/assets/pipeline/ingestion_changes.svg)

### Matrix Generation

Now we are no longer gener

![matrix_generation](/Users/piotrkaniewski/work/matrix/docs/src/assets/pipeline/matrix_generation_changes.svg)

## References
[Pull Request with Changes](https://github.com/everycure-org/matrix/pull/1885)



---

_This ADR represents a significant infrastructure investment that successfully achieved our performance goals while maintaining security and cost efficiency._
