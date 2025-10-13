# Drug List Migration: Matrix Impact Decisions

**Status**: Accepted
**Date**: 2025-10-10  
**Deciders**: Team Data
**Technical Story**: [EC-DATA603] - Matrix Forward Compatibility with New Drug List (Part of Drug List Migration Project)

## Context and Problem Statement

Current drug list used by Every Cure is of good quality however because of its NLP-based nature, it contains some noise which results to drug entites mapping to different CURIEs over time. Because we use those CURIEs as primary identifiers for the list, it leads to issues in the 'spine' of matrix. This in addition to the non-deterministic nature of LLM-based extraction is harmful for keeping historical records of matrix predictions which Every Cure Medical team examines.

## Decision

We are migrating from a text-mined drug list to a manually curated drug list to eliminate NLP-based noise and ensure mapping to IDs is maintained by EC team. This new drug list will have unique EC identifiers which will be stable over time (i.e. identifiers would not be removed/modified, only appended if needed).
We have two options to implement the changes in matrix
* A) We keep NCATS (KG) node CURIEs through the pipeline and join with the drug list (mapping between EC id and kg node id) before exporting the list for medical team examination.
  * This option requires us re-map primary identifiers upon ingestion & keep/join EC ids after matrix generation
* B) We modify the matrix pipeline & KG releases to utilize EC ids through nodes in the pipeline.
  * This option requires us to modify existing KG curies, 'break' biolink model and change the evaluation pipeline to utilize correct identifiers.

As we want to make as little changes to the matrix pipeline as possible, we decide that option A) is the most suitable.

## Impact
As new, manually curated list leverages different unique identifiers than MATRIX, we need to rewire ID mapping process within the matrix pipeline. The following parts of the pipeline were impacted:

### Ingestion & Drug Transformation

Previously ingestion pipeline accepted single ID representing curies which was primary identifier for each drug. Now id primary key represents EC identifiers (with `EC:` prefix) and complementary `translator_id` are generated, representing CURIEs that can be mapped to our KGs. We implemented unique pandera check for both ID and translator ID.
This change enables the data engineering pipeline to function as it did previously.

![ingestion_changes](../../assets/pipeline/ingestion_changes.svg)

### Matrix Generation

Previously matrix pairs were generated using only source & target CURIEs. While the matrix generation process is not changing, we are keeping EC identifier to each source ID with the `ec_drug_id` identifier. This ena
This change enables the evaluation pipeline to function as it did previously while enabling orchard/medical team to navigate through the matrix list using curated identifiers.

![matrix_generation](../../assets/pipeline/matrix_generation_changes.svg)

## References
[Pull Request with Changes](https://github.com/everycure-org/matrix/pull/1885)
