---
title: Home
hide: 
  - toc
  - navigation
---
<!-- NOTE: This file was partially generated using AI assistance. -->
# Matrix Project

Welcome! This documentation provides an introduction to the project and the codebase, and guides you through the process of setting up your local environment.

<div class="grid cards" markdown>

-   ![Developer](assets/img/hero.svg){: style="float: right; width: 150px; margin-left: 1em;" }

    __I'm a Developer__

    I want to contribute to the codebase.

    [:octicons-arrow-right-24: Get started locally](./getting_started/)

-   ![Researcher](assets/img/researcher.svg){: style="float: right; width: 150px; margin-left: 1em;" }

    __I'm a Researcher__

    I want to run the pipeline with real data.

    [:octicons-arrow-right-24: Access the data](./contribute/get_access_to_data/)

-   ![Engineer](assets/img/engineer.svg){: style="float: right; width: 150px; margin-left: 1em;" }

    __I'm an Engineer__

    I want to build a new platform based on this work.

    [:octicons-arrow-right-24: See infrastructure](./infrastructure/)

</div>

## What is Matrix?

MATRIX is Every Cure's main pipeline for generating high-accuracy predictions of drug-disease pairs using an "all vs all" approach. The pipeline:

- Ingests and integrates data from multiple sources
- Builds a knowledge graph
- Creates embeddings and trains models
- Makes predictions on potential drug-disease treatments
- Evaluates performance of the repurposing model

The output of our pipeline is a so-called matrix of approx <strong>60 million drug-disease pairs</strong> with corresponding treat scores which are then examined by our team of physicians.

We provide extensive documentation about both the pipeline itself and the infrastructure that powers it, from local development to production deployment in the cloud.

Sounds interesting? Here are instructions on [how to get started](./getting_started/index.md).

## Matrix pipeline overview

<img src="../../assets/getting_started/matrix_overview.png" alt="Matrix Pipeline Overview" style="width: 50%; margin: 2rem auto; display: block;">

Please note that the MATRIX project is still a work in progress. While we already actively use the pipeline for finding promising repurposing candidates, we are continuously improving it. 

If you discover any typos, bugs, or technical debt, please let us know through [issues](https://github.com/everycure-org/matrix/issues/new?template=bug_report.md) or through PRs. We appreciate all contributions and believe that by open-sourcing this work-in-progress repository, we can create a truly great drug repurposing platform.

## About Every Cure

Matrix is a part of the Every Cure pipeline. [Every Cure](https://everycure.org/about/) is a non-profit organization on a mission to unlock the full potential of existing medicines to treat every disease and every patient we possibly can. You can find more resources below about the Every Cure, Matrix Project and AI Drug Repurposing in Every Cure.

??? info "Every Cure videos"
    <iframe width="640" height="390" src="https://www.youtube.com/embed/3ElaCVvDZfI?si=lk3b1rSMutyiierm" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" referrerpolicy="strict-origin-when-cross-origin" allowfullscreen></iframe>