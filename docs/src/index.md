---
title: Home
hide: 
  - toc
  - navigation
---
# Matrix Project
!!! note "Every Cure"
    [Every Cure](https://everycure.org/about/) is a non-profit organization on a mission to unlock the full potential of existing medicines to treat every disease and every patient we possibly can. We are leveraging our AI Platform MATRIX to find new applicatiions of existing medicine. Here you can learn more about [AI Drug Repurposing](https://www.youtube.com/embed/67_Z40Ap1pU?si=XlCu7fBHxxkBTchH) and [Matrix Project](https://www.youtube.com/embed/3ElaCVvDZfI?si=lk3b1rSMutyiierm). 

Welcome! This documentation provides an introduction to the project and the codebase, and guides you through the process of setting up your local environment.

!!! info
    Please note that the MATRIX project is still a work in progress. While we already actively use the pipeline for finding promising repurposing candidates, we are continuously improving it. If you discover any typos, bugs, or technical debt, please let us know through [issues](https://github.com/everycure-org/matrix/issues/new?template=bug_report.md) or through PRs. We appreciate all contributions and believe that by open-sourcing this work-in-progress repository, we can create a truly great drug repurposing platform.

<div style="display: flex; align-items: center; gap: 2rem;">
    <div style="flex: 1;">
        </br>
        <p><str>What is Matrix?</str></p>
        <p> MATRIX is Every Cure's main pipeline for generating high-accuracy predictions of drug-disease pairs using an "all vs all" approach. The pipeline:</p>
        <ul>
            <li>Ingests and integrates data from multiple sources</li>
            <li>Builds a knowledge graph</li>
            <li>Creates embeddings and trains models</li>
            <li>Makes predictions on potential drug-disease treatments</li>
            <li>Evaluates performance of the repurposing model</li>
        </ul>
        <p>The output of our pipeline is a so-called matrix of approx <strong>60 million drug-disease pairs</strong> with corresponding treat scores which are then examined by our team of physicians.</p>
        <p>We provide extensive documentation about both the pipeline itself and the infrastructure that powers it, from local development to production deployment on cloud architecture.</p>
        <p>Sounds interesting? Here are instructions on <a href="./getting_started/index.md">how to get started</a>.</p>

    </div>
    <div style="flex: 1;">
        <img src="../../assets/getting_started/matrix_overview.png" alt="Matrix Pipeline Overview" style="width: 100%;">
    </div>
</div>