---
title: Deep Dive
---

# Deep Dive into MATRIX

After completing first steps, you should be able to:

- Understand our Tech Stack on a high level
- Set up MATRIX environment on your machine
- Understand how config and kedro parameters work well together
- Run the Repurposing Pipeline with both fabricated data and sample of real data

!!! info
    To utilize our MATRIX system efficiently with real data, you will need get access to some of our internal tools such as Kubernetes Cluster and GCS Storage. In order to do that please create [onboarding issue](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E) so that we can assist you in the best possible way

In this section, we will walk through our tech stack, infrastructure, and pipeline in more detail.

First you will learn about **Core MATRIX Concepts** which are essential for comfortable development and understanding of our codebase. This will involve custom **Kedro Extensions** and how to work with the **Data Fabricator**. Then we will dive into **GCP environment** and how to set up your environment for a run with **real data** with Argo Workflows. Once you complete this section, you will have a good understanding how our kedro pipeline can be used across all environments.

Next, we will deep dive into our data products, including a **Data Catalog Walkthrough**, how to **introduce a new data source** and how we enforce schema using **Pandera for Schema Validation**. After this section you will gain a good understanding of our layered approach for data storing.

Once we covered the data section, we will go through our feature & embeddings pipeline and how we utilize **caching** and **embedding batching** to effectively process large datasets. 

We will then briefly revisit our modelling pipeline as an example to explain **how can one develop custom models** in concert with our Matrix Pipeline using some cool kedro functionalities. 

Once we've walked through the entire pipeline e2e, we will touch on **infrastructure**, how we are utilizing **Continuous Integration Pipeline** and how to go about **kubernetes development**. 

We realize that this section is long and our pipeline is quite complicated. However after completing this section, you should have a comprehensive understanding of the MATRIX system and be able to utilize the codebase and GCP effectively, develop, and contribute!

[Start with GCP Setup :material-arrow-right:](./gcp_setup.md){ .md-button .md-button--primary }