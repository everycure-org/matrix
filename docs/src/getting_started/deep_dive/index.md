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

First we will start with **GCP Setup** to understand how to configure your environment for cloud operations, then explore **GCP Environments** to learn about our cloud infrastructure and deployment strategies.

Then we will dive into **Kedro Extensions** to understand our custom pipeline components and how they enhance the standard Kedro framework. Learn about **Kedro Experiment** management for tracking and reproducing your work. There is also an optional section which builds on environment overview in )_first steps_ where we will cover our comprehensive environment system:
- **Base Environment**: Foundation configuration shared across all environments
- **Test Environment**: Fast testing with synthetic data and optimized parameters
- **Sample Environment**: Development with smaller real datasets
- **Cloud Environment**: Production-scale execution on GCP


At the end of the section we will learn how to use **Jupyter** with Kedro for interactive development and data exploration. Understand **Cross Environment** workflows for debugging and troubleshooting production issues.

The section has links to practical **walkthroughs** that demonstrate real-world usage patterns and best practices for working with the MATRIX pipeline.

We realize that this section is comprehensive and our pipeline is quite sophisticated. However, after completing this section, you should have a thorough understanding of the MATRIX system and be able to utilize the codebase and GCP effectively, develop, and contribute!

[Start with GCP Setup :material-arrow-right:](./gcp_setup.md){ .md-button .md-button--primary }