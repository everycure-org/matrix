---
title: Deep Dive
---

# Deep Dive into MATRIX

!!! info
    To contribute to the MATRIX project and utilize our MATRIX system efficiently with real data, with our deployed infrastructure, you will need get access to some of our internal tools such as Kubernetes Cluster and GCS Storage. In order to do that please create [onboarding issue](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E) so that we can assist you in the best possible way

This section covers:

- **GCP Setup & Environments**: Configure cloud environment and understand infrastructure
- **Kedro Extensions**: Custom pipeline components and framework enhancements
- **Kedro Experiment**: Tracking and reproducing experiments
- **Environment Types**:
    * **Base**: Shared foundation configuration
    * **Test**: Fast testing with synthetic data
    * **Sample**: Development with smaller real datasets  
    * **Cloud**: Production-scale GCP execution
- **Jupyter Integration**: Interactive development with Kedro in a notebook
- **Cross Environment Workflows**: Debug and troubleshoot across environments
- **Practical Walkthroughs**: Real-world usage patterns and best practices

We recommend starting with GCP Setup to establish your cloud environment.

[Start with GCP Setup :material-arrow-right:](./gcp_setup.md){ .md-button .md-button--primary }