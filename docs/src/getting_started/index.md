---
title: Getting Started
---

# Welcome to MATRIX

Welcome to the Matrix onboarding guide! This guide provides an introduction to the project and the codebase, and guides you through the process of setting up your local environment.

## What is MATRIX?

MATRIX is Every Cure's main pipeline for generating high-accuracy predictions of drug-disease pairs using an "all vs all" approach. The pipeline:

- Ingests and integrates data from multiple sources
- Builds a knowledge graph
- Creates embeddings and trains models
- Makes predictions on potential drug-disease treatments

!!! info "Contributing"
    We are aiming to constantly improve our documentation and codebase. If you spot any typos/mistakes, please let us know _via_ GitHub Issues or create a PR correcting those changes. We always appreciate external contributions!

!!! tip "Navigation"
    Start with the First Steps section to learn about our tech stack and set up your local environment. Then proceed to the Deep Dive section to understand our pipeline in detail.

## Who is this guide for? 

We envision there are three types of users and contributors who might be interested in this repository:

* **Pipeline Contributors** - Researchers or data scientists who want to use our drug repurposing pipeline locally or on their own infrastructure to experiment with different models or with their own data sources.

* **Project Contributors** - Community members who want to improve the core project and support Every Cure through novel algorithms, better code quality, test coverage, or new features and capabilities.

* **System Adapters** - Organizations or teams looking to adapt our infrastructure to their own cloud or on-premise environments, deploying on different cloud providers and customizing for specific compliance requirements.

We welcome all contributors however this guide is primarily designed for pipeline and project contributors. If you are interested in the infrastructure side, we would recommend completing this guide but focusing mainly on the [infrastructure](../infrastructure/index.md) part of the documentation.

As you onboard, we really recommend creating an [onboarding issue](https://github.com/everycure-org/matrix/issues/new?assignees=&labels=onboarding&projects=&template=onboarding.md&title=%3Cfirstname%3E+%3Clastname%3E) to help you navigate your progress. This way we can also support you in case of any bottlenecks or errors.
    

[Get Started with First Steps :material-arrow-right:](./first_steps/){ .md-button .md-button--primary }
     