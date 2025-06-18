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

* Pipeline Contributors - These are researchers, data scientists, or developers interested in utilizing our drug repurposing pipeline, either locally or on their own research infrastructure. They may want to:
    * Experiment with different embedding models or ML architectures
    * Add new data sources to enhance predictions
    * Customize pipeline components for their specific use cases
    * Validate our approach on their own datasets

* System Adapters - These are typically organizations or teams looking to adapt our infrastructure to their own cloud or on-premise environments. They may be interested in:
    * Deploying the pipeline on different cloud providers (AWS, Azure, etc.)
    * Integrating with existing enterprise systems
    * Customizing infrastructure for specific compliance requirements
    * Optimizing resource utilization and costs

* Project Contributors - These are community members who want to improve the core project itself. Their contributions might include:
    * Enhancing prediction accuracy through novel algorithms
    * Improving code quality and test coverage
    * Adding new features and capabilities
    * Fixing bugs and addressing technical debt
    * Enhancing documentation and examples
    * Contributing domain expertise in pharmacology or medicine


[Get Started with First Steps :material-arrow-right:](./first_steps/){ .md-button .md-button--primary }
     