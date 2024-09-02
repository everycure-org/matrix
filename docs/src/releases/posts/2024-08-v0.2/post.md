---
draft: false 
date: 2024-09-02 
categories:
  - Release
authors:
  # note we randomly sort these
  - lvijnck
  - marcello-deluca
  - sabrinatoro
  - alexeistepa
  - piotrkan
  - matentzn
  - pascalwhoop
  - leelancashire
---

# `v0.2`: Functional E2E pipeline with first valuable results for Long Covid research

In this release, we've made significant strides in developing our drug repurposing
pipeline, enhancing its functionality, scalability, and accessibility. For a detailed list of contriubutions broken down by contributor check our [GitHub Release Page](https://github.com/everycure-org/matrix/releases/tag/v0.2)

Key achievements include:

1. Long COVID Analysis
    - We've completed an end-to-end run focused on Long COVID, generating top-performing drug predictions for 9 distinct LC subtypes. This marks a significant milestone in our ability to produce actionable insights for specific disease areas.
2. Knowledge Graph Enhancements
    - Released the first versions of our curated [disease list](https://github.com/everycure-org/matrix-disease-list/releases/tag/2024-07-25) and [drug list](https://github.com/everycure-org/matrix-drug-list/releases/tag/v1.0.3), providing a solid foundation for our repurposing efforts.
    - Integrated medical team data into our Knowledge Graph, establishing a prototype for incorporating non-KG custom data sources.
3. Infrastructure and Performance Improvements
    - Implemented pipeline parallelization using ephemeral Neo4j instances, significantly improving processing speed and resource utilization.
    - Enabled SSO-based access to MLFlow and internal tooling (e.g., Argo), enhancing security and user experience.
    - Expanded our Kubernetes cluster with larger node pools to handle increased workloads.
4. Collaboration and Data Integration
    - Introduced a `GoogleSheetsDataset` class, facilitating seamless collaboration between our medical and technical teams.
    - Developed a preprocessing pipeline to ingest experimental nodes and edges proposed by our medical team, enabling rapid hypothesis testing.
5. Model and Embedding Enhancements
    - Conducted several experiments to optimize parameters for our GraphSage model, including updates to learning rates and iteration counts.
    - Implemented and tested various Actor-Critic networks for MOA prediction and shortest path finding in our Knowledge Graph.
    - Initiated benchmarking of LLM embeddings for potential integration into our pipeline.
6. Documentation and Onboarding
    - Added a comprehensive introduction to our technology stack in the [Infrastructure section](../../../infrastructure/index.md) of our documentation.
    - Streamlined the onboarding process with improved guides and issue templates as well as video tutorials.
7. Technical Enhancements
    - Improved our CI/CD pipeline with enforced integration tests on every PR.
    - Enhanced our Argo workflow with Kedro node fusing for more efficient execution.
    - Implemented BigQuery table tagging with Git SHA for improved data lineage.


![](./attachments/medical-integration.excalidraw.svg)


This release represents a significant step forward in our ability to process, analyze,
and derive insights from complex medical data at scale. We're now better positioned to
tackle challenging repurposing scenarios and collaborate effectively across teams.

![](../../../assets/img/infra_intro/speed.excalidraw.svg)