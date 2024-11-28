---
draft: false 
date: 2024-11-01
categories:
  - Release
authors:
  - lvijnck
  - kushal2051
  - emil-k
  - matwasilewski
  - oliverw1
  - pascalwhoop
  - piotrkan


---

# `v0.2.5`: First stable integrated KG release, improving ROBOKOP integration, first version of Matrix CLI, and enhancing pipeline control

This release of the Matrix Platform focuses on improving developer experience and enhancing data integration and pipeline control.  We introduce a new command-line interface (CLI), integrate the ROBOKOP
knowledge graph, and provide more granular control over pipeline execution.

## How to get access to the data published in this release

### Kedro CLI

TLDR
`kedro ipython --env cloud`


### In a Jupyter Notebook
TODO

### Via BigQuery
TODO

### When developing a new kedro node 

TODO

<!-- more -->

For a complete list of changes, please refer to the [GitHub release notes](https://github.com/everycure-org/matrix/releases/tag/v0.2.5).

## Key Enhancements 🚀

### 1. Matrix CLI ⌨️

A new command-line interface, `matrix-cli`, streamlines various project tasks. The CLI offers a centralized interface for managing GitHub users and teams, generating release notes with AI assistance, and
summarizing code changes. This simplifies common workflows and improves developer productivity.

### 2. ROBOKOP Knowledge Graph Integration 🌐

The platform now integrates the ROBOKOP knowledge graph, significantly expanding the scope and depth of our knowledge base. This integration enhances data coverage and provides richer context for drug
repurposing analysis, potentially leading to more accurate and comprehensive predictions.

### 3. Translator-Based Synonymization Enhancements 🔄

The synonymization system, now powered by the [Translator project](https://nodenorm.test.transltr.io/docs#/translator), has undergone significant enhancements. These improvements ensure greater consistency and scalability in mapping drug and disease synonyms,
improving the overall quality and reliability of data integration by more accurately finding identifier equivalences.

### 4. GPU Support ⚡️

The platform now leverages GPUs on the cluster, accelerating computationally intensive tasks.  Additionally, the integration of Neo4j Enterprise license keys unlocks advanced features and performance
optimizations for graph database operations. These infrastructure upgrades significantly improve the platform's efficiency and scalability.  See issue #622 for more details.

### 5. Enhanced `kedro submit` Command  ⚙️

The `kedro submit` command has been significantly enhanced, providing developers with more fine-grained control over pipeline execution. It now supports running pipelines from specific nodes and submitting
to different buckets (test/release), enabling greater flexibility in development, testing, and deployment workflows. See issues #605 and #611 for more details.

### 6. Kedro Catalog Cleanup 🧹

Addressing technical debt, this release includes fixes for missing and unused entries in the Kedro catalog (issue #600).  This cleanup improves pipeline reliability and maintainability.  Further improvements
to the catalog include fixes related to cloud globals (issue #694) and more robust handling of node category selection during integration (issue #654).

### 7. Enhanced Developer Experience 🧰

Several improvements streamline the developer experience:

- **Argo Workflow Submission Enhancements:** Improved Argo workflow submission (issue #565) simplifies pipeline orchestration and execution.
- **BigQuery Integration Enhancements:**  Improvements to BigQuery integration through GCS filesystem datasets and external table registration (issue #563) streamline data loading and access.
- **Developer Namespaces:**  Introduction of developer namespaces (issue #605) improves resource isolation and facilitates parallel development efforts.  This is accompanied by a fix to a critical issue in
the Neo4j template (issue #684).
- **BTE-TRAPI & Redis Integration:** The BTE-TRAPI deployment now integrates Redis for enhanced performance and caching (issue #605).
- **ArgoNode Wrapper:**  The introduction of the ArgoNode wrapper (issue #626) allows for greater customization of task execution and resource allocation within Argo Workflows.
- **Partitioned Dataset Embeddings:**  Implementation of efficient embedding computation using PartitionedDataset, batch processing, and LangChain integration (issue #642) improves performance for
large-scale embedding generation.

### 8. Ongoing Modelling Workstreams

We have progressed the following workstreams: 

 - Successfully completed a first run of our supervised mechanism of action algorithm. This should provide greater transparency into our model predictions ([#510](https://github.com/everycure-org/matrix/pull/510). We are also building an MVP for visualizing these MoA paths. 
 -  We have developed a first version of adding timestamps to the edges, and now have ~27% timestamped. This should enable us to execute more robust time-split-validation experiments in the future. ([#588](https://github.com/everycure-org/matrix/issues/588))
 - Work to compare performance of existing models with TxGNN has made significant progress and our first experimental runs are now complete. Ongoing work will compare this method with our baseline KGML-xDTD approach. ([#586](https://github.com/everycure-org/matrix/issues/586))
- We now have the ability to perform multiple folds of cross validation in the modeling and evaluation suite. This should enable us to better estimate stability and confidence in our model predictions ([#587](https://github.com/everycure-org/matrix/issues/587))
- We have implemented the ability to run a full comparison of treat scores using various embedding models, such as Node2Vec, PubmedBERT, and OpenAI ([#301](https://github.com/everycure-org/matrix/issues/301).




## Next Steps 🔮

We are continuously working to improve the Matrix Platform. Our next steps include further enhancements to the data release pipeline for fully automated releases, ongoing integration of new knowledge graphs,
exploring self-hosted LLMs for embedding generation, and preparing for open-sourcing the repository.