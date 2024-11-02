---
draft: false 
date: 2024-09-02 
categories:
  - Release
authors:

  - RobynMac25
  - alexeistepa
  - chempsEC
  - drhodesbrc
  - eding36
  - elliottsharp
  - james0032
  - leelancashire
  - lvijnck
  - marcello-deluca
  - matwasilewski
  - may-lim
  - pascalwhoop
  - piotrkan
  - redst4r

---

# Matrix Platform Release: Enhanced Data Integration, Inference Capabilities, and Scalability

This release of the Matrix Platform delivers significant improvements across data integration, model evaluation, and infrastructure, enabling more efficient and comprehensive drug repurposing analysis.  Key
highlights include the integration of the Robokop knowledge graph, a new inference pipeline for rapid hypothesis testing, enhanced Neo4j capabilities for improved scalability, and the addition of new
evaluation metrics.

<!-- more -->

For a complete list of changes, please refer to the [GitHub release notes](https://github.com/everycure-org/matrix/releases/tag/v[Insert Version Number Here]).

## Key Enhancements

### 1. Robokop Data Integration: Expanding the Knowledge Base

We've integrated the Robokop knowledge graph into our ingestion pipeline, substantially expanding the dataset used for drug repurposing analysis. This integration provides a richer source of information,
leading to more comprehensive and nuanced predictions.  Robokop's unique data points enrich our existing knowledge graph, improving the accuracy and reliability of our models.


### 2. Streamlined Hypothesis Testing with the Inference Pipeline

A new inference pipeline has been implemented to generate drug-disease predictions based on requests from the medical team.  This significantly streamlines the hypothesis testing process, facilitating rapid
data analysis and accelerating the drug repurposing workflow. The pipeline allows for efficient querying of the enriched knowledge graph and returns predictions based on the most up-to-date models.

### 3. Enhanced Model Evaluation:  Recall@N and Node2Vec

To further improve our performance analysis, we've added the Recall@N evaluation metric to our pipeline. This metric provides a more granular understanding of model performance at different ranking
thresholds.  Additionally, we've integrated the Node2Vec algorithm alongside GraphSage for generating topological embeddings, providing alternative approaches to predictive modeling and enhancing the
robustness of our analysis.

### 4. Scalable Neo4j Infrastructure: Addressing Performance Bottlenecks

Significant infrastructure improvements have been made to address performance bottlenecks within Neo4j.  By converting Neo4j into an ephemeral compute unit scoped to each pipeline run, we now enable
concurrent execution of multiple pipelines within the cluster. This is complemented by a new node fusing algorithm in our Argo workflows, which efficiently executes multiple Kedro nodes on the same
Neo4j-enabled machine, optimizing resource utilization and reducing execution times. This effectively transforms Neo4j from a shared resource constraint into a per-job requirement, unlocking significant
scalability gains.


### 5. Enhanced Reporting and Data Science Methodology Transparency

A new matrix generation pipeline provides enhanced report generation with detailed statistics and metadata, making it easier to interpret results.  Furthermore, comprehensive documentation has been added
detailing our data science methodology for drug repurposing, including a deep dive into our evaluation suite and a walkthrough of new dataset integration.  This enhanced transparency improves understanding
and collaboration among team members and stakeholders.


## Technical Improvements and Bug Fixes

This release also includes numerous technical enhancements and bug fixes, improving the stability, efficiency, and cross-platform compatibility of the Matrix Platform. Notable improvements include:  sparse
checkout in the CI workflow, cross-platform Docker Compose support, flexible environment data transfers, enhanced disk space management, improved model tuning, and robust retry logic for Docker Compose
shutdown.  A detailed list of bug fixes is available in the release notes.


## Conclusion

This release represents a significant step forward for the Matrix Platform. The enhancements to data integration, inference capabilities, and infrastructure lay the foundation for more efficient and
effective drug repurposing research.  The improvements in model evaluation and the added transparency via enhanced documentation will contribute to more robust and reproducible results.  We are excited to
continue building upon this progress in future releases.