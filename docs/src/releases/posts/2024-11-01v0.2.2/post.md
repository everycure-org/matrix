---
draft: false 
date: 2024-11-01
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

# `v0.2.2`: Enhanced Data Integration, Robust Synonymization, and Advanced Evaluation

This release of the Matrix Platform introduces significant improvements to data integration, synonymization capabilities, and evaluation methodologies, enhancing the platform's ability to generate robust and
reliable drug repurposing predictions.

<!-- more -->

For a complete list of changes, please refer to the [GitHub release notes](https://github.com/everycure-org/matrix/releases/tag/v0.2.2).

## Key Enhancements 🚀

### 1. ROBOKOP Data Integration and RTX 2.10 Upgrade :database:

The platform now integrates the ROBOKOP knowledge graph, significantly broadening the knowledge base for drug repurposing analysis.  This integration, coupled with the upgrade to RTX 2.10, delivers:

- **Enhanced data coverage:** A richer and more up-to-date knowledge graph for broader insights.
- **Improved prediction accuracy:** Leveraging the combined power of RTX 2.10 and ROBOKOP enhances the reliability of predictions.

### 2. Robust Synonymization 🔗

The synonymization system has been completely overhauled, replacing the previous implementation with a more robust and accurate solution.  This upgrade ensures:

- **Improved data consistency:** More precise mapping of drug and disease synonyms for better data integration.
- **Enhanced data quality:** Reducing ambiguity and noise in the knowledge graph.
- **Preprocessing Pipeline Integration:** Synonymization is now part of the preprocessing pipeline, streamlining the data preparation process.


### 3. Advanced Evaluation Methodologies 📈

This release introduces key advancements in evaluation methodologies:

- **Time-split Validation:**  Evaluating model performance on data separated by time provides a more realistic assessment of predictive power and reduces the risk of overfitting.
- **Recall@N Metric:** This new metric offers a more granular performance analysis, enabling threshold-based evaluation and a deeper understanding of model behavior.
- **Node2Vec Integration:** Complementing the existing GraphSAGE implementation, Node2Vec enhances topological embedding generation, leading to more robust predictive modeling.
- **Redesigned Evaluation Framework:**  The evaluation framework has been redesigned to integrate seamlessly with the matrix generation pipeline and provide improved reporting, including matrix integration
and enhanced reporting.

### 4. Matrix Generation Pipeline 🚧

A new matrix generation pipeline has been implemented, providing a streamlined process for generating drug-disease prediction matrices. Key features include:

- **Enhanced reporting:** Detailed statistics and metrics provide greater transparency into the prediction process.
- **Drug-disease fabrication:** Facilitates the creation of synthetic data for testing and validation.
- **Drug-stratified train/test split:** A robust 8/1/1 train/test/validation split stratified by drug ensures balanced representation and reduces bias in model training.


### 5. Data Release Pipeline 📤

The introduction of a data release pipeline provides a structured process for managing and distributing knowledge graph snapshots, enabling experimentation with different data releases and improving
reproducibility. This also includes splitting the knowledge graph for experimental purposes.

### 6. Infrastructure and Tooling Enhancements 🛠

Several improvements streamline development and deployment workflows:

- **CI/CD Optimization:** CI jobs are now more efficient, triggering execution only on relevant path changes. The experiment pipeline also now uses `run_name` for consistency.
- **`make` and Onboarding Improvements:** Enhancements to the `make` targets and onboarding documentation streamline developer setup and project contribution.  A `make clean` target simplifies environment
cleanup.
- **Documentation Expansion:** Comprehensive documentation has been added for various aspects of the platform, including data science methodology, onboarding procedures, GraphSAGE comparison experiments,
Node2Vec experiment results, and running Argo Workflows locally.  The README now directs users to the documentation page for easier access.
- **Kedro Pinning:** Kedro 0.19.6 is now pinned to prevent a known runtime error, ensuring stability and reproducibility.
- **MLFlow Removal from Base Environment:** Streamlines the base environment by removing MLFlow, making it lighter and easier to manage.
- **Spark Integration for Node Embeddings:** Node embeddings processing has been migrated to Spark with custom endpoint integration, improving performance and scalability.
- **`--from-env` Kedro Option:**  Enables reading data from different environments using the `--from-env` option with the `kedro run` command, enhancing flexibility during development and testing.
- **Kedro Submit Feature:**  Simplifies end-to-end pipeline execution on the current branch using the `kedro submit` feature.
- **Local Argo Workflows:**  Facilitates local testing and debugging of Argo workflows.
- **Neo4J Enterprise License Keys:** Enables access to enterprise features of Neo4J.
- **Slack Notifications for Breaking Changes:**  Improves communication by sending Slack notifications when breaking changes are merged.



## Conclusion

This release marks a significant step forward for the Matrix Platform, with enhancements across data integration, synonymization, evaluation, and infrastructure.  These improvements provide a more robust,
efficient, and transparent platform for drug repurposing research.
