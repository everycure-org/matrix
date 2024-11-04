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

This release of the Matrix Platform introduces improvements to data integration,
synonymization capabilities, and evaluation methodologies, enhancing the platform's
ability to generate drug repurposing predictions based on different selections of base data.
It also allows the ad-hoc generation of predictions for specific drugs or diseases.

<!-- more -->

For a complete list of changes, please refer to the [GitHub release notes](https://github.com/everycure-org/matrix/releases/tag/v0.2.2).

## Key Enhancements 🚀

### 1. ROBOKOP Data Integration and RTX 2.10 Upgrade :material-database:

The platform now integrates the ROBOKOP knowledge graph, significantly broadening the
knowledge base for drug repurposing analysis.  This integration, coupled with the upgrade
to RTX 2.10, delivers:

- **Enhanced data coverage:** A richer and more up-to-date knowledge graph for broader insights.
- **Improved prediction accuracy:** Leveraging the combined power of RTX 2.10 and ROBOKOP enhances the reliability of predictions.
- **Standardized release in BigQuery:** The Every Cure KG will now be released as a dataset in [BigQuery](https://console.cloud.google.com/bigquery?project=mtrx-hub-dev-3of&ws=!1m4!1m3!3m2!1smtrx-hub-dev-3of!2srelease_v0_2_2). [^1]



### 2. Translator project based synonymization 🔗

The synonymization system has been overhauled, replacing the previous ARAX based implementation with a more scalable solution hosted by the translator project.

- **Improved data consistency:** More precise mapping of drug and disease synonyms for better data integration. The API is able to synonymize more nodes and node types across all KGs. [^2]
- **Preprocessing Pipeline Integration:** Synonymization is now part of the preprocessing pipeline, streamlining the data preparation process.
- **Synonymization of drug, disease and medical inputs**: We aligned the synonymization across the pipeline, all leveraging the same API now.

### 3. Advanced Evaluation Methodologies 📈

This release introduces key advancements in evaluation methodologies:

- **Time-split Validation:**  Evaluating model performance on data separated by time provides a more realistic assessment of predictive power and reduces the risk of overfitting.
- **Recall@N Metric:** This new metric provides a more granular performance analysis by evaluating the model's ability to retrieve relevant drug-disease pairs within the top N predictions. This threshold-independent based evaluation metric offers a deeper understanding of model behavior in ranking and recommendations.
- **Node2Vec Integration:** Complementing the existing GraphSAGE implementation, Node2Vec enhances topological embedding generation. Node2Vec utilizes random walks to capture the structural roles of nodes in a graph, potentially leading to more robust predictive modeling.
- **Redesigned Evaluation Framework:**  The evaluation framework has been redesigned to integrate seamlessly with the matrix generation pipeline and provide improved reporting, including matrix integration
and enhanced reporting.

### 4. Matrix Generation Pipeline 🚧

A new matrix generation pipeline has been implemented, providing a streamlined process for generating drug-disease prediction matrices. Key features include:
- Enhanced evaluation with Recall@N and Hit@K Metrics: The pipeline now incorporates Recall@N and Hit@K metrics to provide comprehensive summaries for full matrix predictions. These metrics offer a more detailed evaluation of model performance by assessing the accuracy of predictions within the top N or K results. 
- The most recent Node2Vec algorithm has been integrated to enhance topological embedding generation. This improvement leads to more robust predictive modeling by better capturing the structural relationships within the data. More details of run [here](https://mlflow.platform.dev.everycure.org/#/experiments/115/runs/f50acfac0b1e4a76964610910dab5bc0), with example Hit@N metric [here](https://mlflow.platform.dev.everycure.org/#/metric?runs=%5B%22f50acfac0b1e4a76964610910dab5bc0%22%5D&metric=%22rf.disease_specific_ranking_hit-100%22&experiments=%5B%22115%22%5D&plot_metric_keys=%5B%22rf.disease_specific_ranking_hit-100%22%5D&plot_layout=%7B%22autosize%22:true,%22xaxis%22:%7B%7D,%22yaxis%22:%7B%7D%7D&x_axis=relative&y_axis_scale=linear&line_smoothness=1&show_point=false&deselected_curves=%5B%5D&last_linear_y_axis_range=%5B%5D)

- **Enhanced reporting:** [Detailed statistics](https://mlflow.platform.dev.everycure.org/#/experiments/115/runs/f50acfac0b1e4a76964610910dab5bc0) and metrics provide greater transparency into the prediction process.
- **Drug-stratified train/test split:** A robust 8/1/1 train/test/validation split stratified by drug ensures balanced representation and reduces bias in model training.

### 5. Ad-hoc Prediction Generation 🧠

A new inference pipeline has been implemented to support on-demand drug repurposing predictions through a user-friendly Google Sheets interface. This pipeline enables:

- **Flexible Prediction Types:**
  - Drug-centric predictions: Generate predictions for a specific drug against all diseases
  - Disease-centric predictions: Generate predictions for a specific disease against all drugs
  - Drug-disease specific predictions: Generate targeted predictions for specific drug-disease pairs

- **Automated Workflow:**
  - Seamless integration with the matrix generation pipeline
  - Automated synonymization and normalization of user inputs
  - Support for multiple model predictions in parallel
  - Real-time visualization of prediction distributions

- **User-Friendly Interface:**
  - Input submission through Google Sheets
  - Automated result reporting and visualization
  - Integration with our versioned drug and disease lists
  - Support for batch processing of prediction requests

While this feature is still dependent on our manual pipeline execution, we aim to make this automated so that it can be run on-demand for specific drugs or diseases.

### 6. Data Release Pipeline 📤

The introduction of a data release pipeline provides a structured process for managing
and distributing knowledge graph snapshots, enabling experimentation with different data
releases and improving reproducibility. This also splitting the pipeline into two, 
allowing `n` experiments linked to a single KG release.

This pipeline is still under development and we aim to fully automate KG snapshot
releases over the next 2 months to reach a weekly patch and monthly minor release
cadence using fully automated workflows.

### 7. Infrastructure and Tooling Enhancements 🛠

Several improvements streamline development and deployment workflows based on continous feedback from everyone in the Matrix community:

- **`make` and Onboarding Improvements:** Enhancements to the `make` targets and onboarding documentation streamline developer setup and project contribution.  A `make clean` target simplifies environment cleanup.
- **Documentation Expansion:** Comprehensive documentation has been added for various aspects of the platform, including data science methodology, onboarding procedures, GraphSAGE comparison experiments and Node2Vec experiment results.
- **Running one node locally with full cloud data:** `--from-env cloud` Enables reading data from specific environments as part of the `kedro run` command, enhancing flexibility during development and testing. This allows developers to test just their changes on full data locally, essentially "forking" off the cloud environment. [(Documentation)](../../../onboarding/local-setup.md#plugging-into-cloud-outputs)
- **Kedro Submit Feature:**  Simplifies end-to-end pipeline execution on the current branch using the `kedro submit` feature. This feature will become more powerful going forward, allowing developers to submit workflows from their machine to our shared cluster for execution. [(Documentation)](../../../infrastructure/runbooks/03_run_pipeline_from_branch.md) 
- **Neo4J Enterprise License Keys:** Enables access to enterprise features of Neo4J. We leverage this mainly for hosting several KG versions on a single server as well as using many CPU cores for the GDS library which is usually limited to a parallelism of 4.

## Next Steps 🔮

In the coming months we will focus on [^3]:

### Data Workstreams

- Fully automating the data release pipeline to reach a weekly patch and monthly minor release cadence. [(#612)](https://github.com/everycure-org/matrix/issues/612) 
- Ingesting and integrating the SPOKE KG into the platform. [#486](https://github.com/everycure-org/matrix/issues/486)
- Implement a strategy pattern to allow for selection of the synonymization strategy and adding a vector similarity based synonymization strategy. ([Milestone](https://github.com/everycure-org/matrix/milestone/23))
- Build a Data Metrics dashboard based on [evidence.dev](https://evidence.dev) and share statistics about our data sources as well as our data releases. ([Milestone](https://github.com/everycure-org/matrix/milestone/21))
- Implement a series of data quality tests and implement filtering steps to remove low quality data. ([#516](https://github.com/everycure-org/matrix/issues/516))

### Modelling Workstreams

- Implement a first mechanism of action algorithm as part of the pipeline. ([#476](https://github.com/everycure-org/matrix/issues/476))
- Implement first version of timestamped edges and execute time-split-validation experiment. ([#588](https://github.com/everycure-org/matrix/issues/588))
- Compare performance of existing models with TxGNN. ([#586](https://github.com/everycure-org/matrix/issues/586))
- Incorporate K fold cross validation into evaluation suite. ([#587](https://github.com/everycure-org/matrix/issues/587))

### Platform Workstreams

- Add GPUs to the cluster and enable select kedro nodes to run on GPUs. ([#77](https://github.com/everycure-org/matrix/issues/77))
- Implement self hosted LLM and embedding system to process node embeddings and enable LLM usage in the pipeline at scale ([Milestone](https://github.com/everycure-org/matrix/milestone/22))
- [Prepare open sourcing of the repository](https://github.com/everycure-org/matrix/issues?q=is%3Aopen+is%3Aissue+milestone%3A%22Open+Source+MATRIX+Repo%22)

[^1]: Today, we release the nodes and edges as two tables. Going forward we will add additional "feature tables" to the releases that contain specific features such as node embeddings and various other "non biolink" properties. These will be easily join'able with the spine tables based on the primary keys.
[^2]: We are [actively exploring](https://github.com/everycure-org/matrix/issues/543) the merged KG to understand why some nodes are not synonymized and what the theortical upper bound is for rule based synonymization.
[^3]: For more details, check our [roadmap view](https://github.com/orgs/everycure-org/projects/2/views/19)