# Matrix Platform v0.2.2: Ad-hoc Predictions and Enhanced Data Integration

The Matrix Platform v0.2.2 release delivers significant advancements in data integration, synonymization, and evaluation, alongside crucial infrastructure upgrades and a new ad-hoc prediction capability. This release strengthens the platform's ability to generate robust drug repurposing predictions.

<!-- more -->

## Enhanced Data Foundation

This release integrates the ROBOKOP knowledge graph and upgrades to RTX 2.10, significantly expanding the knowledge base and improving prediction accuracy. The integration of ROBOKOP provides a richer context for analysis, while the RTX 2.10 upgrade ensures access to the latest knowledge.  The Every Cure Knowledge Graph is now available as a BigQuery dataset, facilitating broader access and analysis.

A key improvement is the transition to Translator-based synonymization, replacing the previous ARAX-based system. This change delivers a more robust and scalable solution, ensuring greater data consistency and improving the quality of data integration.

## Ad-hoc Prediction Generation

v0.2.2 introduces a new ad-hoc prediction generation capability.  Leveraging a Google Sheets interface, users can now generate on-demand drug repurposing predictions. This feature supports drug-centric, disease-centric, and pair-specific predictions, offering flexibility for various research questions.

## Rigorous Evaluation and Experimentation

Several enhancements improve the platform's evaluation methodologies:

* **Time-Split Validation:**  This new feature enables more realistic performance assessments by evaluating models on temporally separated data.
* **Recall@N Metric:** This metric provides granular performance analysis, focusing on the model's ability to retrieve relevant drug-disease pairs within the top N predictions.
* **Node2Vec Integration:**  This addition complements the existing GraphSAGE implementation, offering another approach to topological embedding generation and potentially uncovering different structural insights within the knowledge graph.
* **Matrix Transformation Techniques:** Ongoing experiments with normalization techniques aim to mitigate biases introduced by frequently occurring diseases and drugs, further refining the accuracy of predictions.

## Technical Enhancements and Infrastructure Upgrades

Several technical improvements enhance the platform's performance, scalability, and developer experience:

* **Enhanced Docker Build Process:**  Improvements to the Docker image building process better support multi-platform development.
* **Matrix and Data Release Pipelines:** New pipelines streamline the generation of prediction matrices and the management of knowledge graph snapshots, improving reproducibility and efficiency.
* **`kedro submit` Enhancement:**  This update allows direct submission of the end-to-end workflow from the current branch, simplifying development workflows.
* **GPU Node Pools:**  The platform now leverages GPU node pools for accelerated computation of specific tasks.
* **Secure Neo4j GDS License Storage:**  Secure storage of Neo4j GDS licenses within Kubernetes using external secrets management enhances security.
* **Asynchronous Embedding Computation:**  Embeddings are now computed asynchronously, improving performance and scalability.
* **Node-Level Resource Configuration:**  Fine-grained control over Kubernetes resource limits at the individual node level optimizes resource allocation.
* **Simplified Environment Variable Management:**  Improved environment variable management, including default values and documentation in the `.env.defaults` file, simplifies configuration.


## Bug Fixes and Documentation Improvements

Several bug fixes address issues related to data fabricator versions, conda incompatibility, and category selection.  Expanded documentation, improved onboarding guidance, and enhanced formatting contribute to a better user experience.

## Team Expansion

The Matrix team welcomes Kathleen Carter, Emil Krause, and Siyan Luo.  Their expertise will contribute to the continued development and improvement of the platform.

##  Looking Ahead

Future development will focus on fully automating the data release pipeline, integrating the SPOKE knowledge graph, and enhancing the ad-hoc prediction interface.  Further improvements to modeling, evaluation, and infrastructure are planned to continue enhancing the platform's capabilities and user experience.
