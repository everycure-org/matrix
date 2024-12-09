## Release Notes: v0.2.2 - Enhanced Data Integration, Robust Synonymization, and Advanced Evaluation

This release significantly enhances the Matrix Platform's capabilities, focusing on data integration, synonymization, and evaluation methodologies.  It also introduces a new ad-hoc prediction generation pipeline.


## Exciting New Features 🎉

- **ROBOKOP Data Integration and RTX 2.10 Upgrade:** Integrated ROBOKOP knowledge graph and upgraded to RTX 2.10, significantly expanding the knowledge base and improving prediction accuracy.  The Every Cure KG is now released as a BigQuery dataset.
- **Translator Project-Based Synonymization:** Implemented a scalable synonymization system using the translator project, improving data consistency and streamlining the data preparation process. Synonymization now supports drugs, diseases, and medical inputs.
- **Ad-hoc Prediction Generation Pipeline:**  New pipeline enables on-demand drug repurposing predictions via a user-friendly Google Sheets interface, supporting drug-centric, disease-centric, and drug-disease pair specific predictions.  This supports parallel model predictions and real-time result visualization.
- **Data Release Pipeline:** New pipeline for managing and distributing versioned knowledge graph snapshots, facilitating experimentation and reproducibility. Supports experiments linked to a single KG release.
- **`kedro submit` Enhancement:** New command line feature simplifies end-to-end pipeline execution from a branch.


## Technical Enhancements 🧰

- **`make` and Onboarding Improvements:** The `Makefile` has been enhanced, streamlining the onboarding process.  A `make clean` target has been added.
- **Documentation Expansion:**  Added comprehensive documentation on data science methodology, onboarding, experiments, and results.
- **Running Single Nodes Locally with Cloud Data:** Improved local development by enabling the `--from-env cloud` flag in the `kedro run` command. This enables testing individual nodes with full cloud data locally.
- **Neo4J Enterprise License Keys:** Added support for enterprise features, allowing use of more CPU cores for graph algorithms.
- **Improved Argo Workflow Integration:**  Improved the Argo workflow submission process, improving stability and handling of edge cases. 
- **Dataset Transcoding:**  Added dataset transcoding for seamless transition between Spark and Pandas.
- **Efficient Data Loading:** Using SparkDataset with BigQuery external tables optimizes data loading, saving on costs and streamlining the loading process.
- **Asynchronous Embeddings with PartitionedDataset and LangChain:** Implemented asynchronous embedding computation for improved performance and scalability.
- **Environment Variable Management:** Streamlined environment variable management with the introduction of `.env.defaults` (version-controlled defaults) and `.env` (local overrides).
- **Node Level Resource Configuration:** Added support for configuring resources (CPU, memory, GPUs) at the node level in Argo Workflows, improving resource utilization.


## Experiments 🧪

- **Node2Vec Integration:** Added Node2Vec to complement GraphSAGE for topological embedding generation.
- **Time-split Validation:** Implemented time-split validation for a more robust evaluation methodology in the evaluation pipeline.


## Bugfixes 🐛

- Fixed an issue causing all runs with the same release name to overwrite each other due to a missing `run_name` in the path prefix.
- Fixed incorrect category selection for unified nodes in knowledge graph integration.
- Resolved an issue with Neo4j template and improved embeddings pipeline resource allocation.
- Fixed case sensitivity issues in Biolink hierarchy filtering.
- Addressed a bug with CPU core count allocation during `kedro submit` which was incorrectly relying on the number of CPUs on the local machine.
- Fixed a bug preventing the matrix from being generated correctly due to faulty semmed filtering for RTX KG2 2.10 data.


## Documentation ✏️

- Added a Data API specification to the documentation.
- Improved documentation formatting and fixed broken Markdown links.
- Added documentation for `IndexError` fix and Conda incompatibility.
- Added FAQ section covering common errors and their solutions.
- Improved local setup documentation.


## Newly onboarded colleagues 🚤

- Kathleen Carter
- Emil Krause
- Siyan Luo


## Other Changes

- Added PGP keys for new team members.
- Added more smoke tests.
- Added more test coverage to existing tests.
- Rotated the read-only service account key.
- Managed developer namespaces and integrated Redis into BTE-TRAPI deployment.
- Added Makefile command to diagnose common issues.
- Improved cross-platform compatibility of the `Makefile`.
- Removed unused development scripts.
- Added pipeline and release labels to the workflow template.
- Added new codeowners for the infra folder.
- Added two notebooks to the experiments section.
- Added a dummy GitHub Action workflow for release distribution.
- Updated GH Action to allow forwarding POST data.
- Added logging of Github context in data-release action for better debugging.
- Added a report on the stability and performance of E2E runs.
- Added more information on the node filters.
- Implemented a strategy pattern for node attribute encoders. Restored missing configuration in cloud environment.
- Require release version for Kedro submit deployments.