## Release Notes for v0.2.2

This release focuses on enhancing data integration, synonymization, and evaluation methodologies within the Matrix Platform, along with significant infrastructure and tooling upgrades.  It also introduces a new ad-hoc prediction generation capability.


## Exciting New Features 🎉

- **ROBOKOP Data Integration and RTX 2.10 Upgrade:**  Integrates ROBOKOP knowledge graph and upgrades to RTX 2.10, significantly expanding the knowledge base and improving prediction accuracy. The Every Cure KG is now released as a BigQuery dataset.
- **Translator Project Synonymization:**  Replaces the ARAX-based synonymization with a more robust, scalable solution hosted by the translator project, ensuring data consistency.  
- **Ad-hoc Prediction Generation:** Enables on-demand drug repurposing predictions through a Google Sheets interface, supporting drug-centric, disease-centric, and pair-specific predictions.


## Experiments 🧪
- **Node2Vec Integration:** Adds Node2Vec for topological embedding generation, complementing the existing GraphSAGE implementation.
- **Time-Split Validation:** Implemented time-split validation for more realistic performance assessments.
- **Recall@N Metric:** Introduced the Recall@N metric for granular performance analysis.
- **Matrix Transformation Techniques:** Experimenting with normalization techniques to counteract biases from frequent flyer diseases and drugs. 


## Technical Enhancements 🧰
- **Enhanced Docker Build Process:** Improved Docker image building to better support multi-platform development.
- **Matrix Generation Pipeline:** A new matrix generation pipeline provides a streamlined process for generating drug-disease prediction matrices.
- **Data Release Pipeline:**  A new data release pipeline provides a structured process for managing and distributing knowledge graph snapshots, improving reproducibility.
- **`kedro submit` enhancement:** Enables submitting the end-to-end workflow directly from the current branch.
- **GPU Node Pools:** Enabled GPU Node Pools for leveraging GPUs in certain tasks.
- **Secure Neo4J GDS License Storage:** Implemented secure storage of Neo4J GDS licenses in Kubernetes using external secrets management.
- **Asynchronous Embedding Computation:** Embeddings are now computed asynchronously for improved performance and scalability.
- **Node Level Resource Configuration:** Implemented configuring Kubernetes resource limits at the individual node level, optimizing resource allocation.
- **Simplified Environment Variable Management:** Improved environment variable management with default values and documentation in `.env.defaults` file.


## Bugfixes 🐛
- **IndexError Fix:** Resolved an IndexError related to outdated data fabricator versions.
- **Conda Incompatibility:** Addressed issues caused by incompatibility between conda and the preferred `uv` package manager.
- **Incorrect Category Selection:** Fixed the selection of the most specific category for unified nodes in KG integration.
- **Overwriting Releases:** Fixed the issue where current runs with the same release name were overwriting each other.


## Documentation ✏️
- **Expanded Documentation:** Added comprehensive documentation for data science methodology, onboarding, experiments, and results.
- **Improved Onboarding:** Added explicit issue creation prompt to streamline the onboarding process.
- **Improved Formatting:** Improved documentation formatting and fixed broken Markdown links.


## Newly onboarded colleagues 🚤
- Kathleen Carter
- Emil Krause
- Siyan Luo


## Other Changes
- Added PGP keys for several new team members.
- Added tests for unused catalog entries and removed unused datasets.
- Rotated the read-only service account key for enhanced security.
- Managed developer namespaces and integrated Redis into the BTE-TRAPI deployment.
- Improved Makefile commands for enhanced functionality and cross-platform compatibility.
- Added CLI smoke tests for various commands and improved error handling.
- Added a report on stability and performance of E2E runs.
- Added pipeline and release labels to workflow templates for improved workflow management.
- Updated codeowners for the infra folder.
- Added two notebooks to experiments for reproducibility.
- Updated GH Action to allow forwarding POST data.
- Added a dummy Github Action for release distribution.
- Removed unused dev scripts.
- Added a common error description for a frequently occuring error in the Fabricator.
- Added more documentation and improved existing docs.

