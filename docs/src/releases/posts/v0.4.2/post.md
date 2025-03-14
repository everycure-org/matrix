---
draft: false
date: 2025-03-14
categories:
  - Release
authors:
  - alexeistepa
  - eKathleenCarter
  - lvijnck
  - emil-k
  - pascalwhoop
  - Siyan-Luo
  - JacquesVergine
  - piotrkan
  - app/github-actions
  - amyford
  - oliverw1
  - chunyuma
  - matentzn
---
# Matrix Platform `v0.3.1`: Automated Releases, Enhanced Pipelines, and Improved Infrastructure

This release introduces significant advancements in the Matrix Platform, focusing on automation, pipeline enhancements, and infrastructure improvements.  Key changes include a fully automated release process, a redesigned integration pipeline, and the introduction of a sampling pipeline for faster iterations.

<!-- more -->

## Automated Data Releases and Pipeline Enhancements

A cornerstone of this release is the introduction of a fully automated release workflow orchestrated through GitHub Actions and Argo Workflows. This workflow encompasses:

- **Automated Version Bumping and Tagging:**  Releases are automatically versioned and tagged, streamlining the release process and improving version control.
- **Automated Kedro Pipeline Triggering:** The release workflow automatically triggers the execution of Kedro pipelines on the cluster, eliminating manual intervention.
- **Automated Release Artifact Generation:** Release artifacts, including the KG dashboard and a draft release article (AI-generated), are automatically produced.
- **KG Dashboard Deployment:** The KG dashboard is automatically deployed with each release, providing immediate access to the latest data and visualizations.

These automation enhancements are documented in new runbooks located in `docs/src/infrastructure/runbooks`. Accompanying this is a new issue template for reporting corrupted releases, enhancing the feedback and debugging process.

Alongside automation, significant pipeline improvements are introduced:

- **Unified Integration Layer:** The integration pipeline has undergone a major redesign for improved modularity and maintainability.  A unified schema ensures consistency across data sources, and individual transformers now return dictionaries of nodes and edges, simplifying the integration of new sources.
- **Filtering Pipeline:**  A new filtering pipeline improves modularity by separating filtering logic from other pipeline stages.  This dedicated pipeline enhances control and customization of data filtering operations.
- **KGX Data Release Format:**  Data releases are now produced in the KGX format, improving interoperability with downstream systems and promoting standardization.
- **K-Fold Cross-Validation and Stability Metrics Enhancements:**  K-fold cross-validation is further refined, and stability metrics now provide more granular insights into model performance, including minimum and maximum aggregations. Recall@N calculation logic has also been improved.
- **MLflow Tracking Enhancements:** MLflow tracking has been enhanced to provide a more cohesive view of experiments. Each node now uses a single MLflow run, and the CLI has been reverted to `kedro submit` for consistency.  Dataset logging to MLflow has also been added for improved traceability.

A new sampling pipeline, crucial for development and testing, allows running pipelines on a subset of data, significantly reducing processing time.  This is documented in `docs/src/onboarding/sample_environment.md`.  Furthermore, scheduled and manual triggering options are available for the sampling pipeline, providing flexibility in data sampling operations.

Finally, a new pipeline ingests the released KG into Neo4j, optimizing modeling workflows.

## Infrastructure and Tooling Enhancements

Several improvements enhance infrastructure and tooling:

- **Improved Cluster Monitoring:** The deployment of Grafana and Prometheus improves cluster and experiment monitoring, providing insights into resource usage and performance.
- **Flexible Resource Management:** Memory settings for Spark and Neo4j are now parameterizable, enabling better resource utilization and avoiding memory-related issues.
- **Enhanced CLI:** The `matrix-cli` tool now facilitates bulk addition of users to GitHub teams, streamlining user management. A headless flag has been added for non-interactive CLI usage.
- **Improved Error Handling:** Error messages have been enhanced for better debugging and troubleshooting.

## Documentation Improvements

Extensive documentation updates provide greater clarity and guidance:

- **New Runbooks:** New runbooks provide step-by-step instructions for various tasks, including creating releases, running Argo workflows locally, and creating OAuth clients.
- **Enhanced Onboarding and Local Setup Documentation:**  Improved documentation clarifies Java installation, Docker usage, and other local setup procedures.
- **Updated Contribution Standards:**  Contribution guidelines now emphasize the use of `ruff` for code formatting.
- **Expanded Debugging and Error Handling Documentation:** Documentation for debugging memory issues and common errors has been expanded.

## Other Notable Changes

- **Release Information Enhancements:** Drug and disease versions are now included in release information for better traceability.
- **Improved Release Article Generation:** The release article generation process has been refined, incorporating changes to content, location, and logic.
- **Enhanced Workflow Triggers and Inputs:** GitHub Actions workflows now support specifying experiment names, bump types, and other inputs, providing more control over workflow execution.
- **Improved Slack Notifications:** Slack notifications for pipeline failures on GitHub actions improve visibility and responsiveness to issues.
- **Security Enhancements:** Hardcoded SILC configuration and `GOOGLE_CREDENTIALS` environment variable have been removed, improving security.


This release marks a significant step forward in the maturity and usability of the Matrix Platform.  The automated release process, improved pipelines, and enhanced infrastructure provide a more robust and efficient platform for drug repurposing research.  The extensive documentation updates further enhance the developer experience.  Users are encouraged to review the updated documentation, particularly the new runbooks and pipeline changes, to fully leverage the new features and functionalities.
