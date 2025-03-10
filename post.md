---
draft: false
date: 2025-03-10
categories:
  - Release
authors:
  - Siyan-Luo
  - alexeistepa
  - piotrkan
  - JacquesVergine
  - emil-k
  - pascalwhoop
  - eKathleenCarter
  - amyford
  - lvijnck
  - app/github-actions
  - oliverw1
  - matentzn
---
# Matrix Platform `v0.4.0`: Enhanced Data Integration, Model Training, and Infrastructure Automation

This release focuses on enhancements to data integration, model training workflows, infrastructure automation, and substantial improvements to documentation and developer experience.  It introduces a unified integration layer, Spoke KG integration, DrugMech data integration, and improvements to clinical trial data handling.  Model training benefits from k-fold cross-validation and customizable ensemble model aggregation.  Infrastructure improvements include Grafana and Prometheus deployment, scheduled sampling pipeline runs, and Kubernetes cluster RBAC with Google Groups.  Significant effort has been dedicated to refining documentation, particularly for onboarding and release processes.

<!-- more -->

## Enhanced Data Integration and Preprocessing

A new unified integration layer streamlines the addition of new data sources by standardizing transformation and normalization steps. This simplifies the integration process and improves maintainability.  Spoke KG integration enriches the knowledge base, while DrugMech data integration expands the available data sources.  Clinical trial data integration now includes CURIE resolution and optimized data structure, enhancing data quality and analysis potential.  Deduplication has been moved from preprocessing to integration, optimizing data flow.  (#772, #1041, #1118)

The preprocessing pipeline has been refactored to handle experimental nodes and edges, integrating with Google Sheets for faster hypothesis testing and support for source-specific tags.  It now leverages a name-resolver service and saves data to GCP under a version specified in `globals.yaml`. (#1039, #1082)  Drug and disease list ingestion now utilizes GitHub releases, simplifying data management. (#1050)

## Model Training and Evaluation Improvements

K-fold cross-validation has been implemented, providing a more robust model evaluation methodology. The pipeline output structure has been adjusted to accommodate per-fold results. (#683)  A new parameter allows customization of the ensemble model aggregation function, providing greater flexibility in model development. (#905)  Model stability metrics offer insights into performance variations across data subsets and seeds. (#1126)

MLflow now tracks datasets used in the pipeline, enhancing traceability and reproducibility. (#1180)  Archiving of MLflow runs improves management of experiment history. (#1181)

## Infrastructure Automation and Enhancements

Grafana and Prometheus deployment provides comprehensive monitoring capabilities for cluster performance and experiment runs. (#834)  The sampling pipeline is now scheduled for daily execution via Argo Workflows, ensuring regular updates of sample data. (#1105)

Kubernetes cluster RBAC using Google Groups enhances security and simplifies user/group management for Argo workflow submissions and cluster administration. (#1040)  Google Secret Manager now stores and manages secrets, improving security and cross-account access. (#1073)  A public data zone utilizing Google Cloud Storage and Cloud CDN provides secure and efficient access to public data and static websites. (#1085)  An evidence dashboard enables exploration and visualization of the Matrix KG with interactive filters and charts. (#1153)

## Developer Experience and Documentation

Significant improvements to documentation, including onboarding materials, Kedro resource documentation, and release procedures, enhance developer experience. (#902, #919, #940)  Updates to troubleshooting documentation, common errors, and installation instructions further streamline the development process. (#925, #934)  Java version requirements have been upgraded to 17. (#903)  Documentation for disease tagging, experiment setup, and release tags clarifies key functionalities. (#955, #1159, #1209)

The Matrix CLI streamlines project tasks like user management and release note generation. (#1145)  The `kedro experiment` command enhances MLflow integration and replaces `kedro submit`, improving experiment tracking and management. (#1142)  A mechanism to selectively disable Kedro hooks aids local development.  (#900)

## Bug Fixes and Technical Enhancements

Numerous bug fixes address issues related to Neo4j interaction, schema errors, release branching, data normalization, MLflow metric tracking, interpolation errors, and resource allocation. (#781, #823, #899, #900, #943, #950, #1039, #1060, #1075, #1123, #1165, #1170, #1176)

Technical enhancements include the removal of the `refit` library, simplified Neo4j SSL setup, consistent use of `pyspark.sql`, refactoring of `argo_node` to `ArgoNode`, and improved error handling and logging. (#811, #878, #885, #923)  Modeling pipeline cleanup unifies split generation. (#907)  Pandera replaces a private package for runtime data quality checks. (#938)

## Other Notable Changes

Release notes and articles are now automatically generated for major and minor releases.  A new issue template streamlines tracking and resolving corrupted KG releases.  Release information rendering has been implemented.  MLflow experiment management has been improved.  Various Makefile issues have been resolved. (#858, #1093, #1096, #1103, #1172)


This release significantly advances the Matrix Platform, enhancing data integration, model training workflows, and infrastructure automation while improving developer experience through documentation and tooling improvements.  The numerous bug fixes and technical enhancements further solidify the platform's stability and reliability.
