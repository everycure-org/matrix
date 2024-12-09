# Matrix Platform v0.2.2: Enhanced Data Integration, Advanced Synonymization, and On-Demand Predictions

The Matrix Platform v0.2.2 release delivers substantial improvements to data integration, synonymization, and evaluation methodologies, while introducing a powerful new ad-hoc prediction pipeline. This release solidifies the platform's capabilities for robust and flexible drug repurposing research.

## Key Enhancements

### Expanded Knowledge Graph and Improved Accuracy

This release integrates the ROBOKOP knowledge graph and upgrades to RTX 2.10, significantly expanding the platform's knowledge base and improving prediction accuracy. This broadened context provides richer insights for drug repurposing analysis.  Furthermore, the Every Cure Knowledge Graph is now readily accessible as a BigQuery dataset, simplifying data exploration and analysis.

### Scalable and Consistent Synonymization

A new, scalable synonymization system powered by the Translator project replaces the previous implementation. This enhancement ensures greater data consistency and streamlines data preparation by accurately mapping drug, disease, and medical input synonyms across all integrated knowledge graphs.

### Ad-hoc Prediction Pipeline

A new pipeline enables on-demand drug repurposing predictions through a user-friendly Google Sheets interface.  This allows researchers to generate drug-centric, disease-centric, or drug-disease pair specific predictions with real-time result visualization and support for parallel model execution.  This feature introduces unprecedented flexibility for exploring specific research questions.

### Data Release Pipeline and Reproducibility

A new data release pipeline facilitates the creation and distribution of versioned knowledge graph snapshots.  This promotes experimentation and ensures reproducibility by linking experiments to specific KG releases. This structured approach to data management improves the reliability and traceability of research findings.

### Streamlined Development Workflow

Several enhancements simplify development and improve pipeline efficiency:

- **`kedro submit` from Branch:** A new CLI feature simplifies end-to-end pipeline execution directly from a specific branch.
- **Enhanced `Makefile`:** The `Makefile` has been enhanced with a `make clean` target, streamlining the onboarding process and environment management.
- **Local Development with Cloud Data:** The `kedro run --from-env cloud` flag enables developers to test individual nodes locally using full cloud data, accelerating the development cycle.
- **Neo4j Enterprise Features:** Support for Neo4j Enterprise license keys unlocks access to advanced graph algorithms and improved performance.
- **Improved Argo Workflow Integration:** Enhancements to the Argo workflow submission process improve stability and handle edge cases more effectively.
- **Optimized Data Loading:** Utilizing SparkDataset with BigQuery external tables optimizes data loading, reducing costs and improving efficiency.
- **Asynchronous Embeddings:** Asynchronous embedding computation with PartitionedDataset and LangChain enhances performance and scalability.
- **Streamlined Environment Variables:**  The introduction of `.env.defaults` for version-controlled defaults and `.env` for local overrides simplifies environment variable management.
- **Node-Level Resource Configuration:** Support for configuring CPU, memory, and GPU resources at the node level in Argo workflows optimizes resource utilization.

### Robust Evaluation Methodology

The evaluation pipeline benefits from significant improvements, including:

- **Node2Vec Integration:** The addition of Node2Vec complements GraphSAGE for more robust topological embedding generation.
- **Time-split Validation:** Implementation of time-split validation provides a more rigorous evaluation methodology, ensuring a more realistic assessment of predictive performance.

### Bug Fixes and Documentation

Numerous bug fixes address issues related to data integration, Neo4j template configuration, Biolink hierarchy filtering, and CPU core allocation.  Expanded documentation covers data science methodology, onboarding, experiments, results, and a new Data API specification.


## Conclusion

The Matrix Platform v0.2.2 release represents a significant step forward in drug repurposing research. The combination of enhanced data integration, improved synonymization, advanced evaluation methodologies, and the new ad-hoc prediction pipeline empowers researchers with a powerful and flexible platform for discovering potential drug repurposing candidates.
