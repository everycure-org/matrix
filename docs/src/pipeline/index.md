# MATRIX

The MATRIX pipeline is our main codebase in the Every Cure organization. Its goal is the generation of high accuracy predictions of drug-disease pairs in an "all vs all" approach. For this, we ingest and integrate a number of data sources, build models and make predictions on the ingested data. 

You can find more details about the pipeline stages here:

- [Ingestion](./pipeline_steps/ingestion.md)
- [Integration](./pipeline_steps/integration.md)
- [Filtering](./pipeline_steps/filtering.md)
- [Embedding](./pipeline_steps/embeddings.md)
- [Modelling](./pipeline_steps/modelling.md)
- [Matrix generation](./pipeline_steps/matrix_generation.md)
- [Matrix transformation](./pipeline_steps/matrix_transformation.md)
- [Evaluation](./pipeline_steps/evaluation.md)

Of course there is a lot more to this but the below illustration aims to sketch the high level flow of the pipeline:

![Matrix Pipeline](../../assets/getting_started/matrix_overview.png)
