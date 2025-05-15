# Ground Truth Lists

The MATRIX pipeline integrates several ground truth datasets to validate and evaluate our drug repurposing predictions. Each dataset represents edges (relationships) between drugs and diseases, following our standard edge schema with subject (drug), object (disease), predicate, and metadata fields. These datasets can be used for both training or testing our ML models in the pipeline.

Like other data sources in our pipeline, each dataset has a dedicated transformer class that handles standardization and integration:

## Clinical Trials Data
Clinical trials data (version 20230309) was manually curated by one of EveryCure Medical Team Members and it contains drug-disease pairs from March to September 2023 clinical trials. Provided that the KG time cut-off was before 2023.03.09, we can use this dataset for both unseen test validation as well as time-split validation for ML models. The `ClinicalTrialsTransformer` class handles preprocessing to standardize entity identifiers and remove duplicates.

## KGML-xDTD Ground Truth Dataset 
This ground truth dataset was developed and published as part of the KGML-xDTD project, and is used for model training and validation. It was specifically designed and validated for use with the RTX-KG2 knowledge graph, providing a comprehensive set of validated drug-disease treatment associations, however it has also been used with other KGs (e.g. ROBOKOP). The versions of this dataset are linked to the versions of RTX-KG2 knowledge graph. Integration is handled by the `GroundTruthTransformer` class.

## Off-Label Data
Off-label drug usage data containing documented cases of drugs being used for non-approved indications. This prototype dataset (v0.1) currently combines off-label usage information extracted from external sources like PrimeKG. It can be used for validation of model performance on the unseen test set. Integration is managed through the `OffLabelTransformer` class. In future releases, this will be replaced with more comprehensive off-label data sourced directly from DrugBank.

> These datasets are integrated into our pipeline through the `integration` module, with configurations specified in `settings.py`. The data versions are centrally managed through our `globals.yml` configuration. Each transformer follows the standard `Transformer` interface, ensuring consistent data processing and schema compliance across all data sources.
