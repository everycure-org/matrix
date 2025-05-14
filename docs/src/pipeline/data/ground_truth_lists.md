# Ground Truth Lists

The MATRIX pipeline integrates several ground truth datasets to validate and evaluate our drug repurposing predictions. Each dataset represents edges (relationships) between drugs and diseases, following our standard edge schema with subject (drug), object (disease), predicate, and metadata fields. Like other data sources in our pipeline, each dataset has a dedicated transformer class that handles standardization and integration:

## Clinical Trials Data
Clinical trials data (version 20230309) curated by Sally at the EveryCure medical team, containing cleaned and standardized drug-disease pairs from March to September 2023 clinical trials. The `ClinicalTrialsTransformer` class handles preprocessing to standardize entity identifiers and remove duplicates.

## Ground Truth Dataset 
Ground truth dataset (version v2.10.0_validated) developed and published as part of the KGML-xDTD project. This dataset was specifically designed and validated for use with the RTX-KG2 knowledge graph, providing a comprehensive set of validated drug-disease treatment associations. The dataset has been extensively curated to ensure high quality positive and negative examples. Integration is handled by the `GroundTruthTransformer` class.

## Off-Label Data
Off-label drug usage data (version v0.1) containing documented cases of drugs being used for non-approved indications. This prototype dataset currently combines off-label usage information extracted from external sources like PrimeKG. Integration is managed through the `OffLabelTransformer` class. In future releases, this will be replaced with more comprehensive off-label data sourced directly from DrugBank.

> These datasets are integrated into our pipeline through the `integration` module, with configurations specified in `settings.py`. The data versions are centrally managed through our `globals.yml` configuration. Each transformer follows the standard `Transformer` interface, ensuring consistent data processing and schema compliance across all data sources.
