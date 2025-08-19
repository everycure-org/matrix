# Ground Truth Lists

The MATRIX pipeline integrates several ground truth datasets to train ML models as well as validate and evaluate our drug repurposing predictions. Each dataset represents relationships between drugs and diseases, following our standard edge schema with subject (drug), object (disease), predicate, and metadata fields. Many of those relationships are directly mappable to edges in the our Knowledge Graph, some of them however are independently extracted from the KG (making them valuable for evaluation). 

All those data sources require the identifers (subjects/objects) to be follow KGX format (see Matrix Validator). Like other data sources in our pipeline, each dataset has a dedicated transformer class that handles transformations and integration; each dataset also goes through normalization for CURIEs to be in the same universe.

# Training Datasets
Those datasets are used for training our ML classifiers for predicting drug repurposing candidates. They can be used on their own as well as standalone 

## KGML-xDTD Ground Truth Dataset 
This ground truth dataset was developed and published as part of the KGML-xDTD [publication](https://academic.oup.com/gigascience/article/doi/10.1093/gigascience/giad057/7246583), and is used for model training and validation. It was specifically designed and validated for use with the RTX-KG2 knowledge graph, providing a comprehensive set of validated drug-disease treatment associations, however it has also been used with other KGs (e.g. ROBOKOP). The versions of this dataset are linked to the versions of RTX-KG2 knowledge graph. Integration is handled by the `KGMLTruthTransformer` class.

## EC Ground Truth Dataset 
This ground truth dataset was developed and published as a part of MATRIX project and can be found [within Matrix Indication List repo](https://github.com/everycure-org/matrix-indication-list). It's a curated datasets that was developed collaboratively with medical experts to ensure good quality training pairs can be provieded for the model. It should be KG-agnostic as it's directly extracted from regulatory authorities bodies. The versions of this dataset are linked to the [releases in Github](https://github.com/everycure-org/matrix-indication-list/releases).

## DrugBank Ground Truth Dataset 
This proprietary ground truth dataset was extracted from [DrugBank Database](https://go.drugbank.com) provided to EveryCure. It contains a list of indications & contraindications which are present within the database. Versions follow DrugBank convention.

# Evaluation Datasets
## Clinical Trials Data
Clinical trials data (version 20230309) was manually curated by one of EveryCure Medical Team Members and it contains drug-disease pairs from March to September 2023 clinical trials. Provided that the KG time cut-off was before 2023.03.09, we can use this dataset for both unseen test validation as well as time-split validation for ML models. The `ClinicalTrialsTransformer` class handles preprocessing to standardize entity identifiers and remove duplicates.

## Off-Label Data
Off-label drug usage data containing documented cases of drugs being used for non-approved indications. This prototype dataset (v0.1) currently combines off-label usage information  - for details [see a notebook in our experiment repository](https://github.com/everycure-org/lab-notebooks/blob/main/medical-score/prime_kg_extract_dd.ipynb). It can be used for validation of model performance on the unseen test set. Integration is managed through the `OffLabelTransformer` class. In future releases, this will be replaced with more comprehensive off-label data sourced directly from DrugBank.

> These datasets are integrated into our pipeline through the `integration` module, with configurations specified in `settings.py`. The data versions are centrally managed through our `globals.yml` configuration. Each transformer follows the standard `Transformer` interface, ensuring consistent data processing and schema compliance across all data sources.
