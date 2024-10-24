---
title: Customizing Matrix Generation
---
<!-- NOTE: This file was partially generated using AI assistance. -->

# Background

The matrix generation pipeline is producing scores for drug-disease pairs coming directly from the knowledge graph. The 'raw' output is not very useful for the medical team as it contains over 60 million pairs and is hard to navigate due to non-human readable IDs and lack of metadata. Therefore, we need to make it more human-readable. 

The reporting node is taking top n_reporting pairs and enriching them with metadata, tags, and statistics. The enriched matrix is then being saved in the `matrix_report.csv` and contains columns such as:
* `pair_id` - unique identifier for pairs
* `drug_name` / `disease_name` - drug and disease names coming from the drug and disease lists
* `kg_drug_id` / `kg_disease_id` - drug and disease ids that can be mapped to the KG
* `score` - prediction score for specific pair
* `tags` (e.g. `is_antimicrobial`, `is_steroid`, `is_cancer`) - various tags corresponding to specific drug or disease (used for filtering out pairs)
* `master_filter` - conditional statement excluding specific pairsbased on several tags (e.g. `is_antimicrobial` AND `is_pathogen_caused` -> TRUE)
* `stats` (e.g. `mean_score`, `median_score`) - various disease- or drug-specific statistics calculated for both n_reporting pairs as well as the whole matrix

The `matrix_metadata.xls` file contains a few sheets:
* `matrix_report` - contains the same data as `matrix_report.csv` file
* `metadata` - contains run/experiment metadata such as run name, versions of data used, git sha, etc. 
* `legend` - contains legend for all columns present in the matrix_report.csv
* `statistics` - contains statistics for full matrix as well as top n_reporting pairs

# Customizing the enrichment of the matrix

This guide explains how to customize the enrichment of the matrix with tags, filters, and statistics by modifying the `parameters.yml` file. The file is located at `pipelines/matrix/conf/base/matrix_generation/parameters.yml`, under `matrix_generation.matrix` section:

```yaml
matrix_generation.matrix:
  metadata:
    drug_list:
      drug_id: "Drug identifier"
      drug_name: "Drug name"
    # ... ( more metadata fields)
  stats_col_names:
    # ... (statistics columns)
  tags:
    # ... (tag columns)
```

Also do note that the actual tags/column names are written in a `key:value` format; while only key is used in the matrix output, the value is being used as a 'legend' that is appended to the metadata .csv file.

## 0. Required fields

The parameters.yaml file contains the entries essential for good understanding of the matrix output: these entries are defined as `drug_list`, `disease_list`, and `kg_data`, and are necessary for the matrix generation process so __do not remove them__.

```yaml
matrix_generation.matrix:
  metadata:
    # Drug list extracts drug IDs and drug names from the drug list and appends them to the ids used by the model
    drug_list:
      drug_id: "Drug identifier"
      drug_name: "Drug name"
    # Disease list extracts disease IDs and disease names from the disease list and appends them to the ids used by the model
    disease_list:
      disease_id: "Disease identifier"
      disease_name: "Disease name"
    # KG_data highlights the drug/disease ids which are being used by the model and which are mappable in the KG. The only exception is the `pair_id` which is a unique identifier for each drug-disease pair which we generate in matrix_generation pipeline.
    kg_data:
      pair_id: "Unique identifier for each pair"
      kg_drug_id: "Drug identifier in the knowledge graph"
      kg_drug_name: "Drug name in the knowledge graph"
      kg_disease_id: "Disease identifier in the knowledge graph"
      kg_disease_name: "Disease name in the knowledge graph"
    # ... (tag columns)
```
## 1. Adding or Removing Tags/Filters

To add or remove tags/filters, modify the `metadata` section:

```yaml
matrix_generation.matrix:
  metadata:
    drug_list:
      drug_id: "Drug identifier"
      drug_name: "Drug name"
    disease_list:
      disease_id: "Disease identifier"
      disease_name: "Disease name"
    # Add or remove fields as needed
```

These fields will be used in the `generate_metadata` function to create the metadata for each drug-disease pair. Do note that these tags are being extracted from the drug and disease lists and will be used to enrich the matrix - therefore if a specific tag is not available for a drug or disease, it will not be shown in the final matrix.

## 1.1 Master filter tag

Note that there is also a `master_filter` tag which is essentially a conditional statement based on several other tags. This is the most important filter for the medical team as it utilizes both drug and disease tags to exclude certain pairs - for example a pair where a drug is an antimicrobial and the disease is a pathogen-caused disease. Such pair is not incredibly useful when it comes to drug repurposing as we are trying to find new uses for existing drugs.

The conditions for the `master_filter` tag are defined in the `master` section:
```yaml
matrix_generation.matrix:
  tags:
    master: 
      legend: "excludes any pairs where the following conditions are met: known positive or negative status, antimicrobial drug with pathogen-caused disease, steroid drug, chemotherapy with cancer, or glucose regulator drug with glucose dysfunction disease"
      conditions: # if any of the conditions are met, the tag returns True
        - [is_known_positive] # if is_known_positive is True, the tag returns True
        - [is_known_negative] # if is_known_negative is True, the tag returns True
        - [is_antimicrobial, is_pathogen_caused] # if is_antimicrobial is True AND is_pathogen_caused is True, the tag returns True
        - [is_steroid] # if is_steroid is True, the tag returns True
        - [is_glucose_regulator, is_glucose_dysfunction] # if is_glucose_regulator is True AND is_glucose_dysfunction is True, the tag returns True
        - [is_chemotherapy, is_cancer] # if is_chemotherapy is True AND is_cancer is True, the tag returns True
```

## 2. Customizing Descriptive Statistics

To add, remove, or modify descriptive statistics, update the `stats_col_names` section:

```yaml
matrix_generation.matrix:
  stats_col_names:
    per_disease:
      top:
        mean: "Mean probability score"
        median: "Median probability score"
        # Add or remove statistics as needed
      all:
        # ... (similar structure for all drugs)
    per_drug:
      # ... (similar structure for drugs)
```

These statistics will be calculated in the `generate_report` function using pandas' `transform` method. We have statistics for each disease and each drug, and they are calculated per disease and per drug respectively - this way the end-users will have a better understanding of the data's distribution. 

Furthermore, the `stats_col_names` is divided in two sections: `all` and `top`. The `all` section calculates descriptive statistics (per drug/disease) across the whole matrix (~60 million pairs) while the `top` section calculates statistics only for `n_reporting` pairs (as specified in the `matrix_generation_options` section). You can add any statistics as long as they are compatible with pandas' `transform` method such as `mean`, `median`, `quantile`, `max`, `min`, `std`.

### 2.1 Full Matrix Descriptive Statistics

We are also calculating some statistics across the whole matrix as well as top_n pairs, these are `mean_score`, `median_score` and `quantile score`. Note that these are reported in the final metadata file, as there is only one value per stastic. These can be found within the `full` and `top_n` sections:

```yaml
stats_col_names:
  #...
  full:
    mean: "Mean probability score"
    median: "Median probability score"
    quantile: "50% quantile of probability score"
    max: "Maximum probability score"
    min: "Minimum probability score"
    std: "Standard deviation of probability scores"
```

## Examples of statistics/tags to be included

````yaml
matrix_generation.matrix:
  metadata:
    drug_list:
      drug_id: "Drug identifier"
      drug_name: "Drug name"
    disease_list:
      disease_id: "Disease identifier"
      disease_name: "Disease name"
    kg_data:
      pair_id: "Unique identifier for each pair"
      kg_drug_id: "Drug identifier in the knowledge graph"
      kg_drug_name: "Drug name in the knowledge graph"
      kg_disease_id: "Disease identifier in the knowledge graph"
      kg_disease_name: "Disease name in the knowledge graph"
  # Note that you can add any stats as long as they are compatible with pandas transform function
  stats_col_names:
    per_disease:
      top:
        mean: "Mean probability score"
        quantile: "50% quantile of probability score"
        median: "Median probability score"
        max: "Maximum probability score"
        min: "Minimum probability score"
        std: "Standard deviation of probability scores"
      all:
        mean: "Mean probability score"
        median: "Median probability score"
        quantile: "50% quantile of probability score"
        max: "Maximum probability score"
        min: "Minimum probability score"
        std: "Standard deviation of probability scores"
    per_drug:
      top:
        mean: "Mean probability score"
        median: "Median probability score"
        quantile: "50% quantile of probability score"
        max: "Maximum probability score"
        min: "Minimum probability score"
        std: "Standard deviation of probability scores"
      all:
        mean: "Mean probability score"
        median: "Median probability score"
        quantile: "50% quantile of probability score"
        max: "Maximum probability score"
        min: "Minimum probability score"
        std: "Standard deviation of probability scores"
    full:
      mean: "Mean probability score"
      median: "Median probability score"
      quantile: "50% quantile of probability score"
      max: "Maximum probability score"
      min: "Minimum probability score"
      std: "Standard deviation of probability scores"
  tags:
    drugs:
      available_USA: "Whether drug is prescriped or discontinued in USA"
      is_steroid: "Whether drug is a steroid"
      is_antimicrobial: "Whether drug is an antimicrobial"
      is_glucose_regulator: "Whether drug is a glucose regulator"
      is_chemotherapy: "Whether drug is a chemotherapy"
    pairs:
      is_known_positive: "Whether the pair is a known positive, based on literature and clinical trials"
      is_known_negative: "Whether the pair is a known negative, based on literature and clinical trials"
    diseases:
      is_cancer: "Whether disease is a cancer"
      is_pathogen_caused: "Whether disease is a pathogen caused"
      is_glucose_dysfunction: "Whether disease is a glucose dysfunction"
      tag_existing_treatment: "Whether disease has existing treatments"
      tag_QALY_lost: "Whether disease has Quality Adjusted Life Years (QALY) lost"
    master: 
      legend: "Excludes any pairs where the following conditions are met: known positive or negative status, antimicrobial drug with pathogen-caused disease, steroid drug, chemotherapy with cancer, or glucose regulator drug with glucose dysfunction disease"
      conditions:
        - [is_known_positive]
        - [is_known_negative]
        - [is_antimicrobial, is_pathogen_caused]
        - [is_steroid]
        - [is_glucose_regulator, is_glucose_dysfunction]
        - [is_chemotherapy, is_cancer]

matrix_generation.run:
  workflow_id: ${oc.env:WORKFLOW_ID, local}}
  mlflow_link: ${globals:mlflow_url}
  git_sha: ${globals:git_sha}
  release_version: ${globals:versions.release}
  versions: ${globals:data_sources}
````