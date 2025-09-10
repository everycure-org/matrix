---
title: Customizing Matrix Generation
---
<!-- NOTE: This file was partially generated using AI assistance. -->

# Background

The matrix generation pipeline is producing scores for drug-disease pairs coming directly from the knowledge graph. The 'raw' output is not very useful for the medical team as it contains over 60 million pairs and is hard to navigate due to non-human readable IDs and lack of metadata. Therefore, we need to make it more human-readable. 

The reporting node is taking top n_reporting pairs and enriching them with metadata, tags, and statistics. The final part of the reporting node is also to generate metadata, legend, and statistics sheets. All these files are then being saved in the `matrix_report.xls` file. The file has the following structure:
* `matrix_report` sheet - where the main matrix output is stored:
  * `pair_id` - unique identifier for pairs
  * `drug_name` / `disease_name` - drug and disease names coming from the drug and disease lists
  * `kg_drug_id` / `kg_disease_id` - drug and disease ids that can be mapped to the KG
  * `score` - prediction score for specific pair
  * `tags` (e.g. `is_antimicrobial`, `is_steroid`, `is_cancer`) - various tags corresponding to specific drug or disease (used for filtering out pairs)
  * `master_filter` - conditional statement excluding specific pairsbased on several tags (e.g. `is_antimicrobial` AND `is_pathogen_caused` -> TRUE)
  * `stats` (e.g. `mean_score`, `median_score`) - various disease- or drug-specific statistics calculated for both n_reporting pairs as well as the whole matrix
* `metadata` sheet - contains run/experiment metadata such as run name, versions of data used, git sha, etc. 
* `legend` sheet - contains legend for all columns present in the matrix_report.csv
* `statistics` sheet - contains statistics for full matrix as well as top n_reporting pairs


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


## 1. Required fields for matrix_report

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
## 2. Adding or Removing Tags/Filters

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

## 3. Adding or Modifying Tags

Tags are used to provide additional information about drugs, diseases, and drug-disease pairs. They are defined in the `tags` section of the `parameters.yml` file. Here's an overview of the current tag structure:

```yaml
matrix_generation.matrix:
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
```

To add or modify tags, you will first need to identify whether the specific tag is present in drug or disease list respectively. Note that pairs tags are Work in Progress and will be added in the future. When adding the tag, note that the key will be used as the column name in the output, while the value serves as a description for the legend (which will be used in the legends sheet).

Remember that these tags should correspond to data available in your drug and disease lists. The enrichment process will use these tags to add the relevant information to the matrix report.


## 4. Master filter tag

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

## 5. Customizing Descriptive Statistics

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

## 6. Remaining sheets

While the most important sheet is the `matrix_report.csv`, the metadata file includes all information regarding the versions of the data used, naming conventions as well as short summary statistics. 

### 6.1 Metadata sheet

This sheet contains all the metadata regarding the versions of the data used, run-names and links to MLFlow where metrics and specific parameters can be found. These can be specified under `matrix_generation.run` section:

```yaml
matrix_generation.run:
  workflow_id: ${oc.env:WORKFLOW_ID, local}}
  mlflow_link: ${globals:mlflow_url}
  git_sha: ${globals:git_sha}
  release_version: ${globals:versions.release}
  versions: ${globals:data_sources}
```
### 6.2 Summary statistics

This sheet contains the summary statistics for the whole matrix as well as the top_n pairs. The statistics can include `mean_score`, `median_score`, `quantile score`, `max_score`, `min_score`, and `std_score`. These can be specified under `stats_col_names` section:

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

### 6.3 Legend sheet

This sheet contains the legend for the matrix report. The legend includes the description of the tags used in the matrix report. These are specified as values in the key:value pairs

## Examples of full statistics/tags 

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