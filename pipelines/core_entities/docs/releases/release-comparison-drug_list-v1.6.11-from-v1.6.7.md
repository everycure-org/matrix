# Release Comparison Report

**New Release:** `v1.6.11-drug_list`

**Base Release:** `v1.6.7-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.11/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.7/ec-drug-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 408 |
| `atc_level_2` | 411 |
| `atc_level_3` | 413 |
| `atc_level_4` | 416 |
| `atc_level_5` | 416 |
| `atc_main` | 416 |
| `is_fda_generic_drug` | 1 |
| `l1_label` | 408 |
| `l2_label` | 411 |
| `l3_label` | 413 |
| `l4_label` | 413 |
| `l5_label` | 383 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `L` | `*None*` |
| `EC:00259` | Calcium gluconate | `A` | `*None*` |
| `EC:01379` | Raltitrexed | `L` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `L` |
| `EC:00724` | Framycetin | `S` | `D` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `L01` | `*None*` |
| `EC:00259` | Calcium gluconate | `A12` | `*None*` |
| `EC:01379` | Raltitrexed | `L01` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `L02` |
| `EC:00724` | Framycetin | `S01` | `D09` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `L01E` | `*None*` |
| `EC:00259` | Calcium gluconate | `A12A` | `*None*` |
| `EC:01379` | Raltitrexed | `L01B` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `L02B` |
| `EC:00724` | Framycetin | `S01A` | `D09A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `L01ED` | `*None*` |
| `EC:00259` | Calcium gluconate | `A12AA` | `*None*` |
| `EC:01379` | Raltitrexed | `L01BA` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `L02BX` |
| `EC:00724` | Framycetin | `S01AA` | `D09AA` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `L01ED03` | `*None*` |
| `EC:00259` | Calcium gluconate | `A12AA03` | `*None*` |
| `EC:01379` | Raltitrexed | `L01BA03` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `L02BX03` |
| `EC:00724` | Framycetin | `S01AA07` | `D09AA01` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `L01ED03` | `*None*` |
| `EC:00259` | Calcium gluconate | `A12AA03` | `*None*` |
| `EC:01379` | Raltitrexed | `L01BA03` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `L02BX03` |
| `EC:00724` | Framycetin | `S01AA07` | `D09AA01` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00259` | Calcium gluconate | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01379` | Raltitrexed | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:00724` | Framycetin | `Sensory organ drugs` | `Dermatologicals` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `Antineoplastic agents` | `*None*` |
| `EC:00259` | Calcium gluconate | `Mineral supplements` | `*None*` |
| `EC:01379` | Raltitrexed | `Antineoplastic agents` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `Endocrine therapy antineoplastic and immunomodulating agents` |
| `EC:00724` | Framycetin | `Ophthalmologicals` | `Medicated dressings` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `Protein kinase inhibitors, antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00259` | Calcium gluconate | `Calcium supplements` | `*None*` |
| `EC:01379` | Raltitrexed | `Antimetabolites` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `Hormone antagonists and related agents` |
| `EC:00724` | Framycetin | `Antiinfective ophthalmologics` | `Medicated dressings` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `Anaplastic lymphoma kinase (alk) inhibitors` | `*None*` |
| `EC:00259` | Calcium gluconate | `Calcium` | `*None*` |
| `EC:01379` | Raltitrexed | `Folic acid analogs, antimetabolites` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `Other hormone antagonists and related agents in atc` |
| `EC:00724` | Framycetin | `Antibiotics, ophthalmologic` | `Medicated dressings with antiinfectives` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00040` | Alectinib | `Alectinib` | `*None*` |
| `EC:00259` | Calcium gluconate | `Calcium gluconate` | `*None*` |
| `EC:01379` | Raltitrexed | `Raltitrexed` | `*None*` |
| `EC:00005` | Abiraterone | `*None*` | `Abiraterone` |
| `EC:00975` | Macitentan | `Macitentan` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 508 | 511 |
| `atc_level_2` | 508 | 511 |
| `atc_level_3` | 508 | 511 |
| `atc_level_4` | 508 | 511 |
| `atc_level_5` | 508 | 511 |
| `atc_main` | 508 | 511 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1814 | 1814 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 16 | 16 |
| `id` | 0 | 0 |
| `is_analgesic` | 0 | 0 |
| `is_antimicrobial` | 0 | 0 |
| `is_antipsychotic` | 0 | 0 |
| `is_cardiovascular` | 0 | 0 |
| `is_cell_therapy` | 0 | 0 |
| `is_chemotherapy` | 0 | 0 |
| `is_fda_generic_drug` | 0 | 0 |
| `is_glucose_regulator` | 0 | 0 |
| `is_sedative` | 0 | 0 |
| `is_steroid` | 0 | 0 |
| `l1_label` | 508 | 511 |
| `l2_label` | 508 | 511 |
| `l3_label` | 508 | 511 |
| `l4_label` | 515 | 521 |
| `l5_label` | 555 | 562 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
