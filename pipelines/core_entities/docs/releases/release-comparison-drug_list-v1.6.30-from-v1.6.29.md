# Release Comparison Report

**New Release:** `v1.6.30-drug_list`

**Base Release:** `v1.6.29-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.30/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.29/ec-drug-list.parquet`

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
| `atc_level_1` | 452 |
| `atc_level_2` | 453 |
| `atc_level_3` | 454 |
| `atc_level_4` | 454 |
| `atc_level_5` | 454 |
| `atc_main` | 454 |
| `l1_label` | 452 |
| `l2_label` | 453 |
| `l3_label` | 454 |
| `l4_label` | 452 |
| `l5_label` | 423 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `R` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `C` |
| `EC:01412` | Rifapentine | `J` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `H` | `*None*` |
| `EC:01423` | Risdiplam | `M` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `R07` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `C01` |
| `EC:01412` | Rifapentine | `J04` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `H01` | `*None*` |
| `EC:01423` | Risdiplam | `M09` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `R07A` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `C01A` |
| `EC:01412` | Rifapentine | `J04A` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `H01A` | `*None*` |
| `EC:01423` | Risdiplam | `M09A` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `R07AX` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `C01AA` |
| `EC:01412` | Rifapentine | `J04AB` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `H01AB` | `*None*` |
| `EC:01423` | Risdiplam | `M09AX` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `R07AX01` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `C01AA05` |
| `EC:01412` | Rifapentine | `J04AB05` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `H01AB01` | `*None*` |
| `EC:01423` | Risdiplam | `M09AX10` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `R07AX01` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `C01AA05` |
| `EC:01412` | Rifapentine | `J04AB05` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `H01AB01` | `*None*` |
| `EC:01423` | Risdiplam | `M09AX10` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `Respiratory system drugs` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `Cardiovascular system drugs` |
| `EC:01412` | Rifapentine | `Antiinfectives for systemic use` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `Systemic hormonal preparations, excl. sex hormones and insulins` | `*None*` |
| `EC:01423` | Risdiplam | `Musculo-skeletal system drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `Other respiratory system products in atc` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `Cardiac therapy drugs` |
| `EC:01412` | Rifapentine | `Antimycobacterials` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `Pituitary and hypothalamic hormones and analogues` | `*None*` |
| `EC:01423` | Risdiplam | `Other drugs for disorders of the musculo-skeletal system in atc` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `Other respiratory system products in atc` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `Cardiac glycosides` |
| `EC:01412` | Rifapentine | `Drugs for treatment of tuberculosis` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `Anterior pituitary lobe hormones and analogues` | `*None*` |
| `EC:01423` | Risdiplam | `Other drugs for disorders of the musculo-skeletal system in atc` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `Other respiratory system products in atc` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `Digitalis glycosides` |
| `EC:01412` | Rifapentine | `Antibiotics, antitubercular` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `Thyrotropin class in atc` | `*None*` |
| `EC:01423` | Risdiplam | `Other drugs for disorders of the musculo-skeletal system in atc` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01149` | Nitric oxide | `Nitric oxide` | `*None*` |
| `EC:00503` | Digoxin | `*None*` | `Digoxin` |
| `EC:01412` | Rifapentine | `Rifapentine` | `*None*` |
| `EC:01621` | Thyrotropin alfa | `Thyrotropin alfa` | `*None*` |
| `EC:01423` | Risdiplam | `Risdiplam` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 506 | 491 |
| `atc_level_2` | 506 | 491 |
| `atc_level_3` | 506 | 491 |
| `atc_level_4` | 506 | 491 |
| `atc_level_5` | 506 | 491 |
| `atc_main` | 506 | 491 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1818 | 1818 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 19 | 19 |
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
| `l1_label` | 506 | 491 |
| `l2_label` | 506 | 491 |
| `l3_label` | 506 | 491 |
| `l4_label` | 513 | 500 |
| `l5_label` | 554 | 541 |
| `name` | 0 | 0 |
| `new_id` | 1820 | 1820 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
