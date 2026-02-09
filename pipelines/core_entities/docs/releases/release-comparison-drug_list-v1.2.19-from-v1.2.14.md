# Release Comparison Report

**New Release:** `v1.2.19-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.19/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.14/ec-drug-list.parquet`

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
| `atc_level_1` | 380 |
| `atc_level_2` | 384 |
| `atc_level_3` | 386 |
| `atc_level_4` | 389 |
| `atc_level_5` | 389 |
| `atc_main` | 389 |
| `l1_label` | 380 |
| `l2_label` | 384 |
| `l3_label` | 386 |
| `l4_label` | 384 |
| `l5_label` | 358 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `L` |
| `EC:00055` | Alprazolam | `*None*` | `N` |
| `EC:01215` | Oxytocin | `*None*` | `H` |
| `EC:01376` | Rabeprazole | `A` | `*None*` |
| `EC:00302` | Cefprozil | `J` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `L01` |
| `EC:00055` | Alprazolam | `*None*` | `N05` |
| `EC:01215` | Oxytocin | `*None*` | `H01` |
| `EC:01376` | Rabeprazole | `A02` | `*None*` |
| `EC:00302` | Cefprozil | `J01` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `L01X` |
| `EC:00055` | Alprazolam | `*None*` | `N05B` |
| `EC:01215` | Oxytocin | `*None*` | `H01B` |
| `EC:01376` | Rabeprazole | `A02B` | `*None*` |
| `EC:00302` | Cefprozil | `J01D` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `L01XX` |
| `EC:00055` | Alprazolam | `*None*` | `N05BA` |
| `EC:01215` | Oxytocin | `*None*` | `H01BB` |
| `EC:01376` | Rabeprazole | `A02BC` | `*None*` |
| `EC:00302` | Cefprozil | `J01DC` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `L01XX52` |
| `EC:00055` | Alprazolam | `*None*` | `N05BA12` |
| `EC:01215` | Oxytocin | `*None*` | `H01BB02` |
| `EC:01376` | Rabeprazole | `A02BC04` | `*None*` |
| `EC:00302` | Cefprozil | `J01DC10` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `L01XX52` |
| `EC:00055` | Alprazolam | `*None*` | `N05BA12` |
| `EC:01215` | Oxytocin | `*None*` | `H01BB02` |
| `EC:01376` | Rabeprazole | `A02BC04` | `*None*` |
| `EC:00302` | Cefprozil | `J01DC10` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:00055` | Alprazolam | `*None*` | `Nervous system drugs` |
| `EC:01215` | Oxytocin | `*None*` | `Systemic hormonal preparations, excl. sex hormones and insulins` |
| `EC:01376` | Rabeprazole | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:00302` | Cefprozil | `Antiinfectives for systemic use` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `Antineoplastic agents` |
| `EC:00055` | Alprazolam | `*None*` | `Psycholeptics` |
| `EC:01215` | Oxytocin | `*None*` | `Pituitary and hypothalamic hormones and analogues` |
| `EC:01376` | Rabeprazole | `Drugs for acid related disorders` | `*None*` |
| `EC:00302` | Cefprozil | `Antibacterials for systemic use` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `Other antineoplastic agents in atc` |
| `EC:00055` | Alprazolam | `*None*` | `Anxiolytics` |
| `EC:01215` | Oxytocin | `*None*` | `Posterior pituitary lobe hormones` |
| `EC:01376` | Rabeprazole | `Drugs for peptic ulcer and gastro-oesophageal reflux disease (gord)` | `*None*` |
| `EC:00302` | Cefprozil | `Other beta-lactam antibacterials in atc` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `Other antineoplastic agents in atc` |
| `EC:00055` | Alprazolam | `*None*` | `Benzodiazepine derivative anxiolytics` |
| `EC:01215` | Oxytocin | `*None*` | `Oxytocin and analogues` |
| `EC:01376` | Rabeprazole | `Proton pump inhibitors for peptic ulcer and gord` | `*None*` |
| `EC:00302` | Cefprozil | `Second-generation cephalosporins` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `*None*` | `Venetoclax` |
| `EC:00055` | Alprazolam | `*None*` | `Alprazolam` |
| `EC:01215` | Oxytocin | `*None*` | `Oxytocin` |
| `EC:01376` | Rabeprazole | `Rabeprazole` | `*None*` |
| `EC:00302` | Cefprozil | `Cefprozil` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 499 |
| `atc_level_2` | 512 | 499 |
| `atc_level_3` | 512 | 499 |
| `atc_level_4` | 512 | 499 |
| `atc_level_5` | 512 | 499 |
| `atc_main` | 512 | 499 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1805 | 1805 |
| `drug_class` | 1 | 1 |
| `drug_function` | 17 | 17 |
| `drug_target` | 23 | 23 |
| `drugbank_id` | 15 | 15 |
| `id` | 0 | 0 |
| `is_analgesic` | 0 | 0 |
| `is_antimicrobial` | 0 | 0 |
| `is_antipsychotic` | 0 | 0 |
| `is_cardiovascular` | 0 | 0 |
| `is_cell_therapy` | 0 | 0 |
| `is_chemotherapy` | 0 | 0 |
| `is_glucose_regulator` | 0 | 0 |
| `is_sedative` | 0 | 0 |
| `is_steroid` | 0 | 0 |
| `l1_label` | 512 | 499 |
| `l2_label` | 512 | 499 |
| `l3_label` | 512 | 499 |
| `l4_label` | 521 | 505 |
| `l5_label` | 565 | 547 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1806 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
