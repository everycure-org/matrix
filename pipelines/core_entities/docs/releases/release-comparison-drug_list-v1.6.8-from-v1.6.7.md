# Release Comparison Report

**New Release:** `v1.6.8-drug_list`

**Base Release:** `v1.6.7-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.8/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 409 |
| `atc_level_2` | 412 |
| `atc_level_3` | 414 |
| `atc_level_4` | 417 |
| `atc_level_5` | 417 |
| `atc_main` | 417 |
| `l1_label` | 409 |
| `l2_label` | 412 |
| `l3_label` | 414 |
| `l4_label` | 410 |
| `l5_label` | 376 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `L` | `*None*` |
| `EC:00140` | Avelumab | `L` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `J` |
| `EC:01346` | Prochlorperazine | `N` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `A` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `L04` | `*None*` |
| `EC:00140` | Avelumab | `L01` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `J01` |
| `EC:01346` | Prochlorperazine | `N05` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `A10` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `L04A` | `*None*` |
| `EC:00140` | Avelumab | `L01F` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `J01X` |
| `EC:01346` | Prochlorperazine | `N05A` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `A10B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `L04AC` | `*None*` |
| `EC:00140` | Avelumab | `L01FF` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `J01XX` |
| `EC:01346` | Prochlorperazine | `N05AB` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `A10BH` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `L04AC08` | `*None*` |
| `EC:00140` | Avelumab | `L01FF04` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `J01XX05` |
| `EC:01346` | Prochlorperazine | `N05AB04` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `A10BH02` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `L04AC08` | `*None*` |
| `EC:00140` | Avelumab | `L01FF04` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `J01XX05` |
| `EC:01346` | Prochlorperazine | `N05AB04` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `A10BH02` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00140` | Avelumab | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `Antiinfectives for systemic use` |
| `EC:01346` | Prochlorperazine | `Nervous system drugs` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `Alimentary tract and metabolism drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `Immunosuppressants` | `*None*` |
| `EC:00140` | Avelumab | `Antineoplastic agents` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `Antibacterials for systemic use` |
| `EC:01346` | Prochlorperazine | `Psycholeptics` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `Drugs used in diabetes` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `Immunosuppressants` | `*None*` |
| `EC:00140` | Avelumab | `Monoclonal antibodies and antibody drug conjugates` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `Other antibacterials in atc` |
| `EC:01346` | Prochlorperazine | `Antipsychotics` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `Blood glucose lowering drugs, excl. insulins` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `Interleukin inhibitors` | `*None*` |
| `EC:00140` | Avelumab | `Pd-1/pdl-1 (programmed cell death protein 1/death ligand 1) inhibitors` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `Other antibacterials in atc` |
| `EC:01346` | Prochlorperazine | `Phenothiazines with piperazine structure, antipsychotics` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `Dipeptidyl peptidase 4 (dpp-4) inhibitors for blood glucose lowering` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00262` | Canakinumab | `Canakinumab` | `*None*` |
| `EC:00140` | Avelumab | `Avelumab` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `Methenamine` |
| `EC:01346` | Prochlorperazine | `Prochlorperazine` | `*None*` |
| `EC:01742` | Vildagliptin | `*None*` | `Vildagliptin` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 508 | 494 |
| `atc_level_2` | 508 | 494 |
| `atc_level_3` | 508 | 494 |
| `atc_level_4` | 508 | 494 |
| `atc_level_5` | 508 | 494 |
| `atc_main` | 508 | 494 |
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
| `l1_label` | 508 | 494 |
| `l2_label` | 508 | 494 |
| `l3_label` | 508 | 494 |
| `l4_label` | 515 | 500 |
| `l5_label` | 555 | 539 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
