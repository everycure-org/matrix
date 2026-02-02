# Release Comparison Report

**New Release:** `v1.2.17-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.17/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 494 |
| `atc_level_2` | 496 |
| `atc_level_3` | 496 |
| `atc_level_4` | 500 |
| `atc_level_5` | 500 |
| `atc_main` | 500 |
| `l1_label` | 494 |
| `l2_label` | 496 |
| `l3_label` | 496 |
| `l4_label` | 499 |
| `l5_label` | 463 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `S` | `*None*` |
| `EC:01692` | Trimethoprim | `J` | `*None*` |
| `EC:01485` | Sildenafil | `G` | `*None*` |
| `EC:00482` | Dexlansoprazole | `A` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `A` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `S01` | `*None*` |
| `EC:01692` | Trimethoprim | `J01` | `*None*` |
| `EC:01485` | Sildenafil | `G04` | `*None*` |
| `EC:00482` | Dexlansoprazole | `A02` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `A16` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `S01E` | `*None*` |
| `EC:01692` | Trimethoprim | `J01E` | `*None*` |
| `EC:01485` | Sildenafil | `G04B` | `*None*` |
| `EC:00482` | Dexlansoprazole | `A02B` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `A16A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `S01EB` | `*None*` |
| `EC:01692` | Trimethoprim | `J01EA` | `*None*` |
| `EC:01485` | Sildenafil | `G04BE` | `*None*` |
| `EC:00482` | Dexlansoprazole | `A02BC` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `A16AB` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `S01EB05` | `*None*` |
| `EC:01692` | Trimethoprim | `J01EA01` | `*None*` |
| `EC:01485` | Sildenafil | `G04BE03` | `*None*` |
| `EC:00482` | Dexlansoprazole | `A02BC06` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `A16AB18` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `S01EB05` | `*None*` |
| `EC:01692` | Trimethoprim | `J01EA01` | `*None*` |
| `EC:01485` | Sildenafil | `G04BE03` | `*None*` |
| `EC:00482` | Dexlansoprazole | `A02BC06` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `A16AB18` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `Sensory organ drugs` | `*None*` |
| `EC:01692` | Trimethoprim | `Antiinfectives for systemic use` | `*None*` |
| `EC:01485` | Sildenafil | `Genito urinary system and sex hormones` | `*None*` |
| `EC:00482` | Dexlansoprazole | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `Alimentary tract and metabolism drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `Ophthalmologicals` | `*None*` |
| `EC:01692` | Trimethoprim | `Antibacterials for systemic use` | `*None*` |
| `EC:01485` | Sildenafil | `Urologicals` | `*None*` |
| `EC:00482` | Dexlansoprazole | `Drugs for acid related disorders` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `Other alimentary tract and metabolism products in atc` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `Antiglaucoma preparations and miotics` | `*None*` |
| `EC:01692` | Trimethoprim | `Sulfonamides and trimethoprim antibacterials for systemic use` | `*None*` |
| `EC:01485` | Sildenafil | `Urologicals` | `*None*` |
| `EC:00482` | Dexlansoprazole | `Drugs for peptic ulcer and gastro-oesophageal reflux disease (gord)` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `Other alimentary tract and metabolism products in atc` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `Parasympathomimetics, antiglaucoma preparations and miotics` | `*None*` |
| `EC:01692` | Trimethoprim | `Trimethoprim and derivatives, systemic antibacterials` | `*None*` |
| `EC:01485` | Sildenafil | `Drugs used in erectile dysfunction` | `*None*` |
| `EC:00482` | Dexlansoprazole | `Proton pump inhibitors for peptic ulcer and gord` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `Enzymes for alimentary tract and metabolism` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `Physostigmine` | `*None*` |
| `EC:01692` | Trimethoprim | `Trimethoprim` | `*None*` |
| `EC:01485` | Sildenafil | `Sildenafil` | `*None*` |
| `EC:00482` | Dexlansoprazole | `Dexlansoprazole` | `*None*` |
| `EC:01738` | Vestronidase alfa | `*None*` | `Vestronidase alfa` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 509 |
| `atc_level_2` | 512 | 509 |
| `atc_level_3` | 512 | 509 |
| `atc_level_4` | 512 | 509 |
| `atc_level_5` | 512 | 509 |
| `atc_main` | 512 | 509 |
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
| `l1_label` | 512 | 509 |
| `l2_label` | 512 | 509 |
| `l3_label` | 512 | 509 |
| `l4_label` | 521 | 519 |
| `l5_label` | 565 | 554 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1806 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
