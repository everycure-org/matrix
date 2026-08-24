# Release Comparison Report

**New Release:** `v1.6.29-drug_list`

**Base Release:** `v1.6.25-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.29/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.25/ec-drug-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 1

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01857` | Tranilast |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 411 |
| `atc_level_2` | 415 |
| `atc_level_3` | 417 |
| `atc_level_4` | 419 |
| `atc_level_5` | 419 |
| `atc_main` | 419 |
| `l1_label` | 411 |
| `l2_label` | 415 |
| `l3_label` | 417 |
| `l4_label` | 416 |
| `l5_label` | 384 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `N` |
| `EC:01308` | Polidocanol | `*None*` | `C` |
| `EC:00584` | Enalapril | `C` | `*None*` |
| `EC:01059` | Minocycline | `J` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `L` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `N05` |
| `EC:01308` | Polidocanol | `*None*` | `C05` |
| `EC:00584` | Enalapril | `C09` | `*None*` |
| `EC:01059` | Minocycline | `J01` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `L01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `N05B` |
| `EC:01308` | Polidocanol | `*None*` | `C05B` |
| `EC:00584` | Enalapril | `C09A` | `*None*` |
| `EC:01059` | Minocycline | `J01A` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `L01F` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `N05BA` |
| `EC:01308` | Polidocanol | `*None*` | `C05BB` |
| `EC:00584` | Enalapril | `C09AA` | `*None*` |
| `EC:01059` | Minocycline | `J01AA` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `L01FX` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `N05BA12` |
| `EC:01308` | Polidocanol | `*None*` | `C05BB02` |
| `EC:00584` | Enalapril | `C09AA02` | `*None*` |
| `EC:01059` | Minocycline | `J01AA08` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `L01FX14` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `N05BA12` |
| `EC:01308` | Polidocanol | `*None*` | `C05BB02` |
| `EC:00584` | Enalapril | `C09AA02` | `*None*` |
| `EC:01059` | Minocycline | `J01AA08` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `L01FX14` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `Nervous system drugs` |
| `EC:01308` | Polidocanol | `*None*` | `Cardiovascular system drugs` |
| `EC:00584` | Enalapril | `Cardiovascular system drugs` | `*None*` |
| `EC:01059` | Minocycline | `Antiinfectives for systemic use` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `Antineoplastic and immunomodulating agents` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `Psycholeptics` |
| `EC:01308` | Polidocanol | `*None*` | `Vasoprotectives` |
| `EC:00584` | Enalapril | `Agents acting on the renin-angiotensin system` | `*None*` |
| `EC:01059` | Minocycline | `Antibacterials for systemic use` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `Antineoplastic agents` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `Anxiolytics` |
| `EC:01308` | Polidocanol | `*None*` | `Antivaricose therapy drugs` |
| `EC:00584` | Enalapril | `Ace inhibitors, plain` | `*None*` |
| `EC:01059` | Minocycline | `Tetracycline antibiotics` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `Monoclonal antibodies and antibody drug conjugates` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `Benzodiazepine derivative anxiolytics` |
| `EC:01308` | Polidocanol | `*None*` | `Sclerosing agents for local injection` |
| `EC:00584` | Enalapril | `Ace inhibitors, plain` | `*None*` |
| `EC:01059` | Minocycline | `Tetracyclines` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `Other monoclonal antibodies and antibody drug conjugates in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00055` | Alprazolam | `*None*` | `Alprazolam` |
| `EC:01308` | Polidocanol | `*None*` | `Polidocanol` |
| `EC:00584` | Enalapril | `Enalapril` | `*None*` |
| `EC:01059` | Minocycline | `Minocycline` | `*None*` |
| `EC:01307` | Polatuzumab vedotin | `*None*` | `Polatuzumab vedotin` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 501 | 506 |
| `atc_level_2` | 501 | 506 |
| `atc_level_3` | 501 | 506 |
| `atc_level_4` | 501 | 506 |
| `atc_level_5` | 501 | 506 |
| `atc_main` | 501 | 506 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1817 | 1818 |
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
| `l1_label` | 501 | 506 |
| `l2_label` | 501 | 506 |
| `l3_label` | 501 | 506 |
| `l4_label` | 511 | 513 |
| `l5_label` | 553 | 554 |
| `name` | 0 | 0 |
| `new_id` | 1819 | 1820 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
