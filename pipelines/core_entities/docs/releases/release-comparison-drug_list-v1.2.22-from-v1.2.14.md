# Release Comparison Report

**New Release:** `v1.2.22-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.22/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.14/ec-drug-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 2

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01814` | Pantothenic acid |
| `EC:01813` | Radotinib |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 446 |
| `atc_level_2` | 452 |
| `atc_level_3` | 455 |
| `atc_level_4` | 456 |
| `atc_level_5` | 456 |
| `atc_main` | 456 |
| `l1_label` | 446 |
| `l2_label` | 452 |
| `l3_label` | 455 |
| `l4_label` | 455 |
| `l5_label` | 426 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `N` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N` |
| `EC:01101` | Naldemedine | `A` | `*None*` |
| `EC:00499` | Difenoxin | `A` | `*None*` |
| `EC:00434` | Dapagliflozin | `*None*` | `A` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `N05` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05` |
| `EC:01500` | Sodium oxybate | `N07` | `N01` |
| `EC:01101` | Naldemedine | `A06` | `*None*` |
| `EC:00499` | Difenoxin | `A07` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `N05B` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05B` |
| `EC:01500` | Sodium oxybate | `N07X` | `N01A` |
| `EC:01101` | Naldemedine | `A06A` | `*None*` |
| `EC:00499` | Difenoxin | `A07D` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `N05BA` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05BA` |
| `EC:01500` | Sodium oxybate | `N07XX` | `N01AX` |
| `EC:01101` | Naldemedine | `A06AH` | `*None*` |
| `EC:00499` | Difenoxin | `A07DA` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `N05BA06` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05BA02` |
| `EC:01500` | Sodium oxybate | `N07XX04` | `N01AX11` |
| `EC:01101` | Naldemedine | `A06AH05` | `*None*` |
| `EC:00499` | Difenoxin | `A07DA04` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `N05BA06` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05BA02` |
| `EC:01500` | Sodium oxybate | `N07XX04` | `N01AX11` |
| `EC:01101` | Naldemedine | `A06AH05` | `*None*` |
| `EC:00499` | Difenoxin | `A07DA04` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `Nervous system drugs` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Nervous system drugs` |
| `EC:01101` | Naldemedine | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:00499` | Difenoxin | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:00434` | Dapagliflozin | `*None*` | `Alimentary tract and metabolism drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `Psycholeptics` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Psycholeptics` |
| `EC:01500` | Sodium oxybate | `Other nervous system drugs in atc` | `Anesthetics` |
| `EC:01101` | Naldemedine | `Drugs for constipation` | `*None*` |
| `EC:00499` | Difenoxin | `Antidiarrheals, intestinal antiinflammatory/antiinfective agents` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `Anxiolytics` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Anxiolytics` |
| `EC:01500` | Sodium oxybate | `Other nervous system drugs in atc` | `Anesthetics, general` |
| `EC:01101` | Naldemedine | `Drugs for constipation` | `*None*` |
| `EC:00499` | Difenoxin | `Antipropulsives` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `Benzodiazepine derivative anxiolytics` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Benzodiazepine derivative anxiolytics` |
| `EC:01500` | Sodium oxybate | `Other nervous system drugs in atc` | `Other general anesthetics in atc` |
| `EC:01101` | Naldemedine | `Peripheral opioid receptor antagonists` | `*None*` |
| `EC:00499` | Difenoxin | `Antipropulsives` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00951` | Lorazepam | `*None*` | `Lorazepam` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Chlordiazepoxide` |
| `EC:01101` | Naldemedine | `Naldemedine` | `*None*` |
| `EC:00499` | Difenoxin | `Difenoxin` | `*None*` |
| `EC:00434` | Dapagliflozin | `*None*` | `Dapagliflozin` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 510 |
| `atc_level_2` | 512 | 510 |
| `atc_level_3` | 512 | 510 |
| `atc_level_4` | 512 | 510 |
| `atc_level_5` | 512 | 510 |
| `atc_main` | 512 | 510 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1805 | 1807 |
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
| `l1_label` | 512 | 510 |
| `l2_label` | 512 | 510 |
| `l3_label` | 512 | 510 |
| `l4_label` | 521 | 520 |
| `l5_label` | 565 | 557 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1808 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
