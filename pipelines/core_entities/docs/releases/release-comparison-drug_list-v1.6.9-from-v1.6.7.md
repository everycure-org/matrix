# Release Comparison Report

**New Release:** `v1.6.9-drug_list`

**Base Release:** `v1.6.7-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.9/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 487 |
| `atc_level_2` | 487 |
| `atc_level_3` | 488 |
| `atc_level_4` | 490 |
| `atc_level_5` | 490 |
| `atc_main` | 490 |
| `is_fda_generic_drug` | 1 |
| `l1_label` | 487 |
| `l2_label` | 487 |
| `l3_label` | 488 |
| `l4_label` | 486 |
| `l5_label` | 449 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `A` | `*None*` |
| `EC:01441` | Rosuvastatin | `C` | `*None*` |
| `EC:00050` | Almotriptan | `N` | `*None*` |
| `EC:01593` | Teplizumab | `A` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `J` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `A04` | `*None*` |
| `EC:01441` | Rosuvastatin | `C10` | `*None*` |
| `EC:00050` | Almotriptan | `N02` | `*None*` |
| `EC:01593` | Teplizumab | `A10` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `J01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `A04A` | `*None*` |
| `EC:01441` | Rosuvastatin | `C10A` | `*None*` |
| `EC:00050` | Almotriptan | `N02C` | `*None*` |
| `EC:01593` | Teplizumab | `A10X` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `J01X` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `A04AD` | `*None*` |
| `EC:01441` | Rosuvastatin | `C10AA` | `*None*` |
| `EC:00050` | Almotriptan | `N02CC` | `*None*` |
| `EC:01593` | Teplizumab | `A10XX` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `J01XX` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `A04AD14` | `*None*` |
| `EC:01441` | Rosuvastatin | `C10AA07` | `*None*` |
| `EC:00050` | Almotriptan | `N02CC05` | `*None*` |
| `EC:01593` | Teplizumab | `A10XX01` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `J01XX13` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `A04AD14` | `*None*` |
| `EC:01441` | Rosuvastatin | `C10AA07` | `*None*` |
| `EC:00050` | Almotriptan | `N02CC05` | `*None*` |
| `EC:01593` | Teplizumab | `A10XX01` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `J01XX13` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01441` | Rosuvastatin | `Cardiovascular system drugs` | `*None*` |
| `EC:00050` | Almotriptan | `Nervous system drugs` | `*None*` |
| `EC:01593` | Teplizumab | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `Antiinfectives for systemic use` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `Antiemetics and antinauseants` | `*None*` |
| `EC:01441` | Rosuvastatin | `Lipid modifying agents` | `*None*` |
| `EC:00050` | Almotriptan | `Analgesics` | `*None*` |
| `EC:01593` | Teplizumab | `Drugs used in diabetes` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `Antibacterials for systemic use` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `Antiemetics and antinauseants` | `*None*` |
| `EC:01441` | Rosuvastatin | `Lipid modifying agents, plain` | `*None*` |
| `EC:00050` | Almotriptan | `Antimigraine preparations` | `*None*` |
| `EC:01593` | Teplizumab | `Other drugs used in diabetes in atc` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `Other antibacterials in atc` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `Other antiemetics in atc` | `*None*` |
| `EC:01441` | Rosuvastatin | `Hmg coa reductase inhibitors, plain lipid modifying drugs` | `*None*` |
| `EC:00050` | Almotriptan | `Selective serotonin (5ht1) agonists` | `*None*` |
| `EC:01593` | Teplizumab | `Other drugs used in diabetes in atc` | `*None*` |
| `EC:00748` | Gepotidacin | `*None*` | `Other antibacterials in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01434` | Rolapitant | `Rolapitant` | `*None*` |
| `EC:01441` | Rosuvastatin | `Rosuvastatin` | `*None*` |
| `EC:00050` | Almotriptan | `Almotriptan` | `*None*` |
| `EC:01593` | Teplizumab | `Teplizumab` | `*None*` |
| `EC:00361` | Clioquinol | `*None*` | `Clioquinol` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 508 | 496 |
| `atc_level_2` | 508 | 496 |
| `atc_level_3` | 508 | 496 |
| `atc_level_4` | 508 | 496 |
| `atc_level_5` | 508 | 496 |
| `atc_main` | 508 | 496 |
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
| `l1_label` | 508 | 496 |
| `l2_label` | 508 | 496 |
| `l3_label` | 508 | 496 |
| `l4_label` | 515 | 505 |
| `l5_label` | 555 | 548 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
