# Release Comparison Report

**New Release:** `v1.6.13-drug_list`

**Base Release:** `v1.6.7-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.13/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 442 |
| `atc_level_2` | 446 |
| `atc_level_3` | 448 |
| `atc_level_4` | 449 |
| `atc_level_5` | 449 |
| `atc_main` | 449 |
| `is_fda_generic_drug` | 2 |
| `l1_label` | 442 |
| `l2_label` | 446 |
| `l3_label` | 448 |
| `l4_label` | 445 |
| `l5_label` | 407 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `L` | `*None*` |
| `EC:00152` | Aztreonam | `J` | `*None*` |
| `EC:01761` | Vortioxetine | `N` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N` |
| `EC:00748` | Gepotidacin | `*None*` | `J` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `L01` | `*None*` |
| `EC:00152` | Aztreonam | `J01` | `*None*` |
| `EC:01761` | Vortioxetine | `N06` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05` |
| `EC:00748` | Gepotidacin | `*None*` | `J01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `L01X` | `*None*` |
| `EC:00152` | Aztreonam | `J01D` | `*None*` |
| `EC:01761` | Vortioxetine | `N06A` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05B` |
| `EC:00748` | Gepotidacin | `*None*` | `J01X` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `L01XD` | `*None*` |
| `EC:00152` | Aztreonam | `J01DF` | `*None*` |
| `EC:01761` | Vortioxetine | `N06AX` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05BA` |
| `EC:00748` | Gepotidacin | `*None*` | `J01XX` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `L01XD04` | `*None*` |
| `EC:00152` | Aztreonam | `J01DF01` | `*None*` |
| `EC:01761` | Vortioxetine | `N06AX26` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05BA02` |
| `EC:00748` | Gepotidacin | `*None*` | `J01XX13` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `L01XD04` | `*None*` |
| `EC:00152` | Aztreonam | `J01DF01` | `*None*` |
| `EC:01761` | Vortioxetine | `N06AX26` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `N05BA02` |
| `EC:00748` | Gepotidacin | `*None*` | `J01XX13` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `False` | `True` |
| `EC:00465` | Desflurane | `True` | `False` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00152` | Aztreonam | `Antiinfectives for systemic use` | `*None*` |
| `EC:01761` | Vortioxetine | `Nervous system drugs` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Nervous system drugs` |
| `EC:00748` | Gepotidacin | `*None*` | `Antiinfectives for systemic use` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `Antineoplastic agents` | `*None*` |
| `EC:00152` | Aztreonam | `Antibacterials for systemic use` | `*None*` |
| `EC:01761` | Vortioxetine | `Psychoanaleptics` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Psycholeptics` |
| `EC:00748` | Gepotidacin | `*None*` | `Antibacterials for systemic use` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `Other antineoplastic agents in atc` | `*None*` |
| `EC:00152` | Aztreonam | `Other beta-lactam antibacterials in atc` | `*None*` |
| `EC:01761` | Vortioxetine | `Antidepressants` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Anxiolytics` |
| `EC:00748` | Gepotidacin | `*None*` | `Other antibacterials in atc` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `Sensitizers used in photodynamic/radiation therapy` | `*None*` |
| `EC:00152` | Aztreonam | `Monobactams` | `*None*` |
| `EC:01761` | Vortioxetine | `Other antidepressants in atc` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Benzodiazepine derivative anxiolytics` |
| `EC:00748` | Gepotidacin | `*None*` | `Other antibacterials in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00069` | Aminolevulinic acid | `Aminolevulinic acid` | `*None*` |
| `EC:00152` | Aztreonam | `Aztreonam` | `*None*` |
| `EC:01761` | Vortioxetine | `Vortioxetine` | `*None*` |
| `EC:00326` | Chlordiazepoxide | `*None*` | `Chlordiazepoxide` |
| `EC:01243` | Pazopanib | `*None*` | `Pazopanib` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 508 | 516 |
| `atc_level_2` | 508 | 516 |
| `atc_level_3` | 508 | 516 |
| `atc_level_4` | 508 | 516 |
| `atc_level_5` | 508 | 516 |
| `atc_main` | 508 | 516 |
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
| `l1_label` | 508 | 516 |
| `l2_label` | 508 | 516 |
| `l3_label` | 508 | 516 |
| `l4_label` | 515 | 523 |
| `l5_label` | 555 | 564 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
