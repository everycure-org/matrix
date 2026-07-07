# Release Comparison Report

**New Release:** `v1.6.17-drug_list`

**Base Release:** `v1.6.16-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.17/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.16/ec-drug-list.parquet`

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
| `atc_level_1` | 437 |
| `atc_level_2` | 439 |
| `atc_level_3` | 440 |
| `atc_level_4` | 442 |
| `atc_level_5` | 442 |
| `atc_main` | 442 |
| `is_fda_generic_drug` | 1 |
| `l1_label` | 437 |
| `l2_label` | 439 |
| `l3_label` | 440 |
| `l4_label` | 439 |
| `l5_label` | 405 |
| `synonyms` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `N` |
| `EC:00287` | Carteolol | `C` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `C` |
| `EC:00286` | Carmustine | `L` | `*None*` |
| `EC:00278` | Carbinoxamine | `R` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `N07` |
| `EC:00287` | Carteolol | `C07` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `C02` |
| `EC:00286` | Carmustine | `L01` | `*None*` |
| `EC:00278` | Carbinoxamine | `R06` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `N07A` |
| `EC:00287` | Carteolol | `C07A` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `C02K` |
| `EC:00286` | Carmustine | `L01A` | `*None*` |
| `EC:00278` | Carbinoxamine | `R06A` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `N07AB` |
| `EC:00287` | Carteolol | `C07AA` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `C02KN` |
| `EC:00286` | Carmustine | `L01AD` | `*None*` |
| `EC:00278` | Carbinoxamine | `R06AA` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `N07AB02` |
| `EC:00287` | Carteolol | `C07AA15` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `C02KN01` |
| `EC:00286` | Carmustine | `L01AD01` | `*None*` |
| `EC:00278` | Carbinoxamine | `R06AA08` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `N07AB02` |
| `EC:00287` | Carteolol | `C07AA15` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `C02KN01` |
| `EC:00286` | Carmustine | `L01AD01` | `*None*` |
| `EC:00278` | Carbinoxamine | `R06AA08` | `*None*` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00209` | Bisacodyl | `True` | `False` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `Nervous system drugs` |
| `EC:00287` | Carteolol | `Cardiovascular system drugs` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `Cardiovascular system drugs` |
| `EC:00286` | Carmustine | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00278` | Carbinoxamine | `Respiratory system drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `Other nervous system drugs in atc` |
| `EC:00287` | Carteolol | `Beta-adrenergic blocking agents` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `Antihypertensives` |
| `EC:00286` | Carmustine | `Antineoplastic agents` | `*None*` |
| `EC:00278` | Carbinoxamine | `Antihistamines for systemic use` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `Parasympathomimetics` |
| `EC:00287` | Carteolol | `Beta blocking agents` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `Other antihypertensives in atc` |
| `EC:00286` | Carmustine | `Alkylating agents` | `*None*` |
| `EC:00278` | Carbinoxamine | `Antihistamines for systemic use` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `Choline esters, parasympathomimetics` |
| `EC:00287` | Carteolol | `Beta blocking agents, non-selective` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `Other antihypertensives in atc` |
| `EC:00286` | Carmustine | `Nitrosoureas, antineoplastic alkylating agents` | `*None*` |
| `EC:00278` | Carbinoxamine | `Aminoalkyl ethers, systemic antihistamines` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00196` | Bethanechol | `*None*` | `Bethanechol` |
| `EC:00287` | Carteolol | `Carteolol` | `*None*` |
| `EC:00103` | Aprocitentan | `*None*` | `Aprocitentan` |
| `EC:00286` | Carmustine | `Carmustine` | `*None*` |
| `EC:00278` | Carbinoxamine | `Carbinoxamine` | `*None*` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01451` | Sacubitril | `[]` | `['Entresto']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 478 | 495 |
| `atc_level_2` | 478 | 495 |
| `atc_level_3` | 478 | 495 |
| `atc_level_4` | 478 | 495 |
| `atc_level_5` | 478 | 495 |
| `atc_main` | 478 | 495 |
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
| `l1_label` | 478 | 495 |
| `l2_label` | 478 | 495 |
| `l3_label` | 478 | 495 |
| `l4_label` | 485 | 503 |
| `l5_label` | 529 | 544 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
