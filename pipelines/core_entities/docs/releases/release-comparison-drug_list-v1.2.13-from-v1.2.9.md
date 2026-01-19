# Release Comparison Report

**New Release:** `v1.2.13-drug_list`

**Base Release:** `v1.2.9-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.13/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.9/ec-drug-list.parquet`

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
| `synonyms` | 26 |

### Examples by Column

*Up to 5 examples per column*

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01257` | Pentamidine | `['Pentamidine isetionate']` | `['Pentamidine isetionate' 'Pentamidine isethionate']` |
| `EC:00836` | Interferon beta | `['Interferon beta' 'Human interferon beta']` | `['Interferon beta' 'Human interferon beta' 'Interferon beta natural']` |
| `EC:00359` | Clidinium | `['']` | `['Clidinium and psycholeptics']` |
| `EC:01477` | Senna | `['Sennosides']` | `['Sennosides' 'Senna glycosides']` |
| `EC:01726` | Vasopressin | `['']` | `['Vasopressin (argipressin)']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 58 | 58 |
| `atc_level_2` | 83 | 83 |
| `atc_level_3` | 115 | 115 |
| `atc_level_4` | 247 | 247 |
| `atc_level_5` | 1489 | 1489 |
| `atc_main` | 58 | 58 |
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
| `l1_label` | 58 | 58 |
| `l2_label` | 83 | 83 |
| `l3_label` | 115 | 115 |
| `l4_label` | 247 | 247 |
| `l5_label` | 1489 | 1489 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1806 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
