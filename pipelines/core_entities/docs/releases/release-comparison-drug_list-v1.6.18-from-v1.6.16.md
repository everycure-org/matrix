# Release Comparison Report

**New Release:** `v1.6.18-drug_list`

**Base Release:** `v1.6.16-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.18/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.16/ec-drug-list.parquet`

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
| `EC:01824` | D-ribose |
| `EC:01825` | Intravenous fish oil |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 420 |
| `atc_level_2` | 422 |
| `atc_level_3` | 424 |
| `atc_level_4` | 426 |
| `atc_level_5` | 426 |
| `atc_main` | 426 |
| `l1_label` | 420 |
| `l2_label` | 422 |
| `l3_label` | 424 |
| `l4_label` | 421 |
| `l5_label` | 388 |
| `synonyms` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `A` |
| `EC:01325` | Pralsetinib | `*None*` | `L` |
| `EC:01324` | Pralidoxime | `*None*` | `V` |
| `EC:01518` | Sparsentan | `*None*` | `C` |
| `EC:01792` | Nesiritide | `C` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `A12` |
| `EC:01325` | Pralsetinib | `*None*` | `L01` |
| `EC:01324` | Pralidoxime | `*None*` | `V03` |
| `EC:01518` | Sparsentan | `*None*` | `C09` |
| `EC:01792` | Nesiritide | `C01` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `A12B` |
| `EC:01325` | Pralsetinib | `*None*` | `L01E` |
| `EC:01324` | Pralidoxime | `*None*` | `V03A` |
| `EC:01518` | Sparsentan | `*None*` | `C09X` |
| `EC:01792` | Nesiritide | `C01D` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `A12BA` |
| `EC:01325` | Pralsetinib | `*None*` | `L01EX` |
| `EC:01324` | Pralidoxime | `*None*` | `V03AB` |
| `EC:01518` | Sparsentan | `*None*` | `C09XX` |
| `EC:01792` | Nesiritide | `C01DX` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `A12BA01` |
| `EC:01325` | Pralsetinib | `*None*` | `L01EX23` |
| `EC:01324` | Pralidoxime | `*None*` | `V03AB04` |
| `EC:01518` | Sparsentan | `*None*` | `C09XX01` |
| `EC:01792` | Nesiritide | `C01DX19` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `A12BA01` |
| `EC:01325` | Pralsetinib | `*None*` | `L01EX23` |
| `EC:01324` | Pralidoxime | `*None*` | `V03AB04` |
| `EC:01518` | Sparsentan | `*None*` | `C09XX01` |
| `EC:01792` | Nesiritide | `C01DX19` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `Alimentary tract and metabolism drugs` |
| `EC:01325` | Pralsetinib | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:01324` | Pralidoxime | `*None*` | `Various drug classes in atc` |
| `EC:01518` | Sparsentan | `*None*` | `Cardiovascular system drugs` |
| `EC:01792` | Nesiritide | `Cardiovascular system drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `Mineral supplements` |
| `EC:01325` | Pralsetinib | `*None*` | `Antineoplastic agents` |
| `EC:01324` | Pralidoxime | `*None*` | `All other therapeutic products` |
| `EC:01518` | Sparsentan | `*None*` | `Agents acting on the renin-angiotensin system` |
| `EC:01792` | Nesiritide | `Cardiac therapy drugs` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `Potassium supplements` |
| `EC:01325` | Pralsetinib | `*None*` | `Protein kinase inhibitors, antineoplastic and immunomodulating agents` |
| `EC:01324` | Pralidoxime | `*None*` | `All other therapeutic products` |
| `EC:01518` | Sparsentan | `*None*` | `Other agents acting on the renin-angiotensin system in atc` |
| `EC:01792` | Nesiritide | `Vasodilators used in cardiac diseases` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `Potassium` |
| `EC:01325` | Pralsetinib | `*None*` | `Other protein kinase inhibitors in atc` |
| `EC:01324` | Pralidoxime | `*None*` | `Antidotes` |
| `EC:01792` | Nesiritide | `Other vasodilators used in cardiac diseases in atc` | `*None*` |
| `EC:00919` | Levomilnacipran | `*None*` | `Other antidepressants in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01317` | Potassium chloride | `*None*` | `Potassium chloride` |
| `EC:01325` | Pralsetinib | `*None*` | `Pralsetinib` |
| `EC:01324` | Pralidoxime | `*None*` | `Pralidoxime` |
| `EC:01792` | Nesiritide | `Nesiritide` | `*None*` |
| `EC:00919` | Levomilnacipran | `*None*` | `Levomilnacipran` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01451` | Sacubitril | `[]` | `['Entresto']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 478 | 487 |
| `atc_level_2` | 478 | 487 |
| `atc_level_3` | 478 | 487 |
| `atc_level_4` | 478 | 487 |
| `atc_level_5` | 478 | 487 |
| `atc_main` | 478 | 487 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1814 | 1816 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 16 | 18 |
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
| `l1_label` | 478 | 487 |
| `l2_label` | 478 | 487 |
| `l3_label` | 478 | 487 |
| `l4_label` | 485 | 495 |
| `l5_label` | 529 | 535 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1818 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
