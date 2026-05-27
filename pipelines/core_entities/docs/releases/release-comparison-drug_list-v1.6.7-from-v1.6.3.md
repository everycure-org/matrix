# Release Comparison Report

**New Release:** `v1.6.7-drug_list`

**Base Release:** `v1.6.3-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.7/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.3/ec-drug-list.parquet`

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
| `atc_level_1` | 469 |
| `atc_level_2` | 471 |
| `atc_level_3` | 472 |
| `atc_level_4` | 474 |
| `atc_level_5` | 474 |
| `atc_main` | 474 |
| `drug_target` | 8 |
| `is_fda_generic_drug` | 3 |
| `l1_label` | 469 |
| `l2_label` | 471 |
| `l3_label` | 472 |
| `l4_label` | 471 |
| `l5_label` | 439 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `N` |
| `EC:01648` | Tofersen | `*None*` | `N` |
| `EC:00127` | Atorvastatin | `*None*` | `C` |
| `EC:01531` | Sucralfate | `*None*` | `A` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `N` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `N05` |
| `EC:01648` | Tofersen | `*None*` | `N07` |
| `EC:00127` | Atorvastatin | `*None*` | `C10` |
| `EC:01531` | Sucralfate | `*None*` | `A02` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `N04` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `N05A` |
| `EC:01648` | Tofersen | `*None*` | `N07X` |
| `EC:00127` | Atorvastatin | `*None*` | `C10A` |
| `EC:01531` | Sucralfate | `*None*` | `A02B` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `N04A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `N05AD` |
| `EC:01648` | Tofersen | `*None*` | `N07XX` |
| `EC:00127` | Atorvastatin | `*None*` | `C10AA` |
| `EC:01531` | Sucralfate | `*None*` | `A02BX` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `N04AA` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `N05AD01` |
| `EC:01648` | Tofersen | `*None*` | `N07XX22` |
| `EC:00127` | Atorvastatin | `*None*` | `C10AA05` |
| `EC:01531` | Sucralfate | `*None*` | `A02BX02` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `N04AA01` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `N05AD01` |
| `EC:01648` | Tofersen | `*None*` | `N07XX22` |
| `EC:00127` | Atorvastatin | `*None*` | `C10AA05` |
| `EC:01531` | Sucralfate | `*None*` | `A02BX02` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `N04AA01` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00756` | Glimepiride | `Stimulates insulin release from the pancreatic beta cells` | `SUR1/KIR6.2 channel inhibitor` |
| `EC:01660` | Torsemide | `Inhibit the Na+/K+/2Cl- cotransporter in the thick ascending limb of the loop of Henle` | `NKCC2 cotransporter inhibitor` |
| `EC:00757` | Glipizide | `Stimulates insulin release from the pancreatic beta cells` | `SUR1/KIR6.2 channel inhibitor` |
| `EC:00729` | Furosemide | `Inhibit the Na+/K+/2Cl- cotransporter in the thick ascending limb of the loop of Henle` | `NKCC2 cotransporter inhibitor` |
| `EC:00755` | Gliclazide | `Stimulates insulin release from the pancreatic beta cells` | `SUR1/KIR6.2 channel inhibitor` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01320` | Potassium phosphate | `False` | `True` |
| `EC:00145` | Axitinib | `True` | `False` |
| `EC:01618` | Thonzylamine | `True` | `False` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `Nervous system drugs` |
| `EC:01648` | Tofersen | `*None*` | `Nervous system drugs` |
| `EC:00127` | Atorvastatin | `*None*` | `Cardiovascular system drugs` |
| `EC:01531` | Sucralfate | `*None*` | `Alimentary tract and metabolism drugs` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `Nervous system drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `Psycholeptics` |
| `EC:01648` | Tofersen | `*None*` | `Other nervous system drugs in atc` |
| `EC:00127` | Atorvastatin | `*None*` | `Lipid modifying agents` |
| `EC:01531` | Sucralfate | `*None*` | `Drugs for acid related disorders` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `Anti-parkinson drugs` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `Antipsychotics` |
| `EC:01648` | Tofersen | `*None*` | `Other nervous system drugs in atc` |
| `EC:00127` | Atorvastatin | `*None*` | `Lipid modifying agents, plain` |
| `EC:01531` | Sucralfate | `*None*` | `Drugs for peptic ulcer and gastro-oesophageal reflux disease (gord)` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `Anticholinergic agents` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `Butyrophenone derivatives, antipsychotics` |
| `EC:01648` | Tofersen | `*None*` | `Other nervous system drugs in atc` |
| `EC:00127` | Atorvastatin | `*None*` | `Hmg coa reductase inhibitors, plain lipid modifying drugs` |
| `EC:01531` | Sucralfate | `*None*` | `Other drugs for peptic ulcer and gastro-oesophageal reflux disease (gord) in atc` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `Tertiary amines, anticholinergic` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00778` | Haloperidol | `*None*` | `Haloperidol` |
| `EC:01648` | Tofersen | `*None*` | `Tofersen` |
| `EC:00127` | Atorvastatin | `*None*` | `Atorvastatin` |
| `EC:01531` | Sucralfate | `*None*` | `Sucralfate` |
| `EC:01687` | Trihexyphenidyl | `*None*` | `Trihexyphenidyl` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 500 | 508 |
| `atc_level_2` | 500 | 508 |
| `atc_level_3` | 500 | 508 |
| `atc_level_4` | 500 | 508 |
| `atc_level_5` | 500 | 508 |
| `atc_main` | 500 | 508 |
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
| `l1_label` | 500 | 508 |
| `l2_label` | 500 | 508 |
| `l3_label` | 500 | 508 |
| `l4_label` | 510 | 515 |
| `l5_label` | 556 | 555 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
