# Release Comparison Report

**New Release:** `v1.5.0-drug_list`

**Base Release:** `v1.3.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.5.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.3.0/ec-drug-list.parquet`

## Column Changes

### Added Columns
- `is_fda_generic_drug`

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 2

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01820` | Riboflavin |
| `EC:01819` | Ambroxol |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `aggregated_with` | 1789 |
| `atc_level_1` | 225 |
| `atc_level_2` | 228 |
| `atc_level_3` | 230 |
| `atc_level_4` | 231 |
| `atc_level_5` | 231 |
| `atc_main` | 231 |
| `deleted` | 1 |
| `deleted_reason` | 1 |
| `drug_class` | 2 |
| `drug_function` | 1 |
| `drug_target` | 8 |
| `l1_label` | 225 |
| `l2_label` | 228 |
| `l3_label` | 230 |
| `l4_label` | 229 |
| `l5_label` | 201 |
| `new_id` | 1 |
| `synonyms` | 1424 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `aggregated_with`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `['']` | `[]` |
| `EC:00662` | Fedratinib | `['']` | `[]` |
| `EC:01297` | Pitavastatin | `['']` | `[]` |
| `EC:01315` | Porfimer sodium | `['']` | `[]` |
| `EC:00348` | Ciprofibrate | `['']` | `[]` |

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `A` | `*None*` |
| `EC:01297` | Pitavastatin | `C` | `*None*` |
| `EC:01315` | Porfimer sodium | `L` | `*None*` |
| `EC:00575` | Elotuzumab | `L` | `*None*` |
| `EC:00724` | Framycetin | `D` | `S` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `A10` | `*None*` |
| `EC:01297` | Pitavastatin | `C10` | `*None*` |
| `EC:01315` | Porfimer sodium | `L01` | `*None*` |
| `EC:00575` | Elotuzumab | `L01` | `*None*` |
| `EC:00724` | Framycetin | `D09` | `S01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `A10B` | `*None*` |
| `EC:01297` | Pitavastatin | `C10A` | `*None*` |
| `EC:01315` | Porfimer sodium | `L01X` | `*None*` |
| `EC:00575` | Elotuzumab | `L01F` | `*None*` |
| `EC:00724` | Framycetin | `D09A` | `S01A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `A10BK` | `*None*` |
| `EC:01297` | Pitavastatin | `C10AA` | `*None*` |
| `EC:01315` | Porfimer sodium | `L01XD` | `*None*` |
| `EC:00575` | Elotuzumab | `L01FX` | `*None*` |
| `EC:00724` | Framycetin | `D09AA` | `S01AA` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `A10BK03` | `*None*` |
| `EC:01297` | Pitavastatin | `C10AA08` | `*None*` |
| `EC:01315` | Porfimer sodium | `L01XD01` | `*None*` |
| `EC:00575` | Elotuzumab | `L01FX08` | `*None*` |
| `EC:00724` | Framycetin | `D09AA01` | `S01AA07` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `A10BK03` | `*None*` |
| `EC:01297` | Pitavastatin | `C10AA08` | `*None*` |
| `EC:01315` | Porfimer sodium | `L01XD01` | `*None*` |
| `EC:00575` | Elotuzumab | `L01FX08` | `*None*` |
| `EC:00724` | Framycetin | `D09AA01` | `S01AA07` |

#### `deleted`
*(Full comparison of all changed values)*

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00615` | Ergonovine | `False` | `True` |

#### `deleted_reason`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00615` | Ergonovine | `*None*` | `synonym of EC:00614` |

#### `drug_class`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01355` | Propofol | `Non-barbituate hypnotic` | `Non-barbiturate hypnotic` |
| `EC:00645` | Etomidate | `Non-barbituate hypnotic` | `Non-barbiturate hypnotic` |

#### `drug_function`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00219` | Botulinum toxin type a | `NMJ inihibitor` | `NMJ inhibitor` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00660` | Faricimab | `VEGF inihibitor \| Anti-Angiopoietin-2` | `VEGF inhibitor \| Anti-Angiopoietin-2` |
| `EC:00707` | Flurbiprofen | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00827` | Indomethacin | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00046` | Alirocumab | `PCSK9 inihibitor` | `PCSK9 inhibitor` |
| `EC:00869` | Ketorolac | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01297` | Pitavastatin | `Cardiovascular system drugs` | `*None*` |
| `EC:01315` | Porfimer sodium | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00575` | Elotuzumab | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00724` | Framycetin | `Dermatologicals` | `Sensory organ drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `Drugs used in diabetes` | `*None*` |
| `EC:01297` | Pitavastatin | `Lipid modifying agents` | `*None*` |
| `EC:01315` | Porfimer sodium | `Antineoplastic agents` | `*None*` |
| `EC:00575` | Elotuzumab | `Antineoplastic agents` | `*None*` |
| `EC:00724` | Framycetin | `Medicated dressings` | `Ophthalmologicals` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `Blood glucose lowering drugs, excl. insulins` | `*None*` |
| `EC:01297` | Pitavastatin | `Lipid modifying agents, plain` | `*None*` |
| `EC:01315` | Porfimer sodium | `Other antineoplastic agents in atc` | `*None*` |
| `EC:00575` | Elotuzumab | `Monoclonal antibodies and antibody drug conjugates` | `*None*` |
| `EC:00724` | Framycetin | `Medicated dressings` | `Antiinfective ophthalmologics` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `Sodium-glucose co-transporter 2 (sglt2) inhibitors` | `*None*` |
| `EC:01297` | Pitavastatin | `Hmg coa reductase inhibitors, plain lipid modifying drugs` | `*None*` |
| `EC:01315` | Porfimer sodium | `Sensitizers used in photodynamic/radiation therapy` | `*None*` |
| `EC:00575` | Elotuzumab | `Other monoclonal antibodies and antibody drug conjugates in atc` | `*None*` |
| `EC:00724` | Framycetin | `Medicated dressings with antiinfectives` | `Antibiotics, ophthalmologic` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `Empagliflozin` | `*None*` |
| `EC:01297` | Pitavastatin | `Pitavastatin` | `*None*` |
| `EC:01315` | Porfimer sodium | `Porfimer sodium` | `*None*` |
| `EC:00575` | Elotuzumab | `Elotuzumab` | `*None*` |
| `EC:00580` | Emapalumab | `Emapalumab` | `*None*` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00615` | Ergonovine | `*None*` | `EC:00614` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00582` | Empagliflozin | `['']` | `[]` |
| `EC:00662` | Fedratinib | `['']` | `[]` |
| `EC:01297` | Pitavastatin | `['']` | `[]` |
| `EC:00348` | Ciprofibrate | `['']` | `[]` |
| `EC:00927` | Linagliptin | `['']` | `[]` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 243 | 457 |
| `atc_level_2` | 243 | 457 |
| `atc_level_3` | 243 | 457 |
| `atc_level_4` | 243 | 457 |
| `atc_level_5` | 243 | 457 |
| `atc_main` | 243 | 457 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1810 | 1811 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 24 | 25 |
| `drugbank_id` | 15 | 15 |
| `id` | 0 | 0 |
| `is_analgesic` | 0 | 0 |
| `is_antimicrobial` | 0 | 0 |
| `is_antipsychotic` | 0 | 0 |
| `is_cardiovascular` | 0 | 0 |
| `is_cell_therapy` | 0 | 0 |
| `is_chemotherapy` | 0 | 0 |
| `is_fda_generic_drug` | N/A | 0 |
| `is_glucose_regulator` | 0 | 0 |
| `is_sedative` | 0 | 0 |
| `is_steroid` | 0 | 0 |
| `l1_label` | 243 | 457 |
| `l2_label` | 243 | 457 |
| `l3_label` | 243 | 457 |
| `l4_label` | 253 | 465 |
| `l5_label` | 303 | 504 |
| `name` | 0 | 0 |
| `new_id` | 1812 | 1813 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
