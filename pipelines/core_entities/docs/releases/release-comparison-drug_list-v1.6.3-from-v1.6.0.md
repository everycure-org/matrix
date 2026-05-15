# Release Comparison Report

**New Release:** `v1.6.3-drug_list`

**Base Release:** `v1.6.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.3/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.0/ec-drug-list.parquet`

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
| `EC:01822` | Adenine |
| `EC:01823` | Trehalose |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 443 |
| `atc_level_2` | 445 |
| `atc_level_3` | 447 |
| `atc_level_4` | 448 |
| `atc_level_5` | 448 |
| `atc_main` | 448 |
| `drug_class` | 5 |
| `drug_target` | 2 |
| `is_fda_generic_drug` | 4 |
| `l1_label` | 443 |
| `l2_label` | 445 |
| `l3_label` | 447 |
| `l4_label` | 448 |
| `l5_label` | 421 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `L` |
| `EC:00782` | Homatropine | `S` | `*None*` |
| `EC:01553` | Tafenoquine | `P` | `*None*` |
| `EC:01392` | Remdesivir | `J` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `N` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `L04` |
| `EC:00782` | Homatropine | `S01` | `*None*` |
| `EC:01553` | Tafenoquine | `P01` | `*None*` |
| `EC:01392` | Remdesivir | `J05` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `N04` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `L04A` |
| `EC:00782` | Homatropine | `S01F` | `*None*` |
| `EC:01553` | Tafenoquine | `P01B` | `*None*` |
| `EC:01392` | Remdesivir | `J05A` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `N04B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `L04AF` |
| `EC:00782` | Homatropine | `S01FA` | `*None*` |
| `EC:01553` | Tafenoquine | `P01BA` | `*None*` |
| `EC:01392` | Remdesivir | `J05AB` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `N04BB` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `L04AF07` |
| `EC:00782` | Homatropine | `S01FA05` | `*None*` |
| `EC:01553` | Tafenoquine | `P01BA07` | `*None*` |
| `EC:01392` | Remdesivir | `J05AB16` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `N04BB01` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `L04AF07` |
| `EC:00782` | Homatropine | `S01FA05` | `*None*` |
| `EC:01553` | Tafenoquine | `P01BA07` | `*None*` |
| `EC:01392` | Remdesivir | `J05AB16` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `N04BB01` |

#### `drug_class`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01495` | Sitagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:00051` | Alogliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:01463` | Saxagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:00927` | Linagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:01742` | Vildagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00051` | Alogliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:00553` | Dyclonine | `Sodium channel inhibitor` | `TRPV3 inhibitor, SCN1A inhibitor` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00150` | Azilsartan | `False` | `True` |
| `EC:00261` | Canagliflozin | `False` | `True` |
| `EC:01320` | Potassium phosphate | `True` | `False` |
| `EC:00597` | Enzalutamide | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:00782` | Homatropine | `Sensory organ drugs` | `*None*` |
| `EC:01553` | Tafenoquine | `Antiparasitic products, insecticides and repellents` | `*None*` |
| `EC:01392` | Remdesivir | `Antiinfectives for systemic use` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `Nervous system drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `Immunosuppressants` |
| `EC:00782` | Homatropine | `Ophthalmologicals` | `*None*` |
| `EC:01553` | Tafenoquine | `Antiprotozoals` | `*None*` |
| `EC:01392` | Remdesivir | `Antivirals for systemic use` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `Anti-parkinson drugs` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `Immunosuppressants` |
| `EC:00782` | Homatropine | `Mydriatics and cycloplegics` | `*None*` |
| `EC:01553` | Tafenoquine | `Antimalarials` | `*None*` |
| `EC:01392` | Remdesivir | `Direct acting antivirals` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `Dopaminergic agents` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `Janus-associated kinase (jak) inhibitors (l04af)` |
| `EC:00782` | Homatropine | `Anticholinergic mydriatics and cycloplegics` | `*None*` |
| `EC:01553` | Tafenoquine | `Aminoquinoline antimalarials` | `*None*` |
| `EC:01392` | Remdesivir | `Nucleosides and nucleotides excl. reverse transcriptase inhibitors` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `Adamantane derivatives, dopaminergic` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00474` | Deucravacitinib | `*None*` | `Deucravacitinib` |
| `EC:00782` | Homatropine | `Homatropine` | `*None*` |
| `EC:01553` | Tafenoquine | `Tafenoquine` | `*None*` |
| `EC:01392` | Remdesivir | `Remdesivir` | `*None*` |
| `EC:00060` | Amantadine | `*None*` | `Amantadine` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 496 | 500 |
| `atc_level_2` | 496 | 500 |
| `atc_level_3` | 496 | 500 |
| `atc_level_4` | 496 | 500 |
| `atc_level_5` | 496 | 500 |
| `atc_main` | 496 | 500 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1812 | 1814 |
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
| `l1_label` | 496 | 500 |
| `l2_label` | 496 | 500 |
| `l3_label` | 496 | 500 |
| `l4_label` | 506 | 510 |
| `l5_label` | 549 | 556 |
| `name` | 0 | 0 |
| `new_id` | 1814 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
