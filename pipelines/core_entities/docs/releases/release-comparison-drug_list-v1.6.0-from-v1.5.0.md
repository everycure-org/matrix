# Release Comparison Report

**New Release:** `v1.6.0-drug_list`

**Base Release:** `v1.5.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.5.0/ec-drug-list.parquet`

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
| `EC:01821` | Beta-hydroxybutyrate |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 383 |
| `atc_level_2` | 386 |
| `atc_level_3` | 387 |
| `atc_level_4` | 387 |
| `atc_level_5` | 387 |
| `atc_main` | 387 |
| `drug_class` | 5 |
| `drug_target` | 5 |
| `l1_label` | 383 |
| `l2_label` | 386 |
| `l3_label` | 387 |
| `l4_label` | 385 |
| `l5_label` | 364 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `N` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `J` |
| `EC:01380` | Ramelteon | `*None*` | `N` |
| `EC:01682` | Trientine | `*None*` | `A` |
| `EC:00449` | Decitabine | `*None*` | `L` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `N02` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `J01` |
| `EC:01380` | Ramelteon | `*None*` | `N05` |
| `EC:01682` | Trientine | `*None*` | `A16` |
| `EC:00449` | Decitabine | `*None*` | `L01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `N02C` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `J01A` |
| `EC:01380` | Ramelteon | `*None*` | `N05C` |
| `EC:01682` | Trientine | `*None*` | `A16A` |
| `EC:00449` | Decitabine | `*None*` | `L01B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `N02CC` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `J01AA` |
| `EC:01380` | Ramelteon | `*None*` | `N05CH` |
| `EC:01682` | Trientine | `*None*` | `A16AX` |
| `EC:00449` | Decitabine | `*None*` | `L01BC` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `N02CC02` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `J01AA13` |
| `EC:01380` | Ramelteon | `*None*` | `N05CH02` |
| `EC:01682` | Trientine | `*None*` | `A16AX12` |
| `EC:00449` | Decitabine | `*None*` | `L01BC08` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `N02CC02` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `J01AA13` |
| `EC:01380` | Ramelteon | `*None*` | `N05CH02` |
| `EC:01682` | Trientine | `*None*` | `A16AX12` |
| `EC:00449` | Decitabine | `*None*` | `L01BC08` |

#### `drug_class`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00230` | Brinzolamide | `Carbonic anhydase inhibitor` | `Carbonic anhydrase inhibitor` |
| `EC:00014` | Acetazolamide | `Carbonic anhydase inhibitor` | `Carbonic anhydrase inhibitor` |
| `EC:01019` | Methazolamide | `Carbonic anhydase inhibitor` | `Carbonic anhydrase inhibitor` |
| `EC:00492` | Diclofenamide | `Carbonic anhydase inhibitor` | `Carbonic anhydrase inhibitor` |
| `EC:00532` | Dorzolamide | `Carbonic anhydase inhibitor` | `Carbonic anhydrase inhibitor` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00935` | Lisdexamfetamine | `Amphetamine deriative` | `Amphetamine derivative` |
| `EC:00485` | Dextroamphetamine | `Amphetamine deriative` | `Amphetamine dervative` |
| `EC:00182` | Benzphetamine | `Amphetamine deriative` | `Amphetamine dervative` |
| `EC:01018` | Methamphetamine | `Amphetamine deriative` | `Amphetamine dervative` |
| `EC:00497` | Diethylpropion | `Amphetamine deriative` | `Amphetamine dervative` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `Nervous system drugs` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `Antiinfectives for systemic use` |
| `EC:01380` | Ramelteon | `*None*` | `Nervous system drugs` |
| `EC:01682` | Trientine | `*None*` | `Alimentary tract and metabolism drugs` |
| `EC:00449` | Decitabine | `*None*` | `Antineoplastic and immunomodulating agents` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `Analgesics` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `Antibacterials for systemic use` |
| `EC:01380` | Ramelteon | `*None*` | `Psycholeptics` |
| `EC:01682` | Trientine | `*None*` | `Other alimentary tract and metabolism products in atc` |
| `EC:00449` | Decitabine | `*None*` | `Antineoplastic agents` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `Antimigraine preparations` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `Tetracycline antibiotics` |
| `EC:01380` | Ramelteon | `*None*` | `Hypnotics and sedatives` |
| `EC:01682` | Trientine | `*None*` | `Other alimentary tract and metabolism products in atc` |
| `EC:00449` | Decitabine | `*None*` | `Antimetabolites` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `Selective serotonin (5ht1) agonists` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `Tetracyclines` |
| `EC:01380` | Ramelteon | `*None*` | `Melatonin receptor agonists, hypnotics and sedatives` |
| `EC:01682` | Trientine | `*None*` | `Various alimentary tract and metabolism products` |
| `EC:00449` | Decitabine | `*None*` | `Pyrimidine analogues, antineoplastic antimetabolites` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01108` | Naratriptan | `Naratriptan` | `*None*` |
| `EC:00610` | Eravacycline | `*None*` | `Eravacycline` |
| `EC:01380` | Ramelteon | `*None*` | `Ramelteon` |
| `EC:01682` | Trientine | `*None*` | `Trientine` |
| `EC:00449` | Decitabine | `*None*` | `Decitabine` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 457 | 496 |
| `atc_level_2` | 457 | 496 |
| `atc_level_3` | 457 | 496 |
| `atc_level_4` | 457 | 496 |
| `atc_level_5` | 457 | 496 |
| `atc_main` | 457 | 496 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1811 | 1812 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 25 | 26 |
| `drugbank_id` | 15 | 16 |
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
| `l1_label` | 457 | 496 |
| `l2_label` | 457 | 496 |
| `l3_label` | 457 | 496 |
| `l4_label` | 465 | 506 |
| `l5_label` | 504 | 549 |
| `name` | 0 | 0 |
| `new_id` | 1813 | 1814 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
