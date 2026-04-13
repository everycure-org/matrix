# Release Comparison Report

**New Release:** `v1.6.1-drug_list`

**Base Release:** `v1.6.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.1/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.0/ec-drug-list.parquet`

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
| `atc_level_1` | 448 |
| `atc_level_2` | 449 |
| `atc_level_3` | 451 |
| `atc_level_4` | 452 |
| `atc_level_5` | 452 |
| `atc_main` | 452 |
| `drug_target` | 1 |
| `l1_label` | 448 |
| `l2_label` | 449 |
| `l3_label` | 451 |
| `l4_label` | 451 |
| `l5_label` | 425 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `A` |
| `EC:01630` | Tinidazole | `P` | `J` |
| `EC:00570` | Eletriptan | `*None*` | `N` |
| `EC:00778` | Haloperidol | `N` | `*None*` |
| `EC:01678` | Triamcinolone | `H` | `R` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `A10` |
| `EC:01630` | Tinidazole | `P01` | `J01` |
| `EC:00570` | Eletriptan | `*None*` | `N02` |
| `EC:00778` | Haloperidol | `N05` | `*None*` |
| `EC:01678` | Triamcinolone | `H02` | `R03` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `A10B` |
| `EC:01630` | Tinidazole | `P01A` | `J01X` |
| `EC:00570` | Eletriptan | `*None*` | `N02C` |
| `EC:00778` | Haloperidol | `N05A` | `*None*` |
| `EC:01678` | Triamcinolone | `H02A` | `R03B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `A10BK` |
| `EC:01630` | Tinidazole | `P01AB` | `J01XD` |
| `EC:00570` | Eletriptan | `*None*` | `N02CC` |
| `EC:00778` | Haloperidol | `N05AD` | `*None*` |
| `EC:01678` | Triamcinolone | `H02AB` | `R03BA` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `A10BK08` |
| `EC:01630` | Tinidazole | `P01AB02` | `J01XD02` |
| `EC:00570` | Eletriptan | `*None*` | `N02CC06` |
| `EC:00778` | Haloperidol | `N05AD01` | `*None*` |
| `EC:01678` | Triamcinolone | `H02AB08` | `R03BA06` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `A10BK08` |
| `EC:01630` | Tinidazole | `P01AB02` | `J01XD02` |
| `EC:00570` | Eletriptan | `*None*` | `N02CC06` |
| `EC:00778` | Haloperidol | `N05AD01` | `*None*` |
| `EC:01678` | Triamcinolone | `H02AB08` | `R03BA06` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00553` | Dyclonine | `Sodium channel inhibitor` | `TRPV3 inhibitor, SCN1A inhibitor` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `Alimentary tract and metabolism drugs` |
| `EC:01630` | Tinidazole | `Antiparasitic products, insecticides and repellents` | `Antiinfectives for systemic use` |
| `EC:00570` | Eletriptan | `*None*` | `Nervous system drugs` |
| `EC:00778` | Haloperidol | `Nervous system drugs` | `*None*` |
| `EC:01678` | Triamcinolone | `Systemic hormonal preparations, excl. sex hormones and insulins` | `Respiratory system drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `Drugs used in diabetes` |
| `EC:01630` | Tinidazole | `Antiprotozoals` | `Antibacterials for systemic use` |
| `EC:00570` | Eletriptan | `*None*` | `Analgesics` |
| `EC:00778` | Haloperidol | `Psycholeptics` | `*None*` |
| `EC:01678` | Triamcinolone | `Corticosteroids for systemic use` | `Drugs for obstructive airway diseases` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `Blood glucose lowering drugs, excl. insulins` |
| `EC:01630` | Tinidazole | `Agents against amoebiasis and other protozoal diseases` | `Other antibacterials in atc` |
| `EC:00570` | Eletriptan | `*None*` | `Antimigraine preparations` |
| `EC:00778` | Haloperidol | `Antipsychotics` | `*None*` |
| `EC:01678` | Triamcinolone | `Corticosteroids for systemic use, plain` | `Other drugs for obstructive airway diseases, inhalants in atc` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `Sodium-glucose co-transporter 2 (sglt2) inhibitors` |
| `EC:01630` | Tinidazole | `Nitroimidazole derivatives, antiprotozoal agents against amoebiasis and other protozoal diseases` | `Imidazole derivatives, antibacterial for systemic use` |
| `EC:00570` | Eletriptan | `*None*` | `Selective serotonin (5ht1) agonists` |
| `EC:00778` | Haloperidol | `Butyrophenone derivatives, antipsychotics` | `*None*` |
| `EC:01678` | Triamcinolone | `Glucocorticoids, systemic` | `Glucocorticoid inhalants for obstructive airway disease` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00199` | Bexagliflozin | `*None*` | `Bexagliflozin` |
| `EC:00570` | Eletriptan | `*None*` | `Eletriptan` |
| `EC:00778` | Haloperidol | `Haloperidol` | `*None*` |
| `EC:01036` | Methyltestosterone | `Methyltestosterone` | `*None*` |
| `EC:00430` | Danaparoid | `Danaparoid` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 496 | 504 |
| `atc_level_2` | 496 | 504 |
| `atc_level_3` | 496 | 504 |
| `atc_level_4` | 496 | 504 |
| `atc_level_5` | 496 | 504 |
| `atc_main` | 496 | 504 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1812 | 1812 |
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
| `l1_label` | 496 | 504 |
| `l2_label` | 496 | 504 |
| `l3_label` | 496 | 504 |
| `l4_label` | 506 | 513 |
| `l5_label` | 549 | 556 |
| `name` | 0 | 0 |
| `new_id` | 1814 | 1814 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
