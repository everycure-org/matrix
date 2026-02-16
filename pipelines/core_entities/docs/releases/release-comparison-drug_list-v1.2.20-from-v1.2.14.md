# Release Comparison Report

**New Release:** `v1.2.20-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.20/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.14/ec-drug-list.parquet`

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
| `atc_level_1` | 445 |
| `atc_level_2` | 449 |
| `atc_level_3` | 450 |
| `atc_level_4` | 454 |
| `atc_level_5` | 454 |
| `atc_main` | 454 |
| `l1_label` | 445 |
| `l2_label` | 449 |
| `l3_label` | 450 |
| `l4_label` | 449 |
| `l5_label` | 420 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `D` |
| `EC:01599` | Terconazole | `*None*` | `G` |
| `EC:01034` | Methylphenidate | `N` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `L` |
| `EC:01735` | Vericiguat | `*None*` | `C` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `D01` |
| `EC:01599` | Terconazole | `*None*` | `G01` |
| `EC:01034` | Methylphenidate | `N06` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `L01` |
| `EC:01735` | Vericiguat | `*None*` | `C01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `D01A` |
| `EC:01599` | Terconazole | `*None*` | `G01A` |
| `EC:01034` | Methylphenidate | `N06B` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `L01X` |
| `EC:01735` | Vericiguat | `*None*` | `C01D` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `D01AE` |
| `EC:01599` | Terconazole | `*None*` | `G01AG` |
| `EC:01034` | Methylphenidate | `N06BA` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `L01XB` |
| `EC:01735` | Vericiguat | `*None*` | `C01DX` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `D01AE18` |
| `EC:01599` | Terconazole | `*None*` | `G01AG02` |
| `EC:01034` | Methylphenidate | `N06BA04` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `L01XB01` |
| `EC:01735` | Vericiguat | `*None*` | `C01DX22` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `D01AE18` |
| `EC:01599` | Terconazole | `*None*` | `G01AG02` |
| `EC:01034` | Methylphenidate | `N06BA04` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `L01XB01` |
| `EC:01735` | Vericiguat | `*None*` | `C01DX22` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `Dermatologicals` |
| `EC:01599` | Terconazole | `*None*` | `Genito urinary system and sex hormones` |
| `EC:01034` | Methylphenidate | `Nervous system drugs` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:01735` | Vericiguat | `*None*` | `Cardiovascular system drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `Antifungals for dermatological use` |
| `EC:01599` | Terconazole | `*None*` | `Gynecological antiinfectives and antiseptics` |
| `EC:01034` | Methylphenidate | `Psychoanaleptics` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `Antineoplastic agents` |
| `EC:01735` | Vericiguat | `*None*` | `Cardiac therapy drugs` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `Antifungals for topical use` |
| `EC:01599` | Terconazole | `*None*` | `Antiinfectives and antiseptics, excl. combinations with corticosteroids` |
| `EC:01034` | Methylphenidate | `Psychostimulants, agents used for adhd and nootropics` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `Other antineoplastic agents in atc` |
| `EC:01735` | Vericiguat | `*None*` | `Vasodilators used in cardiac diseases` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `Other antifungals for topical use in atc` |
| `EC:01599` | Terconazole | `*None*` | `Triazole derivatives, gynecological antiinfectives and antiseptics` |
| `EC:01034` | Methylphenidate | `Centrally acting sympathomimetics` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `Methylhydrazines, antineoplastics` |
| `EC:01735` | Vericiguat | `*None*` | `Other vasodilators used in cardiac diseases in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01653` | Tolnaftate | `*None*` | `Tolnaftate` |
| `EC:01599` | Terconazole | `*None*` | `Terconazole` |
| `EC:01034` | Methylphenidate | `Methylphenidate` | `*None*` |
| `EC:01345` | Procarbazine | `*None*` | `Procarbazine` |
| `EC:01735` | Vericiguat | `*None*` | `Vericiguat` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 501 |
| `atc_level_2` | 512 | 501 |
| `atc_level_3` | 512 | 501 |
| `atc_level_4` | 512 | 501 |
| `atc_level_5` | 512 | 501 |
| `atc_main` | 512 | 501 |
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
| `l1_label` | 512 | 501 |
| `l2_label` | 512 | 501 |
| `l3_label` | 512 | 501 |
| `l4_label` | 521 | 507 |
| `l5_label` | 565 | 547 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1806 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
