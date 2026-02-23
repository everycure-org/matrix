# Release Comparison Report

**New Release:** `v1.2.21-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.21/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.14/ec-drug-list.parquet`

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
| `EC:01813` | Radotinib |
| `EC:01814` | Pantothenic acid |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 429 |
| `atc_level_2` | 433 |
| `atc_level_3` | 434 |
| `atc_level_4` | 438 |
| `atc_level_5` | 438 |
| `atc_main` | 438 |
| `l1_label` | 429 |
| `l2_label` | 433 |
| `l3_label` | 434 |
| `l4_label` | 437 |
| `l5_label` | 405 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `J` | `P` |
| `EC:00703` | Flupentixol | `*None*` | `N` |
| `EC:00622` | Escitalopram | `*None*` | `N` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `H` |
| `EC:01670` | Trastuzumab deruxtecan | `L` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `J01` | `P01` |
| `EC:00703` | Flupentixol | `*None*` | `N05` |
| `EC:00622` | Escitalopram | `*None*` | `N06` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `H01` |
| `EC:01670` | Trastuzumab deruxtecan | `L01` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `J01X` | `P01A` |
| `EC:00703` | Flupentixol | `*None*` | `N05A` |
| `EC:00622` | Escitalopram | `*None*` | `N06A` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `H01A` |
| `EC:01670` | Trastuzumab deruxtecan | `L01F` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `J01XD` | `P01AB` |
| `EC:00703` | Flupentixol | `*None*` | `N05AF` |
| `EC:00622` | Escitalopram | `*None*` | `N06AB` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `H01AB` |
| `EC:01670` | Trastuzumab deruxtecan | `L01FD` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `J01XD02` | `P01AB02` |
| `EC:00703` | Flupentixol | `*None*` | `N05AF01` |
| `EC:00622` | Escitalopram | `*None*` | `N06AB10` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `H01AB01` |
| `EC:01670` | Trastuzumab deruxtecan | `L01FD04` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `J01XD02` | `P01AB02` |
| `EC:00703` | Flupentixol | `*None*` | `N05AF01` |
| `EC:00622` | Escitalopram | `*None*` | `N06AB10` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `H01AB01` |
| `EC:01670` | Trastuzumab deruxtecan | `L01FD04` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `Antiinfectives for systemic use` | `Antiparasitic products, insecticides and repellents` |
| `EC:00703` | Flupentixol | `*None*` | `Nervous system drugs` |
| `EC:00622` | Escitalopram | `*None*` | `Nervous system drugs` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `Systemic hormonal preparations, excl. sex hormones and insulins` |
| `EC:01670` | Trastuzumab deruxtecan | `Antineoplastic and immunomodulating agents` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `Antibacterials for systemic use` | `Antiprotozoals` |
| `EC:00703` | Flupentixol | `*None*` | `Psycholeptics` |
| `EC:00622` | Escitalopram | `*None*` | `Psychoanaleptics` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `Pituitary and hypothalamic hormones and analogues` |
| `EC:01670` | Trastuzumab deruxtecan | `Antineoplastic agents` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `Other antibacterials in atc` | `Agents against amoebiasis and other protozoal diseases` |
| `EC:00703` | Flupentixol | `*None*` | `Antipsychotics` |
| `EC:00622` | Escitalopram | `*None*` | `Antidepressants` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `Anterior pituitary lobe hormones and analogues` |
| `EC:01670` | Trastuzumab deruxtecan | `Monoclonal antibodies and antibody drug conjugates` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01630` | Tinidazole | `Imidazole derivatives, antibacterial for systemic use` | `Nitroimidazole derivatives, antiprotozoal agents against amoebiasis and other protozoal diseases` |
| `EC:00703` | Flupentixol | `*None*` | `Thioxanthene derivatives, antipsychotic` |
| `EC:00622` | Escitalopram | `*None*` | `Selective serotonin reuptake inhibitors` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `Thyrotropin class in atc` |
| `EC:01670` | Trastuzumab deruxtecan | `Her2 (human epidermal growth factor receptor 2) inhibitors` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00703` | Flupentixol | `*None*` | `Flupentixol` |
| `EC:00622` | Escitalopram | `*None*` | `Escitalopram` |
| `EC:01621` | Thyrotropin alfa | `*None*` | `Thyrotropin alfa` |
| `EC:01670` | Trastuzumab deruxtecan | `Trastuzumab deruxtecan` | `*None*` |
| `EC:01275` | Phenoxybenzamine | `*None*` | `Phenoxybenzamine` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 452 |
| `atc_level_2` | 512 | 452 |
| `atc_level_3` | 512 | 452 |
| `atc_level_4` | 512 | 452 |
| `atc_level_5` | 512 | 452 |
| `atc_main` | 512 | 452 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1805 | 1807 |
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
| `l1_label` | 512 | 452 |
| `l2_label` | 512 | 452 |
| `l3_label` | 512 | 452 |
| `l4_label` | 521 | 462 |
| `l5_label` | 565 | 506 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1808 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
