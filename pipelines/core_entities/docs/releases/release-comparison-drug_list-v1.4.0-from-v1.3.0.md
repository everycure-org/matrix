# Release Comparison Report

**New Release:** `v1.4.0-drug_list`

**Base Release:** `v1.3.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.4.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.3.0/ec-drug-list.parquet`

## Column Changes

### Added Columns
- `is_fda_generic_drug`

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 1

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01819` | Ambroxol |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `aggregated_with` | 1789 |
| `atc_level_1` | 273 |
| `atc_level_2` | 276 |
| `atc_level_3` | 278 |
| `atc_level_4` | 280 |
| `atc_level_5` | 280 |
| `atc_main` | 280 |
| `deleted` | 1 |
| `deleted_reason` | 1 |
| `drug_class` | 2 |
| `drug_function` | 1 |
| `drug_target` | 8 |
| `l1_label` | 273 |
| `l2_label` | 276 |
| `l3_label` | 278 |
| `l4_label` | 278 |
| `l5_label` | 252 |
| `new_id` | 1 |
| `synonyms` | 1424 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `aggregated_with`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `['']` | `[]` |
| `EC:01164` | Nystatin | `['']` | `[]` |
| `EC:01059` | Minocycline | `['']` | `[]` |
| `EC:00655` | Exemestane | `['']` | `[]` |
| `EC:00034` | Agomelatine | `['']` | `[]` |

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `S` | `*None*` |
| `EC:01164` | Nystatin | `A` | `*None*` |
| `EC:00854` | Isotretinoin | `D` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `C` | `*None*` |
| `EC:00859` | Ivabradine | `C` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `S01` | `*None*` |
| `EC:01164` | Nystatin | `A07` | `*None*` |
| `EC:00854` | Isotretinoin | `D10` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `C01` | `*None*` |
| `EC:00859` | Ivabradine | `C01` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `S01A` | `*None*` |
| `EC:01164` | Nystatin | `A07A` | `*None*` |
| `EC:00854` | Isotretinoin | `D10A` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `C01D` | `*None*` |
| `EC:00859` | Ivabradine | `C01E` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `S01AE` | `*None*` |
| `EC:01164` | Nystatin | `A07AA` | `*None*` |
| `EC:00854` | Isotretinoin | `D10AD` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `C01DA` | `*None*` |
| `EC:00859` | Ivabradine | `C01EB` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `S01AE08` | `*None*` |
| `EC:01164` | Nystatin | `A07AA02` | `*None*` |
| `EC:00854` | Isotretinoin | `D10AD04` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `C01DA14` | `*None*` |
| `EC:00859` | Ivabradine | `C01EB17` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `S01AE08` | `*None*` |
| `EC:01164` | Nystatin | `A07AA02` | `*None*` |
| `EC:00854` | Isotretinoin | `D10AD04` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `C01DA14` | `*None*` |
| `EC:00859` | Ivabradine | `C01EB17` | `*None*` |

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
| `EC:00645` | Etomidate | `Non-barbituate hypnotic` | `Non-barbiturate hypnotic` |
| `EC:01355` | Propofol | `Non-barbituate hypnotic` | `Non-barbiturate hypnotic` |

#### `drug_function`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00219` | Botulinum toxin type a | `NMJ inihibitor` | `NMJ inhibitor` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00995` | Mechlorethamine | `DNA replication inihibtor` | `DNA replication inhibitor` |
| `EC:00868` | Ketoprofen | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00803` | Ibuprofen | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00869` | Ketorolac | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00827` | Indomethacin | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `Sensory organ drugs` | `*None*` |
| `EC:01164` | Nystatin | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:00854` | Isotretinoin | `Dermatologicals` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `Cardiovascular system drugs` | `*None*` |
| `EC:00859` | Ivabradine | `Cardiovascular system drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `Ophthalmologicals` | `*None*` |
| `EC:01164` | Nystatin | `Antidiarrheals, intestinal antiinflammatory/antiinfective agents` | `*None*` |
| `EC:00854` | Isotretinoin | `Anti-acne preparations` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `Cardiac therapy drugs` | `*None*` |
| `EC:00859` | Ivabradine | `Cardiac therapy drugs` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `Antiinfective ophthalmologics` | `*None*` |
| `EC:01164` | Nystatin | `Intestinal antiinfectives` | `*None*` |
| `EC:00854` | Isotretinoin | `Anti-acne preparations for topical use` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `Vasodilators used in cardiac diseases` | `*None*` |
| `EC:00859` | Ivabradine | `Other cardiac preparations in atc` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `Fluoroquinolone antiinfectives, ophthalmologic` | `*None*` |
| `EC:01164` | Nystatin | `Antibiotics, intestinal` | `*None*` |
| `EC:00854` | Isotretinoin | `Retinoids for topical use in acne` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `Organic nitrates used in cardiac disease` | `*None*` |
| `EC:00859` | Ivabradine | `Other plain cardiac preparations in atc` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `Besifloxacin` | `*None*` |
| `EC:01164` | Nystatin | `Nystatin` | `*None*` |
| `EC:00854` | Isotretinoin | `Isotretinoin` | `*None*` |
| `EC:00853` | Isosorbide mononitrate | `Isosorbide mononitrate` | `*None*` |
| `EC:00859` | Ivabradine | `Ivabradine` | `*None*` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00615` | Ergonovine | `*None*` | `EC:00614` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00191` | Besifloxacin | `['']` | `[]` |
| `EC:01164` | Nystatin | `['']` | `[]` |
| `EC:01059` | Minocycline | `['']` | `[]` |
| `EC:00655` | Exemestane | `['']` | `[]` |
| `EC:00034` | Agomelatine | `['']` | `[]` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 243 | 503 |
| `atc_level_2` | 243 | 503 |
| `atc_level_3` | 243 | 503 |
| `atc_level_4` | 243 | 503 |
| `atc_level_5` | 243 | 503 |
| `atc_main` | 243 | 503 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1810 | 1810 |
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
| `l1_label` | 243 | 503 |
| `l2_label` | 243 | 503 |
| `l3_label` | 243 | 503 |
| `l4_label` | 253 | 511 |
| `l5_label` | 303 | 555 |
| `name` | 0 | 0 |
| `new_id` | 1812 | 1812 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
