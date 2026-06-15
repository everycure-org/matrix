# Release Comparison Report

**New Release:** `v1.6.12-drug_list`

**Base Release:** `v1.6.7-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.12/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.7/ec-drug-list.parquet`

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
| `atc_level_1` | 461 |
| `atc_level_2` | 465 |
| `atc_level_3` | 466 |
| `atc_level_4` | 471 |
| `atc_level_5` | 471 |
| `atc_main` | 471 |
| `is_fda_generic_drug` | 1 |
| `l1_label` | 461 |
| `l2_label` | 465 |
| `l3_label` | 466 |
| `l4_label` | 468 |
| `l5_label` | 433 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `N` |
| `EC:01722` | Vandetanib | `L` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `S` |
| `EC:01518` | Sparsentan | `*None*` | `C` |
| `EC:01176` | Oliceridine | `N` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `N05` |
| `EC:01722` | Vandetanib | `L01` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `S01` |
| `EC:01518` | Sparsentan | `*None*` | `C09` |
| `EC:01176` | Oliceridine | `N02` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `N05C` |
| `EC:01722` | Vandetanib | `L01E` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `S01E` |
| `EC:01518` | Sparsentan | `*None*` | `C09X` |
| `EC:01176` | Oliceridine | `N02A` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `N05CJ` |
| `EC:01722` | Vandetanib | `L01EX` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `S01EE` |
| `EC:01518` | Sparsentan | `*None*` | `C09XX` |
| `EC:01176` | Oliceridine | `N02AX` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `N05CJ02` |
| `EC:01722` | Vandetanib | `L01EX04` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `S01EE06` |
| `EC:01518` | Sparsentan | `*None*` | `C09XX01` |
| `EC:01176` | Oliceridine | `N02AX07` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `N05CJ02` |
| `EC:01722` | Vandetanib | `L01EX04` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `S01EE06` |
| `EC:01518` | Sparsentan | `*None*` | `C09XX01` |
| `EC:01176` | Oliceridine | `N02AX07` | `*None*` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `Nervous system drugs` |
| `EC:01722` | Vandetanib | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `Sensory organ drugs` |
| `EC:01518` | Sparsentan | `*None*` | `Cardiovascular system drugs` |
| `EC:01176` | Oliceridine | `Nervous system drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `Psycholeptics` |
| `EC:01722` | Vandetanib | `Antineoplastic agents` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `Ophthalmologicals` |
| `EC:01518` | Sparsentan | `*None*` | `Agents acting on the renin-angiotensin system` |
| `EC:01176` | Oliceridine | `Analgesics` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `Hypnotics and sedatives` |
| `EC:01722` | Vandetanib | `Protein kinase inhibitors, antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `Antiglaucoma preparations and miotics` |
| `EC:01518` | Sparsentan | `*None*` | `Other agents acting on the renin-angiotensin system in atc` |
| `EC:01176` | Oliceridine | `Opioid analgesics` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `Orexin receptor antagonists` |
| `EC:01722` | Vandetanib | `Other protein kinase inhibitors in atc` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `Prostaglandin analogues, antiglaucoma drugs and miotics` |
| `EC:01176` | Oliceridine | `Other opioids in atc` | `*None*` |
| `EC:01493` | Sipuleucel-t | `*None*` | `Other immunostimulants in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00895` | Lemborexant | `*None*` | `Lemborexant` |
| `EC:01722` | Vandetanib | `Vandetanib` | `*None*` |
| `EC:00888` | Latanoprostene bunod | `*None*` | `Latanoprostene bunod` |
| `EC:01176` | Oliceridine | `Oliceridine` | `*None*` |
| `EC:01493` | Sipuleucel-t | `*None*` | `Sipuleucel-t` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 508 | 491 |
| `atc_level_2` | 508 | 491 |
| `atc_level_3` | 508 | 491 |
| `atc_level_4` | 508 | 491 |
| `atc_level_5` | 508 | 491 |
| `atc_main` | 508 | 491 |
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
| `l1_label` | 508 | 491 |
| `l2_label` | 508 | 491 |
| `l3_label` | 508 | 491 |
| `l4_label` | 515 | 501 |
| `l5_label` | 555 | 546 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
