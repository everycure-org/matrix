# Release Comparison Report

**New Release:** `v1.6.14-drug_list`

**Base Release:** `v1.6.7-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.14/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 479 |
| `atc_level_2` | 484 |
| `atc_level_3` | 487 |
| `atc_level_4` | 490 |
| `atc_level_5` | 490 |
| `atc_main` | 490 |
| `is_fda_generic_drug` | 6 |
| `l1_label` | 479 |
| `l2_label` | 484 |
| `l3_label` | 487 |
| `l4_label` | 486 |
| `l5_label` | 443 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `N` | `*None*` |
| `EC:00451` | Deferasirox | `V` | `*None*` |
| `EC:01192` | Ondansetron | `A` | `*None*` |
| `EC:00680` | Filgotinib | `L` | `*None*` |
| `EC:00015` | Acetylcysteine | `R` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `N06` | `*None*` |
| `EC:00451` | Deferasirox | `V03` | `*None*` |
| `EC:01192` | Ondansetron | `A04` | `*None*` |
| `EC:00680` | Filgotinib | `L04` | `*None*` |
| `EC:00015` | Acetylcysteine | `R05` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `N06B` | `*None*` |
| `EC:00451` | Deferasirox | `V03A` | `*None*` |
| `EC:01192` | Ondansetron | `A04A` | `*None*` |
| `EC:00680` | Filgotinib | `L04A` | `*None*` |
| `EC:00015` | Acetylcysteine | `R05C` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `N06BA` | `*None*` |
| `EC:00451` | Deferasirox | `V03AC` | `*None*` |
| `EC:01192` | Ondansetron | `A04AA` | `*None*` |
| `EC:00680` | Filgotinib | `L04AF` | `*None*` |
| `EC:00015` | Acetylcysteine | `R05CB` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `N06BA07` | `*None*` |
| `EC:00451` | Deferasirox | `V03AC03` | `*None*` |
| `EC:01192` | Ondansetron | `A04AA01` | `*None*` |
| `EC:00680` | Filgotinib | `L04AF04` | `*None*` |
| `EC:00015` | Acetylcysteine | `R05CB01` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `N06BA07` | `*None*` |
| `EC:00451` | Deferasirox | `V03AC03` | `*None*` |
| `EC:01192` | Ondansetron | `A04AA01` | `*None*` |
| `EC:00680` | Filgotinib | `L04AF04` | `*None*` |
| `EC:00015` | Acetylcysteine | `R05CB01` | `*None*` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01412` | Rifapentine | `False` | `True` |
| `EC:00155` | Baloxavir | `False` | `True` |
| `EC:00722` | Fostamatinib | `False` | `True` |
| `EC:00465` | Desflurane | `True` | `False` |
| `EC:01507` | Solriamfetol | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `Nervous system drugs` | `*None*` |
| `EC:00451` | Deferasirox | `Various drug classes in atc` | `*None*` |
| `EC:01192` | Ondansetron | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:00680` | Filgotinib | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00015` | Acetylcysteine | `Respiratory system drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `Psychoanaleptics` | `*None*` |
| `EC:00451` | Deferasirox | `All other therapeutic products` | `*None*` |
| `EC:01192` | Ondansetron | `Antiemetics and antinauseants` | `*None*` |
| `EC:00680` | Filgotinib | `Immunosuppressants` | `*None*` |
| `EC:00015` | Acetylcysteine | `Cough and cold preparations` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `Psychostimulants, agents used for adhd and nootropics` | `*None*` |
| `EC:00451` | Deferasirox | `All other therapeutic products` | `*None*` |
| `EC:01192` | Ondansetron | `Antiemetics and antinauseants` | `*None*` |
| `EC:00680` | Filgotinib | `Immunosuppressants` | `*None*` |
| `EC:00015` | Acetylcysteine | `Expectorants, excl. combinations with cough suppressants` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `Centrally acting sympathomimetics` | `*None*` |
| `EC:00451` | Deferasirox | `Iron chelating agents` | `*None*` |
| `EC:01192` | Ondansetron | `Serotonin (5ht3) antagonists` | `*None*` |
| `EC:00680` | Filgotinib | `Janus-associated kinase (jak) inhibitors (l04af)` | `*None*` |
| `EC:00015` | Acetylcysteine | `Mucolytics` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01074` | Modafinil | `Modafinil` | `*None*` |
| `EC:00451` | Deferasirox | `Deferasirox` | `*None*` |
| `EC:01192` | Ondansetron | `Ondansetron` | `*None*` |
| `EC:00680` | Filgotinib | `Filgotinib` | `*None*` |
| `EC:00015` | Acetylcysteine | `Acetylcysteine` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 508 | 499 |
| `atc_level_2` | 508 | 499 |
| `atc_level_3` | 508 | 499 |
| `atc_level_4` | 508 | 499 |
| `atc_level_5` | 508 | 499 |
| `atc_main` | 508 | 499 |
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
| `l1_label` | 508 | 499 |
| `l2_label` | 508 | 499 |
| `l3_label` | 508 | 499 |
| `l4_label` | 515 | 508 |
| `l5_label` | 555 | 546 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
