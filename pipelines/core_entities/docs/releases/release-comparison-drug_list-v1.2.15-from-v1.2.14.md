# Release Comparison Report

**New Release:** `v1.2.15-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.15/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 422 |
| `atc_level_2` | 424 |
| `atc_level_3` | 425 |
| `atc_level_4` | 431 |
| `atc_level_5` | 431 |
| `atc_main` | 431 |
| `l1_label` | 422 |
| `l2_label` | 424 |
| `l3_label` | 425 |
| `l4_label` | 430 |
| `l5_label` | 389 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `J` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `J` |
| `EC:00120` | Ataluren | `*None*` | `M` |
| `EC:01203` | Oxaliplatin | `L` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `R` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `J02` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `J05` |
| `EC:00120` | Ataluren | `*None*` | `M09` |
| `EC:01203` | Oxaliplatin | `L01` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `R06` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `J02A` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `J05A` |
| `EC:00120` | Ataluren | `*None*` | `M09A` |
| `EC:01203` | Oxaliplatin | `L01X` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `R06A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `J02AX` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `J05AX` |
| `EC:00120` | Ataluren | `*None*` | `M09AX` |
| `EC:01203` | Oxaliplatin | `L01XA` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `R06AB` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `J02AX01` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `J05AX09` |
| `EC:00120` | Ataluren | `*None*` | `M09AX03` |
| `EC:01203` | Oxaliplatin | `L01XA03` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `R06AB06` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `J02AX01` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `J05AX09` |
| `EC:00120` | Ataluren | `*None*` | `M09AX03` |
| `EC:01203` | Oxaliplatin | `L01XA03` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `R06AB06` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `Antiinfectives for systemic use` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `Antiinfectives for systemic use` |
| `EC:00120` | Ataluren | `*None*` | `Musculo-skeletal system drugs` |
| `EC:01203` | Oxaliplatin | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `Respiratory system drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `Antimycotics for systemic use` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `Antivirals for systemic use` |
| `EC:00120` | Ataluren | `*None*` | `Other drugs for disorders of the musculo-skeletal system in atc` |
| `EC:01203` | Oxaliplatin | `Antineoplastic agents` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `Antihistamines for systemic use` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `Antimycotics for systemic use` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `Direct acting antivirals` |
| `EC:00120` | Ataluren | `*None*` | `Other drugs for disorders of the musculo-skeletal system in atc` |
| `EC:01203` | Oxaliplatin | `Other antineoplastic agents in atc` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `Antihistamines for systemic use` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `Other antimycotics for systemic use in atc` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `Other antivirals in atc` |
| `EC:00120` | Ataluren | `*None*` | `Other drugs for disorders of the musculo-skeletal system in atc` |
| `EC:01203` | Oxaliplatin | `Platinum compounds, antineoplastic drugs` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `Substituted alkylamines, systemic antihistamines` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00692` | Flucytosine | `Flucytosine` | `*None*` |
| `EC:00985` | Maraviroc | `*None*` | `Maraviroc` |
| `EC:00120` | Ataluren | `*None*` | `Ataluren` |
| `EC:01203` | Oxaliplatin | `Oxaliplatin` | `*None*` |
| `EC:00479` | Dexbrompheniramine | `*None*` | `Dexbrompheniramine` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 505 |
| `atc_level_2` | 512 | 505 |
| `atc_level_3` | 512 | 505 |
| `atc_level_4` | 512 | 505 |
| `atc_level_5` | 512 | 505 |
| `atc_main` | 512 | 505 |
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
| `l1_label` | 512 | 505 |
| `l2_label` | 512 | 505 |
| `l3_label` | 512 | 505 |
| `l4_label` | 521 | 515 |
| `l5_label` | 565 | 548 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1806 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
