# Release Comparison Report

**New Release:** `v1.3.4-drug_list`

**Base Release:** `v1.3.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.3.4/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.3.0/ec-drug-list.parquet`

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
| `EC:01819` | Ambroxol |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 274 |
| `atc_level_2` | 278 |
| `atc_level_3` | 280 |
| `atc_level_4` | 280 |
| `atc_level_5` | 280 |
| `atc_main` | 280 |
| `drug_target` | 5 |
| `l1_label` | 274 |
| `l2_label` | 278 |
| `l3_label` | 280 |
| `l4_label` | 280 |
| `l5_label` | 256 |
| `synonyms` | 3 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `A` | `*None*` |
| `EC:01667` | Tranexamic acid | `B` | `*None*` |
| `EC:01416` | Riluzole | `N` | `*None*` |
| `EC:00750` | Givinostat | `M` | `*None*` |
| `EC:00298` | Cefotaxime | `J` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `A16` | `*None*` |
| `EC:01667` | Tranexamic acid | `B02` | `*None*` |
| `EC:01416` | Riluzole | `N07` | `*None*` |
| `EC:00750` | Givinostat | `M09` | `*None*` |
| `EC:00298` | Cefotaxime | `J01` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `A16A` | `*None*` |
| `EC:01667` | Tranexamic acid | `B02A` | `*None*` |
| `EC:01416` | Riluzole | `N07X` | `*None*` |
| `EC:00750` | Givinostat | `M09A` | `*None*` |
| `EC:00298` | Cefotaxime | `J01D` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `A16AX` | `*None*` |
| `EC:01667` | Tranexamic acid | `B02AA` | `*None*` |
| `EC:01416` | Riluzole | `N07XX` | `*None*` |
| `EC:00750` | Givinostat | `M09AX` | `*None*` |
| `EC:00298` | Cefotaxime | `J01DD` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `A16AX12` | `*None*` |
| `EC:01667` | Tranexamic acid | `B02AA02` | `*None*` |
| `EC:01416` | Riluzole | `N07XX02` | `*None*` |
| `EC:00750` | Givinostat | `M09AX14` | `*None*` |
| `EC:00298` | Cefotaxime | `J01DD01` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `A16AX12` | `*None*` |
| `EC:01667` | Tranexamic acid | `B02AA02` | `*None*` |
| `EC:01416` | Riluzole | `N07XX02` | `*None*` |
| `EC:00750` | Givinostat | `M09AX14` | `*None*` |
| `EC:00298` | Cefotaxime | `J01DD01` | `*None*` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00803` | Ibuprofen | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00827` | Indomethacin | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00707` | Flurbiprofen | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00869` | Ketorolac | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |
| `EC:00868` | Ketoprofen | `COX-1 and COX-2 inihibitor` | `COX-1 and COX-2 inhibitor` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01667` | Tranexamic acid | `Blood and blood forming organ drugs` | `*None*` |
| `EC:01416` | Riluzole | `Nervous system drugs` | `*None*` |
| `EC:00750` | Givinostat | `Musculo-skeletal system drugs` | `*None*` |
| `EC:00298` | Cefotaxime | `Antiinfectives for systemic use` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `Other alimentary tract and metabolism products in atc` | `*None*` |
| `EC:01667` | Tranexamic acid | `Antihemorrhagics` | `*None*` |
| `EC:01416` | Riluzole | `Other nervous system drugs in atc` | `*None*` |
| `EC:00750` | Givinostat | `Other drugs for disorders of the musculo-skeletal system in atc` | `*None*` |
| `EC:00298` | Cefotaxime | `Antibacterials for systemic use` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `Other alimentary tract and metabolism products in atc` | `*None*` |
| `EC:01667` | Tranexamic acid | `Antifibrinolytics` | `*None*` |
| `EC:01416` | Riluzole | `Other nervous system drugs in atc` | `*None*` |
| `EC:00750` | Givinostat | `Other drugs for disorders of the musculo-skeletal system in atc` | `*None*` |
| `EC:00298` | Cefotaxime | `Other beta-lactam antibacterials in atc` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `Various alimentary tract and metabolism products` | `*None*` |
| `EC:01667` | Tranexamic acid | `Antifibrinolytic amino acids` | `*None*` |
| `EC:01416` | Riluzole | `Other nervous system drugs in atc` | `*None*` |
| `EC:00750` | Givinostat | `Other drugs for disorders of the musculo-skeletal system in atc` | `*None*` |
| `EC:00298` | Cefotaxime | `Third-generation cephalosporins` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01682` | Trientine | `Trientine` | `*None*` |
| `EC:01667` | Tranexamic acid | `Tranexamic acid` | `*None*` |
| `EC:01416` | Riluzole | `Riluzole` | `*None*` |
| `EC:00750` | Givinostat | `Givinostat` | `*None*` |
| `EC:00298` | Cefotaxime | `Cefotaxime` | `*None*` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00962` | Lumasiran | `['Antisense oligonucleotide']` | `['']` |
| `EC:00860` | Ivacaftor | `['Cftr potentiator']` | `['']` |
| `EC:00835` | Interferon alfa | `['Peginterferon alfa' 'Interferon alfa natural']` | `['Peginterferon alfa' 'Interferon alfa natural' 'Interferon alpha'
 'Interferon-alpha' 'Interfero...` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 243 | 507 |
| `atc_level_2` | 243 | 507 |
| `atc_level_3` | 243 | 507 |
| `atc_level_4` | 243 | 507 |
| `atc_level_5` | 243 | 507 |
| `atc_main` | 243 | 507 |
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
| `is_glucose_regulator` | 0 | 0 |
| `is_sedative` | 0 | 0 |
| `is_steroid` | 0 | 0 |
| `l1_label` | 243 | 507 |
| `l2_label` | 243 | 507 |
| `l3_label` | 243 | 507 |
| `l4_label` | 253 | 517 |
| `l5_label` | 303 | 559 |
| `name` | 0 | 0 |
| `new_id` | 1812 | 1813 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
