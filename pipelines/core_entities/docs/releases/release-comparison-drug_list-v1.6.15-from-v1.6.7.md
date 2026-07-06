# Release Comparison Report

**New Release:** `v1.6.15-drug_list`

**Base Release:** `v1.6.7-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.15/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 427 |
| `atc_level_2` | 428 |
| `atc_level_3` | 429 |
| `atc_level_4` | 432 |
| `atc_level_5` | 432 |
| `atc_main` | 432 |
| `is_fda_generic_drug` | 6 |
| `l1_label` | 427 |
| `l2_label` | 428 |
| `l3_label` | 429 |
| `l4_label` | 428 |
| `l5_label` | 392 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `A` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `L` |
| `EC:01216` | Ozanimod | `L` | `*None*` |
| `EC:01589` | Teniposide | `L` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `H` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `A08` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `L01` |
| `EC:01216` | Ozanimod | `L04` | `*None*` |
| `EC:01589` | Teniposide | `L01` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `H05` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `A08A` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `L01X` |
| `EC:01216` | Ozanimod | `L04A` | `*None*` |
| `EC:01589` | Teniposide | `L01C` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `H05B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `A08AA` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `L01XX` |
| `EC:01216` | Ozanimod | `L04AE` | `*None*` |
| `EC:01589` | Teniposide | `L01CB` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `H05BX` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `A08AA12` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `L01XX81` |
| `EC:01216` | Ozanimod | `L04AE02` | `*None*` |
| `EC:01589` | Teniposide | `L01CB02` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `H05BX01` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `A08AA12` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `L01XX81` |
| `EC:01216` | Ozanimod | `L04AE02` | `*None*` |
| `EC:01589` | Teniposide | `L01CB02` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `H05BX01` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01732` | Venetoclax | `False` | `True` |
| `EC:01412` | Rifapentine | `False` | `True` |
| `EC:00722` | Fostamatinib | `False` | `True` |
| `EC:00155` | Baloxavir | `False` | `True` |
| `EC:01507` | Solriamfetol | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:01216` | Ozanimod | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01589` | Teniposide | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `Systemic hormonal preparations, excl. sex hormones and insulins` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `Antiobesity preparations, excl. diet products` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `Antineoplastic agents` |
| `EC:01216` | Ozanimod | `Immunosuppressants` | `*None*` |
| `EC:01589` | Teniposide | `Antineoplastic agents` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `Calcium homeostasis` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `Antiobesity preparations, excl. diet products` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `Other antineoplastic agents in atc` |
| `EC:01216` | Ozanimod | `Immunosuppressants` | `*None*` |
| `EC:01589` | Teniposide | `Plant alkaloids and other natural products, antineoplastic drugs` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `Anti-parathyroid agents` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `Centrally acting antiobesity products` | `*None*` |
| `EC:01143` | Nirogacestat | `*None*` | `Other antineoplastic agents in atc` |
| `EC:01216` | Ozanimod | `Sphingosine-1-phosphate (s1p) receptor modulators` | `*None*` |
| `EC:01589` | Teniposide | `Podophyllotoxin derivatives, antineoplastic drugs` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `Other anti-parathyroid agents in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01482` | Setmelanotide | `Setmelanotide` | `*None*` |
| `EC:01216` | Ozanimod | `Ozanimod` | `*None*` |
| `EC:01589` | Teniposide | `Teniposide` | `*None*` |
| `EC:00345` | Cinacalcet | `*None*` | `Cinacalcet` |
| `EC:00205` | Bimatoprost | `*None*` | `Bimatoprost` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 508 | 458 |
| `atc_level_2` | 508 | 458 |
| `atc_level_3` | 508 | 458 |
| `atc_level_4` | 508 | 458 |
| `atc_level_5` | 508 | 458 |
| `atc_main` | 508 | 458 |
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
| `l1_label` | 508 | 458 |
| `l2_label` | 508 | 458 |
| `l3_label` | 508 | 458 |
| `l4_label` | 515 | 467 |
| `l5_label` | 555 | 509 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
