# Release Comparison Report

**New Release:** `v1.3.3-drug_list`

**Base Release:** `v1.3.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.3.3/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 258 |
| `atc_level_2` | 263 |
| `atc_level_3` | 265 |
| `atc_level_4` | 266 |
| `atc_level_5` | 266 |
| `atc_main` | 266 |
| `l1_label` | 258 |
| `l2_label` | 263 |
| `l3_label` | 265 |
| `l4_label` | 265 |
| `l5_label` | 239 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `L` | `*None*` |
| `EC:00167` | Belumosudil | `L` | `*None*` |
| `EC:01069` | Mitotane | `L` | `*None*` |
| `EC:00719` | Fosdenopterin | `A` | `*None*` |
| `EC:00632` | Estriol | `G` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `L02` | `*None*` |
| `EC:00167` | Belumosudil | `L04` | `*None*` |
| `EC:01069` | Mitotane | `L01` | `*None*` |
| `EC:00719` | Fosdenopterin | `A16` | `*None*` |
| `EC:00632` | Estriol | `G03` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `L02B` | `*None*` |
| `EC:00167` | Belumosudil | `L04A` | `*None*` |
| `EC:01069` | Mitotane | `L01X` | `*None*` |
| `EC:00719` | Fosdenopterin | `A16A` | `*None*` |
| `EC:00632` | Estriol | `G03C` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `L02BG` | `*None*` |
| `EC:00167` | Belumosudil | `L04AA` | `*None*` |
| `EC:01069` | Mitotane | `L01XX` | `*None*` |
| `EC:00719` | Fosdenopterin | `A16AX` | `*None*` |
| `EC:00632` | Estriol | `G03CA` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `L02BG03` | `*None*` |
| `EC:00167` | Belumosudil | `L04AA48` | `*None*` |
| `EC:01069` | Mitotane | `L01XX23` | `*None*` |
| `EC:00719` | Fosdenopterin | `A16AX19` | `*None*` |
| `EC:00632` | Estriol | `G03CA04` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `L02BG03` | `*None*` |
| `EC:00167` | Belumosudil | `L04AA48` | `*None*` |
| `EC:01069` | Mitotane | `L01XX23` | `*None*` |
| `EC:00719` | Fosdenopterin | `A16AX19` | `*None*` |
| `EC:00632` | Estriol | `G03CA04` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00167` | Belumosudil | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01069` | Mitotane | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00719` | Fosdenopterin | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:00632` | Estriol | `Genito urinary system and sex hormones` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `Endocrine therapy antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00167` | Belumosudil | `Immunosuppressants` | `*None*` |
| `EC:01069` | Mitotane | `Antineoplastic agents` | `*None*` |
| `EC:00719` | Fosdenopterin | `Other alimentary tract and metabolism products in atc` | `*None*` |
| `EC:00632` | Estriol | `Sex hormones and modulators of the genital system` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `Hormone antagonists and related agents` | `*None*` |
| `EC:00167` | Belumosudil | `Immunosuppressants` | `*None*` |
| `EC:01069` | Mitotane | `Other antineoplastic agents in atc` | `*None*` |
| `EC:00719` | Fosdenopterin | `Other alimentary tract and metabolism products in atc` | `*None*` |
| `EC:00632` | Estriol | `Estrogens, sex hormones and modulators of the genital system` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `Aromatase inhibitors` | `*None*` |
| `EC:00167` | Belumosudil | `Selective immunosuppressants` | `*None*` |
| `EC:01069` | Mitotane | `Other antineoplastic agents in atc` | `*None*` |
| `EC:00719` | Fosdenopterin | `Various alimentary tract and metabolism products` | `*None*` |
| `EC:00632` | Estriol | `Natural and semisynthetic estrogens, plain` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00088` | Anastrozole | `Anastrozole` | `*None*` |
| `EC:00167` | Belumosudil | `Belumosudil` | `*None*` |
| `EC:01069` | Mitotane | `Mitotane` | `*None*` |
| `EC:00719` | Fosdenopterin | `Fosdenopterin` | `*None*` |
| `EC:00632` | Estriol | `Estriol` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 243 | 489 |
| `atc_level_2` | 243 | 489 |
| `atc_level_3` | 243 | 489 |
| `atc_level_4` | 243 | 489 |
| `atc_level_5` | 243 | 489 |
| `atc_main` | 243 | 489 |
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
| `l1_label` | 243 | 489 |
| `l2_label` | 243 | 489 |
| `l3_label` | 243 | 489 |
| `l4_label` | 253 | 498 |
| `l5_label` | 303 | 542 |
| `name` | 0 | 0 |
| `new_id` | 1812 | 1813 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
