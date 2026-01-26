# Release Comparison Report

**New Release:** `v1.2.16-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.16/03_primary/release/ec-drug-list.parquet`

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
| `atc_level_1` | 427 |
| `atc_level_2` | 431 |
| `atc_level_3` | 431 |
| `atc_level_4` | 433 |
| `atc_level_5` | 433 |
| `atc_main` | 433 |
| `l1_label` | 427 |
| `l2_label` | 431 |
| `l3_label` | 431 |
| `l4_label` | 432 |
| `l5_label` | 409 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `L` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `H` |
| `EC:01016` | Metformin | `A` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `N` |
| `EC:01482` | Setmelanotide | `A` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `L01` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `H01` |
| `EC:01016` | Metformin | `A10` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `N05` |
| `EC:01482` | Setmelanotide | `A08` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `L01X` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `H01C` |
| `EC:01016` | Metformin | `A10B` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `N05C` |
| `EC:01482` | Setmelanotide | `A08A` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `L01XX` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `H01CB` |
| `EC:01016` | Metformin | `A10BA` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `N05CA` |
| `EC:01482` | Setmelanotide | `A08AA` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `L01XX87` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `H01CB02` |
| `EC:01016` | Metformin | `A10BA02` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `N05CA01` |
| `EC:01482` | Setmelanotide | `A08AA12` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `L01XX87` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `H01CB02` |
| `EC:01016` | Metformin | `A10BA02` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `N05CA01` |
| `EC:01482` | Setmelanotide | `A08AA12` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `Systemic hormonal preparations, excl. sex hormones and insulins` |
| `EC:01016` | Metformin | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `Nervous system drugs` |
| `EC:01482` | Setmelanotide | `Alimentary tract and metabolism drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `Antineoplastic agents` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `Pituitary and hypothalamic hormones and analogues` |
| `EC:01016` | Metformin | `Drugs used in diabetes` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `Psycholeptics` |
| `EC:01482` | Setmelanotide | `Antiobesity preparations, excl. diet products` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `Other antineoplastic agents in atc` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `Hypothalamic hormones` |
| `EC:01016` | Metformin | `Blood glucose lowering drugs, excl. insulins` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `Hypnotics and sedatives` |
| `EC:01482` | Setmelanotide | `Antiobesity preparations, excl. diet products` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01405` | Revumenib | `Other antineoplastic agents in atc` | `*None*` |
| `EC:01169` | Octreotide | `*None*` | `Somatostatin and analogues` |
| `EC:01016` | Metformin | `Biguanide blood glucose lower drugs` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `Barbiturates, hypnotics and sedatives, plain` |
| `EC:01482` | Setmelanotide | `Centrally acting antiobesity products` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01169` | Octreotide | `*None*` | `Octreotide` |
| `EC:01016` | Metformin | `Metformin` | `*None*` |
| `EC:01259` | Pentobarbital | `*None*` | `Pentobarbital` |
| `EC:01482` | Setmelanotide | `Setmelanotide` | `*None*` |
| `EC:00625` | Esmolol | `*None*` | `Esmolol` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 486 |
| `atc_level_2` | 512 | 486 |
| `atc_level_3` | 512 | 486 |
| `atc_level_4` | 512 | 486 |
| `atc_level_5` | 512 | 486 |
| `atc_main` | 512 | 486 |
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
| `l1_label` | 512 | 486 |
| `l2_label` | 512 | 486 |
| `l3_label` | 512 | 486 |
| `l4_label` | 521 | 494 |
| `l5_label` | 565 | 536 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1806 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
