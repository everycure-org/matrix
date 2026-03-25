# Release Comparison Report

**New Release:** `v1.3.0-drug_list`

**Base Release:** `v1.2.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.3.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.14/ec-drug-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
- `smiles`

## Row Changes

### Added Rows
**Total:** 6

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01815` | Donidalorsen |
| `EC:01818` | Remibrutinib |
| `EC:01813` | Radotinib |
| `EC:01816` | Imlunestrant |
| `EC:01817` | Paltusotine |
| `EC:01814` | Pantothenic acid |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 283 |
| `atc_level_2` | 288 |
| `atc_level_3` | 289 |
| `atc_level_4` | 293 |
| `atc_level_5` | 293 |
| `atc_main` | 293 |
| `deleted` | 1 |
| `deleted_reason` | 1 |
| `l1_label` | 283 |
| `l2_label` | 288 |
| `l3_label` | 289 |
| `l4_label` | 292 |
| `l5_label` | 268 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `H` |
| `EC:01163` | Nusinersen | `*None*` | `M` |
| `EC:00717` | Fosamprenavir | `*None*` | `J` |
| `EC:01730` | Velmanase alfa | `*None*` | `A` |
| `EC:00879` | Landiolol | `*None*` | `C` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `H02` |
| `EC:01163` | Nusinersen | `*None*` | `M09` |
| `EC:00717` | Fosamprenavir | `*None*` | `J05` |
| `EC:01730` | Velmanase alfa | `*None*` | `A16` |
| `EC:00879` | Landiolol | `*None*` | `C07` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `H02C` |
| `EC:01163` | Nusinersen | `*None*` | `M09A` |
| `EC:00717` | Fosamprenavir | `*None*` | `J05A` |
| `EC:01730` | Velmanase alfa | `*None*` | `A16A` |
| `EC:00879` | Landiolol | `*None*` | `C07A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `H02CA` |
| `EC:01163` | Nusinersen | `*None*` | `M09AX` |
| `EC:00717` | Fosamprenavir | `*None*` | `J05AE` |
| `EC:01730` | Velmanase alfa | `*None*` | `A16AB` |
| `EC:00879` | Landiolol | `*None*` | `C07AB` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `H02CA04` |
| `EC:01163` | Nusinersen | `*None*` | `M09AX07` |
| `EC:00717` | Fosamprenavir | `*None*` | `J05AE07` |
| `EC:01730` | Velmanase alfa | `*None*` | `A16AB15` |
| `EC:00879` | Landiolol | `*None*` | `C07AB14` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `H02CA04` |
| `EC:01163` | Nusinersen | `*None*` | `M09AX07` |
| `EC:00717` | Fosamprenavir | `*None*` | `J05AE07` |
| `EC:01730` | Velmanase alfa | `*None*` | `A16AB15` |
| `EC:00879` | Landiolol | `*None*` | `C07AB14` |

#### `deleted`
*(Full comparison of all changed values)*

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00907` | Levamisole | `False` | `True` |

#### `deleted_reason`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00907` | Levamisole | `*None*` | `withdrawn for human use plus complicated legal status given its use in the production of illicit ...` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `Systemic hormonal preparations, excl. sex hormones and insulins` |
| `EC:01163` | Nusinersen | `*None*` | `Musculo-skeletal system drugs` |
| `EC:00717` | Fosamprenavir | `*None*` | `Antiinfectives for systemic use` |
| `EC:01730` | Velmanase alfa | `*None*` | `Alimentary tract and metabolism drugs` |
| `EC:00879` | Landiolol | `*None*` | `Cardiovascular system drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `Corticosteroids for systemic use` |
| `EC:01163` | Nusinersen | `*None*` | `Other drugs for disorders of the musculo-skeletal system in atc` |
| `EC:00717` | Fosamprenavir | `*None*` | `Antivirals for systemic use` |
| `EC:01730` | Velmanase alfa | `*None*` | `Other alimentary tract and metabolism products in atc` |
| `EC:00879` | Landiolol | `*None*` | `Beta-adrenergic blocking agents` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `Antiadrenal preparations` |
| `EC:01163` | Nusinersen | `*None*` | `Other drugs for disorders of the musculo-skeletal system in atc` |
| `EC:00717` | Fosamprenavir | `*None*` | `Direct acting antivirals` |
| `EC:01730` | Velmanase alfa | `*None*` | `Other alimentary tract and metabolism products in atc` |
| `EC:00879` | Landiolol | `*None*` | `Beta blocking agents` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `Anticorticosteroids` |
| `EC:01163` | Nusinersen | `*None*` | `Other drugs for disorders of the musculo-skeletal system in atc` |
| `EC:00717` | Fosamprenavir | `*None*` | `Protease inhibitors, direct acting antivirals` |
| `EC:01730` | Velmanase alfa | `*None*` | `Enzymes for alimentary tract and metabolism` |
| `EC:00879` | Landiolol | `*None*` | `Beta blocking agents, selective` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00916` | Levoketoconazole | `*None*` | `Levoketoconazole` |
| `EC:01163` | Nusinersen | `*None*` | `Nusinersen` |
| `EC:00717` | Fosamprenavir | `*None*` | `Fosamprenavir` |
| `EC:01730` | Velmanase alfa | `*None*` | `Velmanase alfa` |
| `EC:00879` | Landiolol | `*None*` | `Landiolol` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 512 | 243 |
| `atc_level_2` | 512 | 243 |
| `atc_level_3` | 512 | 243 |
| `atc_level_4` | 512 | 243 |
| `atc_level_5` | 512 | 243 |
| `atc_main` | 512 | 243 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1805 | 1810 |
| `drug_class` | 1 | 1 |
| `drug_function` | 17 | 18 |
| `drug_target` | 23 | 24 |
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
| `l1_label` | 512 | 243 |
| `l2_label` | 512 | 243 |
| `l3_label` | 512 | 243 |
| `l4_label` | 521 | 253 |
| `l5_label` | 565 | 303 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1812 |
| `smiles` | 331 | N/A |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
