# Release Comparison Report

**New Release:** `v1.6.20-drug_list`

**Base Release:** `v1.6.18-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.20/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.18/ec-drug-list.parquet`

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
| `atc_level_1` | 443 |
| `atc_level_2` | 446 |
| `atc_level_3` | 448 |
| `atc_level_4` | 450 |
| `atc_level_5` | 450 |
| `atc_main` | 450 |
| `drug_target` | 1 |
| `is_fda_generic_drug` | 1 |
| `l1_label` | 443 |
| `l2_label` | 446 |
| `l3_label` | 448 |
| `l4_label` | 447 |
| `l5_label` | 412 |
| `synonyms` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `N` |
| `EC:00205` | Bimatoprost | `S` | `*None*` |
| `EC:01125` | Neratinib | `L` | `*None*` |
| `EC:01331` | Pravastatin | `C` | `*None*` |
| `EC:01708` | Upadacitinib | `L` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `N02` |
| `EC:00205` | Bimatoprost | `S01` | `*None*` |
| `EC:01125` | Neratinib | `L01` | `*None*` |
| `EC:01331` | Pravastatin | `C10` | `*None*` |
| `EC:01708` | Upadacitinib | `L04` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `N02A` |
| `EC:00205` | Bimatoprost | `S01E` | `*None*` |
| `EC:01125` | Neratinib | `L01E` | `*None*` |
| `EC:01331` | Pravastatin | `C10A` | `*None*` |
| `EC:01708` | Upadacitinib | `L04A` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `N02AD` |
| `EC:00205` | Bimatoprost | `S01EE` | `*None*` |
| `EC:00633` | Estrone | `G03CA` | `G03CC` |
| `EC:01125` | Neratinib | `L01EH` | `*None*` |
| `EC:01331` | Pravastatin | `C10AA` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `N02AD01` |
| `EC:00205` | Bimatoprost | `S01EE03` | `*None*` |
| `EC:00633` | Estrone | `G03CA07` | `G03CC04` |
| `EC:01125` | Neratinib | `L01EH02` | `*None*` |
| `EC:01331` | Pravastatin | `C10AA03` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `N02AD01` |
| `EC:00205` | Bimatoprost | `S01EE03` | `*None*` |
| `EC:00633` | Estrone | `G03CA07` | `G03CC04` |
| `EC:01125` | Neratinib | `L01EH02` | `*None*` |
| `EC:01331` | Pravastatin | `C10AA03` | `*None*` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01567` | Taurolidine | `Taruine derivative` | `Taurine derivative` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00508` | Dimethyl sulfoxide | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `Nervous system drugs` |
| `EC:00205` | Bimatoprost | `Sensory organ drugs` | `*None*` |
| `EC:01125` | Neratinib | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01331` | Pravastatin | `Cardiovascular system drugs` | `*None*` |
| `EC:01708` | Upadacitinib | `Antineoplastic and immunomodulating agents` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `Analgesics` |
| `EC:00205` | Bimatoprost | `Ophthalmologicals` | `*None*` |
| `EC:01125` | Neratinib | `Antineoplastic agents` | `*None*` |
| `EC:01331` | Pravastatin | `Lipid modifying agents` | `*None*` |
| `EC:01708` | Upadacitinib | `Immunosuppressants` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `Opioid analgesics` |
| `EC:00205` | Bimatoprost | `Antiglaucoma preparations and miotics` | `*None*` |
| `EC:01125` | Neratinib | `Protein kinase inhibitors, antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01331` | Pravastatin | `Lipid modifying agents, plain` | `*None*` |
| `EC:01708` | Upadacitinib | `Immunosuppressants` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `Benzomorphan derivative analgesics` |
| `EC:00205` | Bimatoprost | `Prostaglandin analogues, antiglaucoma drugs and miotics` | `*None*` |
| `EC:00633` | Estrone | `Natural and semisynthetic estrogens, plain` | `Estrogens, combinations with other drugs` |
| `EC:01125` | Neratinib | `Human epidermal growth factor receptor 2 (her2) tyrosine kinase inhibitors` | `*None*` |
| `EC:01331` | Pravastatin | `Hmg coa reductase inhibitors, plain lipid modifying drugs` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01258` | Pentazocine | `*None*` | `Pentazocine` |
| `EC:00205` | Bimatoprost | `Bimatoprost` | `*None*` |
| `EC:01125` | Neratinib | `Neratinib` | `*None*` |
| `EC:01331` | Pravastatin | `Pravastatin` | `*None*` |
| `EC:01708` | Upadacitinib | `Upadacitinib` | `*None*` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01711` | Ursodeoxycholic acid | `['Ursodiol']` | `['Ursodiol' 'Udca']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 487 | 506 |
| `atc_level_2` | 487 | 506 |
| `atc_level_3` | 487 | 506 |
| `atc_level_4` | 487 | 506 |
| `atc_level_5` | 487 | 506 |
| `atc_main` | 487 | 506 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1816 | 1816 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 18 | 18 |
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
| `l1_label` | 487 | 506 |
| `l2_label` | 487 | 506 |
| `l3_label` | 487 | 506 |
| `l4_label` | 495 | 515 |
| `l5_label` | 535 | 551 |
| `name` | 0 | 0 |
| `new_id` | 1818 | 1818 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
