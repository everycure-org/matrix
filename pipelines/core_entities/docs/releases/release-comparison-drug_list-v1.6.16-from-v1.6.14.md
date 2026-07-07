# Release Comparison Report

**New Release:** `v1.6.16-drug_list`

**Base Release:** `v1.6.14-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.16/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.14/ec-drug-list.parquet`

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
| `atc_level_1` | 439 |
| `atc_level_2` | 444 |
| `atc_level_3` | 446 |
| `atc_level_4` | 447 |
| `atc_level_5` | 447 |
| `atc_main` | 447 |
| `l1_label` | 439 |
| `l2_label` | 444 |
| `l3_label` | 446 |
| `l4_label` | 445 |
| `l5_label` | 411 |
| `synonyms` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `L` |
| `EC:00198` | Bevacizumab | `L` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `R` |
| `EC:00184` | Benzydamine | `A` | `R` |
| `EC:01469` | Seladelpar | `*None*` | `A` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `L01` |
| `EC:00198` | Bevacizumab | `L01` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `R03` |
| `EC:00184` | Benzydamine | `A01` | `R02` |
| `EC:01469` | Seladelpar | `*None*` | `A05` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `L01F` |
| `EC:00198` | Bevacizumab | `L01F` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `R03D` |
| `EC:00184` | Benzydamine | `A01A` | `R02A` |
| `EC:01469` | Seladelpar | `*None*` | `A05A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `L01FE` |
| `EC:00198` | Bevacizumab | `L01FG` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `R03DC` |
| `EC:00184` | Benzydamine | `A01AD` | `R02AX` |
| `EC:01469` | Seladelpar | `*None*` | `A05AX` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `L01FE03` |
| `EC:00198` | Bevacizumab | `L01FG01` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `R03DC03` |
| `EC:00184` | Benzydamine | `A01AD02` | `R02AX03` |
| `EC:01469` | Seladelpar | `*None*` | `A05AX07` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `L01FE03` |
| `EC:00198` | Bevacizumab | `L01FG01` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `R03DC03` |
| `EC:00184` | Benzydamine | `A01AD02` | `R02AX03` |
| `EC:01469` | Seladelpar | `*None*` | `A05AX07` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:00198` | Bevacizumab | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `Respiratory system drugs` |
| `EC:00184` | Benzydamine | `Alimentary tract and metabolism drugs` | `Respiratory system drugs` |
| `EC:01469` | Seladelpar | `*None*` | `Alimentary tract and metabolism drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `Antineoplastic agents` |
| `EC:00198` | Bevacizumab | `Antineoplastic agents` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `Drugs for obstructive airway diseases` |
| `EC:00184` | Benzydamine | `Stomatological preparations` | `Throat preparations` |
| `EC:01469` | Seladelpar | `*None*` | `Bile and liver therapy drugs` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `Monoclonal antibodies and antibody drug conjugates` |
| `EC:00198` | Bevacizumab | `Monoclonal antibodies and antibody drug conjugates` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `Other systemic drugs for obstructive airway diseases in atc` |
| `EC:00184` | Benzydamine | `Stomatological preparations` | `Throat preparations` |
| `EC:01469` | Seladelpar | `*None*` | `Bile therapy drugs` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `Egfr (epidermal growth factor receptor) inhibitors` |
| `EC:00198` | Bevacizumab | `Vegf/vegfr (vascular endothelial growth factor) inhibitors` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `Leukotriene receptor antagonists for obstructive airway diseases` |
| `EC:00184` | Benzydamine | `Other agents for local oral treatment in atc` | `Other throat preparations in atc` |
| `EC:01469` | Seladelpar | `*None*` | `Other drugs for bile therapy in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01114` | Necitumumab | `*None*` | `Necitumumab` |
| `EC:00198` | Bevacizumab | `Bevacizumab` | `*None*` |
| `EC:01082` | Montelukast | `*None*` | `Montelukast` |
| `EC:01446` | Rufinamide | `*None*` | `Rufinamide` |
| `EC:01715` | Valganciclovir | `*None*` | `Valganciclovir` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00561` | Efgartigimod alfa | `[]` | `['Efgartigimod']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 499 | 478 |
| `atc_level_2` | 499 | 478 |
| `atc_level_3` | 499 | 478 |
| `atc_level_4` | 499 | 478 |
| `atc_level_5` | 499 | 478 |
| `atc_main` | 499 | 478 |
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
| `l1_label` | 499 | 478 |
| `l2_label` | 499 | 478 |
| `l3_label` | 499 | 478 |
| `l4_label` | 508 | 485 |
| `l5_label` | 546 | 529 |
| `name` | 0 | 0 |
| `new_id` | 1816 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
