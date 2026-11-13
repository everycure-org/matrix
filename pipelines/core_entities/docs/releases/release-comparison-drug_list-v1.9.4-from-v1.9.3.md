# Release Comparison Report

**New Release:** `v1.9.4-drug_list`

**Base Release:** `v1.9.3-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.9.4/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.9.3/ec-drug-list.parquet`

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
| `atc_level_1` | 420 |
| `atc_level_2` | 422 |
| `atc_level_3` | 423 |
| `atc_level_4` | 426 |
| `atc_level_5` | 426 |
| `atc_main` | 426 |
| `is_fda_generic_drug` | 4 |
| `l1_label` | 420 |
| `l2_label` | 422 |
| `l3_label` | 423 |
| `l4_label` | 423 |
| `l5_label` | 381 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `L` |
| `EC:01303` | Plazomicin | `*None*` | `J` |
| `EC:00671` | Ferric citrate | `*None*` | `V` |
| `EC:01759` | Voriconazole | `*None*` | `J` |
| `EC:01742` | Vildagliptin | `A` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `L02` |
| `EC:01303` | Plazomicin | `*None*` | `J01` |
| `EC:00671` | Ferric citrate | `*None*` | `V03` |
| `EC:01759` | Voriconazole | `*None*` | `J02` |
| `EC:01742` | Vildagliptin | `A10` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `L02B` |
| `EC:01303` | Plazomicin | `*None*` | `J01G` |
| `EC:00671` | Ferric citrate | `*None*` | `V03A` |
| `EC:01759` | Voriconazole | `*None*` | `J02A` |
| `EC:01742` | Vildagliptin | `A10B` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `L02BB` |
| `EC:01303` | Plazomicin | `*None*` | `J01GB` |
| `EC:00671` | Ferric citrate | `*None*` | `V03AE` |
| `EC:01759` | Voriconazole | `*None*` | `J02AC` |
| `EC:01742` | Vildagliptin | `A10BH` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `L02BB06` |
| `EC:01303` | Plazomicin | `*None*` | `J01GB14` |
| `EC:00671` | Ferric citrate | `*None*` | `V03AE08` |
| `EC:01759` | Voriconazole | `*None*` | `J02AC03` |
| `EC:01742` | Vildagliptin | `A10BH02` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `L02BB06` |
| `EC:01303` | Plazomicin | `*None*` | `J01GB14` |
| `EC:00671` | Ferric citrate | `*None*` | `V03AE08` |
| `EC:01759` | Voriconazole | `*None*` | `J02AC03` |
| `EC:01742` | Vildagliptin | `A10BH02` | `*None*` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00656` | Exenatide | `True` | `False` |
| `EC:01291` | Piperacillin | `True` | `False` |
| `EC:01320` | Potassium phosphate | `True` | `False` |
| `EC:01708` | Upadacitinib | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:01303` | Plazomicin | `*None*` | `Antiinfectives for systemic use` |
| `EC:00671` | Ferric citrate | `*None*` | `Various drug classes in atc` |
| `EC:01759` | Voriconazole | `*None*` | `Antiinfectives for systemic use` |
| `EC:01742` | Vildagliptin | `Alimentary tract and metabolism drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `Endocrine therapy antineoplastic and immunomodulating agents` |
| `EC:01303` | Plazomicin | `*None*` | `Antibacterials for systemic use` |
| `EC:00671` | Ferric citrate | `*None*` | `All other therapeutic products` |
| `EC:01759` | Voriconazole | `*None*` | `Antimycotics for systemic use` |
| `EC:01742` | Vildagliptin | `Drugs used in diabetes` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `Hormone antagonists and related agents` |
| `EC:01303` | Plazomicin | `*None*` | `Aminoglycoside antibacterials` |
| `EC:00671` | Ferric citrate | `*None*` | `All other therapeutic products` |
| `EC:01759` | Voriconazole | `*None*` | `Antimycotics for systemic use` |
| `EC:01742` | Vildagliptin | `Blood glucose lowering drugs, excl. insulins` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `Anti-androgen hormone antagonists and related agents` |
| `EC:01303` | Plazomicin | `*None*` | `Other aminoglycosides in atc` |
| `EC:00671` | Ferric citrate | `*None*` | `Drugs for treatment of hyperkalemia and hyperphosphatemia` |
| `EC:01759` | Voriconazole | `*None*` | `Triazole and tetrazole derivatives` |
| `EC:01742` | Vildagliptin | `Dipeptidyl peptidase 4 (dpp-4) inhibitors for blood glucose lowering` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00442` | Darolutamide | `*None*` | `Darolutamide` |
| `EC:01303` | Plazomicin | `*None*` | `Plazomicin` |
| `EC:00671` | Ferric citrate | `*None*` | `Ferric citrate` |
| `EC:01759` | Voriconazole | `*None*` | `Voriconazole` |
| `EC:01742` | Vildagliptin | `Vildagliptin` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 578 | 597 |
| `atc_level_2` | 578 | 597 |
| `atc_level_3` | 578 | 597 |
| `atc_level_4` | 578 | 597 |
| `atc_level_5` | 578 | 597 |
| `atc_main` | 578 | 597 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1927 | 1927 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 56 | 56 |
| `gras_usa` | 0 | 0 |
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
| `l1_label` | 578 | 597 |
| `l2_label` | 578 | 597 |
| `l3_label` | 578 | 597 |
| `l4_label` | 586 | 606 |
| `l5_label` | 626 | 643 |
| `name` | 0 | 0 |
| `new_id` | 1929 | 1929 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
