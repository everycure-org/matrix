# Release Comparison Report

**New Release:** `v1.6.25-drug_list`

**Base Release:** `v1.6.18-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.25/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.18/ec-drug-list.parquet`

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
| `EC:01856` | Alpha-ketoglutarate |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 454 |
| `atc_level_2` | 460 |
| `atc_level_3` | 463 |
| `atc_level_4` | 466 |
| `atc_level_5` | 466 |
| `atc_main` | 466 |
| `drug_target` | 1 |
| `is_fda_generic_drug` | 6 |
| `l1_label` | 454 |
| `l2_label` | 460 |
| `l3_label` | 463 |
| `l4_label` | 464 |
| `l5_label` | 427 |
| `synonyms` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `R` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `N` |
| `EC:00962` | Lumasiran | `*None*` | `A` |
| `EC:00805` | Icatibant | `B` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `A` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `R05` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `N05` |
| `EC:00962` | Lumasiran | `*None*` | `A16` |
| `EC:00805` | Icatibant | `B06` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `A10` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `R05D` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `N05A` |
| `EC:00962` | Lumasiran | `*None*` | `A16A` |
| `EC:00805` | Icatibant | `B06A` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `A10B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `R05DB` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `N05AL` |
| `EC:00962` | Lumasiran | `*None*` | `A16AX` |
| `EC:00805` | Icatibant | `B06AC` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `A10BJ` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `R05DB10` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `N05AL05` |
| `EC:00962` | Lumasiran | `*None*` | `A16AX18` |
| `EC:00805` | Icatibant | `B06AC02` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `A10BJ03` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `R05DB10` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `N05AL05` |
| `EC:00962` | Lumasiran | `*None*` | `A16AX18` |
| `EC:00805` | Icatibant | `B06AC02` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `A10BJ03` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01567` | Taurolidine | `Taruine derivative` | `Taurine derivative` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01704` | Ubrogepant | `False` | `True` |
| `EC:00765` | Golimumab | `False` | `True` |
| `EC:00031` | Afatinib | `False` | `True` |
| `EC:00814` | Iloprost | `False` | `True` |
| `EC:00508` | Dimethyl sulfoxide | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `Respiratory system drugs` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `Nervous system drugs` |
| `EC:00962` | Lumasiran | `*None*` | `Alimentary tract and metabolism drugs` |
| `EC:00805` | Icatibant | `Blood and blood forming organ drugs` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `Alimentary tract and metabolism drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `Cough and cold preparations` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `Psycholeptics` |
| `EC:00962` | Lumasiran | `*None*` | `Other alimentary tract and metabolism products in atc` |
| `EC:00805` | Icatibant | `Other hematological agents in atc` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `Drugs used in diabetes` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `Cough suppressants, excl. combinations with expectorants` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `Antipsychotics` |
| `EC:00962` | Lumasiran | `*None*` | `Other alimentary tract and metabolism products in atc` |
| `EC:00805` | Icatibant | `Other hematological agents in atc` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `Blood glucose lowering drugs, excl. insulins` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `Other cough suppressants in atc` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `Benzamide antipsychotics` |
| `EC:00962` | Lumasiran | `*None*` | `Various alimentary tract and metabolism products` |
| `EC:00805` | Icatibant | `Drugs used in hereditary angioedema` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `Glucagon-like peptide-1 (glp-1) analogues` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00323` | Clofedanol | `Clofedanol` | `*None*` |
| `EC:00073` | Amisulpride | `*None*` | `Amisulpride` |
| `EC:00962` | Lumasiran | `*None*` | `Lumasiran` |
| `EC:00805` | Icatibant | `Icatibant` | `*None*` |
| `EC:00939` | Lixisenatide | `*None*` | `Lixisenatide` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01711` | Ursodeoxycholic acid | `['Ursodiol']` | `['Ursodiol' 'Udca']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 487 | 501 |
| `atc_level_2` | 487 | 501 |
| `atc_level_3` | 487 | 501 |
| `atc_level_4` | 487 | 501 |
| `atc_level_5` | 487 | 501 |
| `atc_main` | 487 | 501 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1816 | 1817 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 18 | 19 |
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
| `l1_label` | 487 | 501 |
| `l2_label` | 487 | 501 |
| `l3_label` | 487 | 501 |
| `l4_label` | 495 | 511 |
| `l5_label` | 535 | 553 |
| `name` | 0 | 0 |
| `new_id` | 1818 | 1819 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
