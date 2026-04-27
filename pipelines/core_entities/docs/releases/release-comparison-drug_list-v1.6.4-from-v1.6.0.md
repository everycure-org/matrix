# Release Comparison Report

**New Release:** `v1.6.4-drug_list`

**Base Release:** `v1.6.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.4/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.0/ec-drug-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 2

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01822` | Adenine |
| `EC:01823` | Trehalose |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 439 |
| `atc_level_2` | 443 |
| `atc_level_3` | 445 |
| `atc_level_4` | 446 |
| `atc_level_5` | 446 |
| `atc_main` | 446 |
| `drug_class` | 5 |
| `drug_target` | 2 |
| `is_fda_generic_drug` | 3 |
| `l1_label` | 439 |
| `l2_label` | 443 |
| `l3_label` | 445 |
| `l4_label` | 445 |
| `l5_label` | 414 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `C` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `B` |
| `EC:00165` | Belimumab | `*None*` | `L` |
| `EC:01335` | Prednisolone | `H` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `M` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `C02` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `B02` |
| `EC:00165` | Belimumab | `*None*` | `L04` |
| `EC:01335` | Prednisolone | `H02` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `M01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `C02C` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `B02B` |
| `EC:00165` | Belimumab | `*None*` | `L04A` |
| `EC:01335` | Prednisolone | `H02A` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `M01A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `C02CA` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `B02BX` |
| `EC:00165` | Belimumab | `*None*` | `L04AG` |
| `EC:01335` | Prednisolone | `H02AB` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `M01AB` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `C02CA01` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `B02BX06` |
| `EC:00165` | Belimumab | `*None*` | `L04AG04` |
| `EC:01335` | Prednisolone | `H02AB06` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `M01AB05` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `C02CA01` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `B02BX06` |
| `EC:00165` | Belimumab | `*None*` | `L04AG04` |
| `EC:01335` | Prednisolone | `H02AB06` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `M01AB05` |

#### `drug_class`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01495` | Sitagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:01742` | Vildagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:00927` | Linagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:00051` | Alogliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |
| `EC:01463` | Saxagliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00553` | Dyclonine | `Sodium channel inhibitor` | `TRPV3 inhibitor, SCN1A inhibitor` |
| `EC:00051` | Alogliptin | `DDP-4 inhibitor` | `DPP-4 inhibitor` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00150` | Azilsartan | `False` | `True` |
| `EC:00597` | Enzalutamide | `False` | `True` |
| `EC:00261` | Canagliflozin | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `Cardiovascular system drugs` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `Blood and blood forming organ drugs` |
| `EC:00165` | Belimumab | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:01335` | Prednisolone | `Systemic hormonal preparations, excl. sex hormones and insulins` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `Musculo-skeletal system drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `Antihypertensives` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `Antihemorrhagics` |
| `EC:00165` | Belimumab | `*None*` | `Immunosuppressants` |
| `EC:01335` | Prednisolone | `Corticosteroids for systemic use` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `Antiinflammatory and antirheumatic products` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `Antiadrenergic agents, peripherally acting` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `Vitamin k and other hemostatics` |
| `EC:00165` | Belimumab | `*None*` | `Immunosuppressants` |
| `EC:01335` | Prednisolone | `Corticosteroids for systemic use, plain` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `Antiinflammatory and antirheumatic products, non-steroids` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `Alpha-adrenoreceptor antagonists, peripherally acting` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `Other systemic hemostatics in atc` |
| `EC:00165` | Belimumab | `*None*` | `Monoclonal antibodies` |
| `EC:01335` | Prednisolone | `Glucocorticoids, systemic` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `Acetic acid derivatives and related substances` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01333` | Prazosin | `Prazosin` | `*None*` |
| `EC:00581` | Emicizumab | `*None*` | `Emicizumab` |
| `EC:00165` | Belimumab | `*None*` | `Belimumab` |
| `EC:01335` | Prednisolone | `Prednisolone` | `*None*` |
| `EC:00491` | Diclofenac | `*None*` | `Diclofenac` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 496 | 502 |
| `atc_level_2` | 496 | 502 |
| `atc_level_3` | 496 | 502 |
| `atc_level_4` | 496 | 502 |
| `atc_level_5` | 496 | 502 |
| `atc_main` | 496 | 502 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1812 | 1814 |
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
| `l1_label` | 496 | 502 |
| `l2_label` | 496 | 502 |
| `l3_label` | 496 | 502 |
| `l4_label` | 506 | 511 |
| `l5_label` | 549 | 559 |
| `name` | 0 | 0 |
| `new_id` | 1814 | 1816 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
