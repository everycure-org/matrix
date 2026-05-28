# Release Comparison Report

**New Release:** `v2.0.5-disease_list`

**Base Release:** `v2.0.2-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v2.0.5/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v2.0.2/ec-disease-list.parquet`

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
| `MONDO:1010182` | Premenstrual dysphoric disorder |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `level` | 1 |
| `supergroup` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `level`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007050` | Acromegaloid changes, cutis verticis gyrata, an... | `clinically_recognized` | `exclude` |

#### `supergroup`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007050` | Acromegaloid changes, cutis verticis gyrata, an... | `NNNI` | `exclude` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22112 | 22113 |
| `anatomical_name` | 21981 | 21982 |
| `benign_malignant` | 22048 | 22049 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 113 | 114 |
| `is_glucose_dysfunction` | 23 | 24 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 113 | 114 |
| `is_psychiatric_disease` | 113 | 114 |
| `level` | 1869 | 1869 |
| `mondo_top_grouping` | 24 | 24 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22737 | 22738 |
| `precancerous` | 22 | 22 |
| `prevalence_experimental` | 15657 | 15658 |
| `prevalence_world` | 15522 | 15523 |
| `speciality_breast` | 22 | 22 |
| `speciality_cardiovascular` | 22 | 22 |
| `speciality_chromosomal` | 22 | 22 |
| `speciality_connective_tissue` | 22 | 22 |
| `speciality_dermatologic` | 22 | 22 |
| `speciality_ear_nose_throat` | 22 | 22 |
| `speciality_endocrine` | 22 | 22 |
| `speciality_eye_and_adnexa` | 22 | 22 |
| `speciality_gastrointestinal` | 22 | 22 |
| `speciality_hematologic` | 22 | 22 |
| `speciality_immune` | 22 | 22 |
| `speciality_infection` | 22 | 22 |
| `speciality_metabolic` | 22 | 22 |
| `speciality_musculoskeletal` | 22 | 22 |
| `speciality_neoplasm` | 22 | 22 |
| `speciality_neurological` | 22 | 22 |
| `speciality_obstetric` | 22 | 22 |
| `speciality_poisoning_and_toxicity` | 22 | 22 |
| `speciality_psychiatric` | 22 | 22 |
| `speciality_renal_and_urinary` | 22 | 22 |
| `speciality_reproductive` | 22 | 22 |
| `speciality_respiratory` | 22 | 22 |
| `speciality_syndromic` | 22 | 22 |
| `strategically_viable` | 22 | 22 |
| `supergroup` | 22 | 22 |
| `synonyms` | 22 | 22 |
| `txgnn` | 23 | 24 |
| `unmet_medical_need` | 15522 | 15523 |
