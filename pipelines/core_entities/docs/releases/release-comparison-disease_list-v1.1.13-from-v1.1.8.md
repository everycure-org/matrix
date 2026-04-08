# Release Comparison Report

**New Release:** `v1.1.13-disease_list`

**Base Release:** `v1.1.8-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.1.13/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.1.8/ec-disease-list.parquet`

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
| `level` | 2 |
| `name` | 4 |
| `supergroup` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `level`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0037937` | Pyrimidine metabolism disease | `clinically_recognized` | `grouping` |
| `MONDO:0000661` | Alexithymia | `clinically_recognized` | `exclude` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0014205` | `Severe feeding difficulties-failure to thrive-microcephaly due to asxl3 deficiency syndrome` | `Bainbridge-Roppers syndrome` |
| `MONDO:0013999` | `Retinal dystrophy, optic nerve edema, splenomegaly, anhidrosis, and migraine headache syndrome` | `ROSAH syndrome` |
| `MONDO:0030018` | `Autoinflammation with episodic fever and lymphadenopathy` | `Cria Syndrome` |
| `MONDO:0800195` | `Achalasia-alacrima syndrome` | `Allgrove syndrome` |

#### `supergroup`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0000661` | Alexithymia | `NNNI` | `exclude` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 21 | 21 |
| `anatomical_id` | 22127 | 22127 |
| `anatomical_name` | 21996 | 21996 |
| `benign_malignant` | 22063 | 22063 |
| `core` | 21 | 21 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 112 | 112 |
| `is_glucose_dysfunction` | 22 | 22 |
| `is_infectious_disease` | 21 | 21 |
| `is_malignant_cancer` | 112 | 112 |
| `is_psychiatric_disease` | 112 | 112 |
| `level` | 1869 | 1869 |
| `mondo_top_grouping` | 66 | 66 |
| `mondo_txgnn` | 21 | 21 |
| `name` | 0 | 0 |
| `new_id` | 22753 | 22753 |
| `precancerous` | 21 | 21 |
| `prevalence_experimental` | 15670 | 15670 |
| `prevalence_world` | 15535 | 15535 |
| `speciality_breast` | 21 | 21 |
| `speciality_cardiovascular` | 21 | 21 |
| `speciality_chromosomal` | 21 | 21 |
| `speciality_connective_tissue` | 21 | 21 |
| `speciality_dermatologic` | 21 | 21 |
| `speciality_ear_nose_throat` | 21 | 21 |
| `speciality_endocrine` | 21 | 21 |
| `speciality_eye_and_adnexa` | 21 | 21 |
| `speciality_gastrointestinal` | 21 | 21 |
| `speciality_hematologic` | 21 | 21 |
| `speciality_immune` | 21 | 21 |
| `speciality_infection` | 21 | 21 |
| `speciality_metabolic` | 21 | 21 |
| `speciality_musculoskeletal` | 21 | 21 |
| `speciality_neoplasm` | 21 | 21 |
| `speciality_neurological` | 21 | 21 |
| `speciality_obstetric` | 21 | 21 |
| `speciality_poisoning_and_toxicity` | 21 | 21 |
| `speciality_psychiatric` | 21 | 21 |
| `speciality_renal_and_urinary` | 21 | 21 |
| `speciality_reproductive` | 21 | 21 |
| `speciality_respiratory` | 21 | 21 |
| `speciality_syndromic` | 21 | 21 |
| `supergroup` | 21 | 21 |
| `synonyms` | 4877 | 4877 |
| `txgnn` | 22 | 22 |
| `unmet_medical_need` | 15535 | 15535 |
