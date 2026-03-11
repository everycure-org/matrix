# Release Comparison Report

**New Release:** `v1.1.8-disease_list`

**Base Release:** `v1.1.6-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.1.8/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.1.6/ec-disease-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 1

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0800029` | obsolete interstitial lung disease 2 |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `harrisons_view` | 41 |
| `mondo_top_grouping` | 41 |
| `name` | 7 |
| `synonyms` | 18 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0017724` | Tay-sachs disease, b variant, infantile form | `psychiatric_disorder\|hereditary_disease\|metabolic_disease\|nervous_system_disorder\|disorder_of_vis...` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder\|disorder_of_visual_system` |
| `MONDO:0013866` | Neuronal ceroid lipofuscinosis 11 | `psychiatric_disorder\|hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` |
| `MONDO:0008767` | Neuronal ceroid lipofuscinosis 3 | `psychiatric_disorder\|hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` |
| `MONDO:0012188` | Neuronal ceroid lipofuscinosis 9 | `psychiatric_disorder\|hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` |
| `MONDO:0012414` | Neuronal ceroid lipofuscinosis 10 | `psychiatric_disorder\|hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0017724` | Tay-sachs disease, b variant, infantile form | `disorder_of_visual_system\|nervous_system_disorder\|metabolic_disease\|hereditary_disease\|psychiatri...` | `disorder_of_visual_system\|nervous_system_disorder\|metabolic_disease\|hereditary_disease\|disorder_o...` |
| `MONDO:0013866` | Neuronal ceroid lipofuscinosis 11 | `nervous_system_disorder\|metabolic_disease\|hereditary_disease\|psychiatric_disorder` | `nervous_system_disorder\|metabolic_disease\|hereditary_disease` |
| `MONDO:0008767` | Neuronal ceroid lipofuscinosis 3 | `nervous_system_disorder\|metabolic_disease\|hereditary_disease\|psychiatric_disorder` | `nervous_system_disorder\|metabolic_disease\|hereditary_disease` |
| `MONDO:0012188` | Neuronal ceroid lipofuscinosis 9 | `nervous_system_disorder\|metabolic_disease\|hereditary_disease\|psychiatric_disorder` | `nervous_system_disorder\|metabolic_disease\|hereditary_disease` |
| `MONDO:0012414` | Neuronal ceroid lipofuscinosis 10 | `nervous_system_disorder\|metabolic_disease\|hereditary_disease\|psychiatric_disorder` | `nervous_system_disorder\|metabolic_disease\|hereditary_disease` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0035423` | `Triglyceride deposit cardiomyovasculopathy` | `Primary triglyceride deposit cardiomyovasculopathy` |
| `MONDO:0001252` | `Plummer disease` | `Toxic multinodular goitre` |
| `MONDO:0018820` | `Recurrent metabolic encephalomyopathic crises-rhabdomyolysis-cardiac arrhythmia-intellectual disa...` | `TANGO2 deficiency syndrome` |
| `MONDO:0011690` | `Camurati-engelmann disease, type 2` | `Camurati-engelmann disease type 2` |
| `MONDO:0009644` | `Sulfite oxidase deficiency due to molybdenum cofactor deficiency type b` | `Sulfite oxidase deficiency due to molybdenum cofactor deficiency type b1` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0010417` | Syndromic x-linked intellectual disability najm... | `syndromic X-linked intellectual disability Najm type; mental retardation and microcephaly with po...` | `syndromic X-linked intellectual disability Najm type; mental retardation, X-linked, syndromic, Na...` |
| `MONDO:0035423` | Primary triglyceride deposit cardiomyovasculopathy | `TGCV; Neutral lipid storage disease with severe cardiovascular involvement` | `primary neutral lipid storage disease with severe cardiovascular involvement; P-TGCV` |
| `MONDO:0005364` | Graves disease | `grave's disease; exophthalmic goitre; exophthalmic goiter; Graves disease` | `grave's disease; exophthalmic goiter; Graves disease` |
| `MONDO:0014035` | Severe intellectual disability-progressive spas... | `severe intellectual disability-progressive spastic diplegia syndrome; neurodevelopmental disorder...` | `severe intellectual disability-progressive spastic diplegia syndrome; neurodevelopmental disorder...` |
| `MONDO:0014629` | Autoimmune interstitial lung disease-arthritis ... | `copa syndrome; copa defect; COPA Syndrome` | `autoinflammation and autoimmunity, systemic, with immune dysregulation; COPA Syndrome` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 21 |
| `anatomical_id` | 22128 | 22127 |
| `anatomical_name` | 21997 | 21996 |
| `benign_malignant` | 22064 | 22063 |
| `core` | 22 | 21 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 56 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 113 | 112 |
| `is_glucose_dysfunction` | 23 | 22 |
| `is_infectious_disease` | 22 | 21 |
| `is_malignant_cancer` | 113 | 112 |
| `is_psychiatric_disease` | 113 | 112 |
| `level` | 1870 | 1869 |
| `mondo_top_grouping` | 67 | 66 |
| `mondo_txgnn` | 22 | 21 |
| `name` | 0 | 0 |
| `new_id` | 22754 | 22753 |
| `precancerous` | 22 | 21 |
| `prevalence_experimental` | 15671 | 15670 |
| `prevalence_world` | 15536 | 15535 |
| `speciality_breast` | 22 | 21 |
| `speciality_cardiovascular` | 22 | 21 |
| `speciality_chromosomal` | 22 | 21 |
| `speciality_connective_tissue` | 22 | 21 |
| `speciality_dermatologic` | 22 | 21 |
| `speciality_ear_nose_throat` | 22 | 21 |
| `speciality_endocrine` | 22 | 21 |
| `speciality_eye_and_adnexa` | 22 | 21 |
| `speciality_gastrointestinal` | 22 | 21 |
| `speciality_hematologic` | 22 | 21 |
| `speciality_immune` | 22 | 21 |
| `speciality_infection` | 22 | 21 |
| `speciality_metabolic` | 22 | 21 |
| `speciality_musculoskeletal` | 22 | 21 |
| `speciality_neoplasm` | 22 | 21 |
| `speciality_neurological` | 22 | 21 |
| `speciality_obstetric` | 22 | 21 |
| `speciality_poisoning_and_toxicity` | 22 | 21 |
| `speciality_psychiatric` | 22 | 21 |
| `speciality_renal_and_urinary` | 22 | 21 |
| `speciality_reproductive` | 22 | 21 |
| `speciality_respiratory` | 22 | 21 |
| `speciality_syndromic` | 22 | 21 |
| `supergroup` | 22 | 21 |
| `synonyms` | 4879 | 4877 |
| `txgnn` | 23 | 22 |
| `unmet_medical_need` | 15536 | 15535 |
