# Release Comparison Report

**New Release:** `v1.1.5-disease_list`

**Base Release:** `v1.1.0-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.1.5/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.1.0/ec-disease-list.parquet`

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
| `MONDO:0800504` | Idiopathic pulmonary fibrosis |

### Removed Rows
**Total:** 3

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0005731` | Dipetalonemiasis |
| `MONDO:0100317` | Deficiency of adenosine deaminase 2 |
| `MONDO:0800197` | Achromatopsia 6 |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `harrisons_view` | 4 |
| `mondo_top_grouping` | 5 |
| `name` | 5 |
| `synonyms` | 14 |

### Examples by Column

*Up to 5 examples per column*

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0008436` | Sneddon syndrome | `integumentary_system_disorder\|syndromic_disease\|hereditary_disease\|connective_tissue_disorder\|car...` | `integumentary_system_disorder\|cardiovascular_disorder` |
| `MONDO:0971179` | Arterial tortuosity-bone fragility syndrome | `syndromic_disease\|hereditary_disease` | `integumentary_system_disorder\|musculoskeletal_system_disorder\|syndromic_disease\|hereditary_diseas...` |
| `MONDO:0004813` | Tuberculous pneumothorax | `respiratory_system_disorder` | `respiratory_system_disorder\|infectious_disease\|inflammatory_disease` |
| `MONDO:0007088` | Alzheimer disease type 1 | `psychiatric_disorder\|hereditary_disease\|nervous_system_disorder` | `psychiatric_disorder\|hereditary_disease\|cardiovascular_disorder\|metabolic_disease\|nervous_system_...` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0008436` | Sneddon syndrome | `cardiovascular_disorder\|connective_tissue_disorder\|hereditary_disease\|syndromic_disease\|integumen...` | `cardiovascular_disorder\|integumentary_system_disorder` |
| `MONDO:0971179` | Arterial tortuosity-bone fragility syndrome | `hereditary_disease\|syndromic_disease` | `disorder_of_development_or_morphogenesis\|cardiovascular_disorder\|connective_tissue_disorder\|hered...` |
| `MONDO:0004813` | Tuberculous pneumothorax | `respiratory_system_disorder` | `inflammatory_disease\|infectious_disease\|respiratory_system_disorder` |
| `MONDO:0004919` | Infected hydrocele | `reproductive_system_disorder` | `post_infectious_disorder\|reproductive_system_disorder` |
| `MONDO:0007088` | Alzheimer disease type 1 | `nervous_system_disorder\|hereditary_disease\|psychiatric_disorder` | `nervous_system_disorder\|metabolic_disease\|cardiovascular_disorder\|hereditary_disease\|psychiatric_...` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0018146` | `Idiopathic macular telangiectasia type 1` | `Macular telangiectasia type 1` |
| `MONDO:0012398` | `Retinal cone dystrophy 3a` | `Achromatopsia 6` |
| `MONDO:0014306` | `Vasculitis due to ada2 deficiency` | `Deficiency of adenosine deaminase 2` |
| `MONDO:0018147` | `Idiopathic macular telangiectasia type 3` | `Macular telangiectasia type 3` |
| `MONDO:0018013` | `Non-immunoglobulin-mediated membranoproliferative glomerulonephritis` | `Complement 3 glomerulopathy` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0013293` | Isolated microphthalmia 6 | `posterior nonsyndromic microphthalmia; microphthalmia, isolated type 6; isolated microphthalmia t...` | `posterior nonsyndromic microphthalmia; microphthalmia, isolated type 6; isolated microphthalmia t...` |
| `MONDO:0018146` | Macular telangiectasia type 1 | `visible and exudative idiopathic juxtafoveolar retinal telangiectasis; aneurysmal telangiectasia` | `visible and exudative idiopathic juxtafoveolar retinal telangiectasis; idiopathic macular telangi...` |
| `MONDO:0004169` | Premenstrual tension | `*None*` | `premenstrual syndrome; PMS` |
| `MONDO:0971179` | Arterial tortuosity-bone fragility syndrome | `*None*` | `EMILIN1-related arterial tortuosity syndrome` |
| `MONDO:0013576` | Recurrent infections associated with rare immun... | `recurrent infections associated with rare immunoglobulin isotypes deficiency; kappa-chain deficie...` | `recurrent infections associated with rare immunoglobulin isotypes deficiency; kappa-chain deficie...` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22130 | 22128 |
| `anatomical_name` | 21999 | 21997 |
| `benign_malignant` | 22066 | 22064 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 56 | 56 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 112 | 113 |
| `is_glucose_dysfunction` | 22 | 23 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 112 | 113 |
| `is_psychiatric_disease` | 112 | 113 |
| `level` | 1871 | 1870 |
| `mondo_top_grouping` | 67 | 67 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22756 | 22754 |
| `precancerous` | 22 | 22 |
| `prevalence_experimental` | 15673 | 15671 |
| `prevalence_world` | 15538 | 15536 |
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
| `supergroup` | 22 | 22 |
| `synonyms` | 4882 | 4879 |
| `txgnn` | 22 | 23 |
| `unmet_medical_need` | 15538 | 15536 |
