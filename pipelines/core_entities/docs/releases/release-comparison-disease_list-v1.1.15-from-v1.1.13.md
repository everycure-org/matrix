# Release Comparison Report

**New Release:** `v1.1.15-disease_list`

**Base Release:** `v1.1.13-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.1.15/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.1.13/ec-disease-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 13

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0034820` | Cleft lip and palate-craniofacial dysmorphism-congenital heart defect-hearing loss syndrome |
| `MONDO:0035661` | Traf7-associated heart defect-digital anomalies-facial dysmorphism-motor and speech delay syndrome |
| `MONDO:0004914` | Celiac artery stenosis from compression by median arcuate ligament of diaphragm |
| `MONDO:0009327` | Heart, malformation of |
| `MONDO:0011961` | Hereditary sensory and autonomic neuropathy type 1b |
| `MONDO:0023243` | Glass-chapman-hockley syndrome |
| `MONDO:0009027` | Cramps, familial adolescent |
| `MONDO:0035529` | Infantile-onset pulmonary alveolar proteinosis-hypogammaglobulinemia |
| `MONDO:0035660` | Gnao1-related developmental delay-seizures-movement disorder spectrum |
| `MONDO:0010553` | Charcot-marie-tooth peroneal muscular atrophy and friedreich ataxia, combined |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `anatomical_deformity` | 1 |
| `core` | 1 |
| `deleted` | 1 |
| `harrisons_view` | 42 |
| `is_benign_tumour` | 1 |
| `is_glucose_dysfunction` | 1 |
| `is_infectious_disease` | 1 |
| `is_malignant_cancer` | 1 |
| `is_psychiatric_disease` | 1 |
| `level` | 1 |
| `mondo_top_grouping` | 22730 |
| `mondo_txgnn` | 1 |
| `name` | 14 |
| `new_id` | 1 |
| `precancerous` | 1 |
| `speciality_breast` | 1 |
| `speciality_cardiovascular` | 1 |
| `speciality_chromosomal` | 1 |
| `speciality_connective_tissue` | 1 |
| `speciality_dermatologic` | 1 |
| `speciality_ear_nose_throat` | 1 |
| `speciality_endocrine` | 1 |
| `speciality_eye_and_adnexa` | 1 |
| `speciality_gastrointestinal` | 1 |
| `speciality_hematologic` | 1 |
| `speciality_immune` | 1 |
| `speciality_infection` | 1 |
| `speciality_metabolic` | 1 |
| `speciality_musculoskeletal` | 1 |
| `speciality_neoplasm` | 1 |
| `speciality_neurological` | 1 |
| `speciality_obstetric` | 1 |
| `speciality_poisoning_and_toxicity` | 1 |
| `speciality_psychiatric` | 1 |
| `speciality_renal_and_urinary` | 1 |
| `speciality_reproductive` | 1 |
| `speciality_respiratory` | 1 |
| `speciality_syndromic` | 1 |
| `supergroup` | 1 |
| `synonyms` | 2979 |
| `txgnn` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `anatomical_deformity`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `core`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `deleted`
*(Full comparison of all changed values)*

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `True` |

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0014952` | Intellectual disability-epilepsy-extrapyramidal... | `syndromic_disease\|hereditary_disease` | `syndromic_disease\|hereditary_disease\|nervous_system_disorder` |
| `MONDO:0005067` | Monophasic synovial sarcoma | `cancer_or_benign_tumor` | `hereditary_disease\|cancer_or_benign_tumor` |
| `MONDO:0006402` | Salivary gland basal cell adenocarcinoma | `integumentary_system_disorder\|digestive_system_disorder\|otorhinolaryngologic_disease\|cancer_or_be...` | `integumentary_system_disorder\|digestive_system_disorder\|cancer_or_benign_tumor` |
| `MONDO:0976127` | Muggenthaler-chowdhury-chioza syndrome | `hereditary_disease` | `syndromic_disease\|hereditary_disease` |
| `MONDO:0018360` | Neonatal lupus erythematosus | `connective_tissue_disorder\|immune_system_disorder` | `hereditary_disease\|connective_tissue_disorder\|immune_system_disorder` |

#### `is_benign_tumour`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `is_glucose_dysfunction`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `is_infectious_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `is_malignant_cancer`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `is_psychiatric_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `level`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `subgroup` | `*None*` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0030708` | Trichomonas cervicitis | `infectious_disease\|reproductive_system_disorder` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0013426` | Aneurysm-osteoarthritis syndrome | `cardiovascular_disorder\|connective_tissue_disorder\|hereditary_disease\|syndromic_disease` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0100631` | Sleep-related hypermotor epilepsy | `nervous_system_disorder` | `disease_by_body_system_or_component` |
| `MONDO:0004398` | Mediastinal schwannoma | `cancer_or_benign_tumor\|nervous_system_disorder` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0019621` | Chronic pneumonitis of infancy | `inflammatory_disease\|respiratory_system_disorder` | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` |

#### `mondo_txgnn`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `other` | `*None*` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0859008` | `Neurofibromatosis/schwannomatosis` | `Mosaic neurofibromatosis/schwannomatosis` |
| `MONDO:0014602` | `Hogue-janssens syndrome 1` | `Houge-janssens syndrome 1` |
| `MONDO:0043693` | `Alcoholic liver diseases` | `Alcoholic liver disease` |
| `MONDO:0023124` | `Familial pulmonary arterial hypertension leucopenia and atrial septal defect` | `Dursun syndrome` |
| `MONDO:0008340` | `Ptosis, hereditary congenital, 1` | `Congenital ptosis` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `*None*` | `MONDO:0000030` |

#### `precancerous`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_breast`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_cardiovascular`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_chromosomal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_connective_tissue`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_dermatologic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_ear_nose_throat`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_endocrine`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_eye_and_adnexa`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_gastrointestinal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_hematologic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_immune`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_infection`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_metabolic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_musculoskeletal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_neoplasm`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_neurological`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `True` | `*None*` |

#### `speciality_obstetric`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_poisoning_and_toxicity`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_psychiatric`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_renal_and_urinary`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_reproductive`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_respiratory`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `speciality_syndromic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `False` | `*None*` |

#### `supergroup`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `NNNI` | `*None*` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0008865` | Bietti crystalline corneoretinal dystrophy | `Bietti's crystalline dystrophy; Bietti crystalline retinopathy; Bietti crystalline corneoretinal ...` | `Bietti crystalline retinopathy; Bietti crystalline corneoretinal dystrophy; BCD` |
| `MONDO:0001658` | Nontoxic goiter | `nontoxic goiter; non-toxic simple goitre; non-toxic simple goiter; non-toxic goitre; non-toxic go...` | `nontoxic goiter; non-toxic simple goiter; non-toxic goiter; euthyroid goitre; euthyroid goiter` |
| `MONDO:0002450` | Prostatic adenoma | `prostate gland adenoma; prostate adenoma; benign adenoma of prostate; adenoma of the prostate; ad...` | `prostate gland adenoma; adenoma of the prostate` |
| `MONDO:0007286` | Cataract 30 | `cataract type 30; cataract Coppock-like; cataract 30 pulverulent; cataract 30; CTRCT30` | `cataract type 30; cataract 30; CTRCT30` |
| `MONDO:0005946` | Rhinosporidiosis | `infection by Rhinosporidium seeberi; Rhinosporidium seeberi infectious disease; Rhinosporidium se...` | `Rhinosporidium seeberi infectious disease; Rhinosporidium seeberi disease or disorder; Rhinospori...` |

#### `txgnn`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `mental_health_disorder\|neurodegenerative_disease` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 21 | 22 |
| `anatomical_id` | 22127 | 22114 |
| `anatomical_name` | 21996 | 21983 |
| `benign_malignant` | 22063 | 22050 |
| `core` | 21 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 112 | 113 |
| `is_glucose_dysfunction` | 22 | 23 |
| `is_infectious_disease` | 21 | 22 |
| `is_malignant_cancer` | 112 | 113 |
| `is_psychiatric_disease` | 112 | 113 |
| `level` | 1869 | 1869 |
| `mondo_top_grouping` | 66 | 24 |
| `mondo_txgnn` | 21 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22753 | 22739 |
| `precancerous` | 21 | 22 |
| `prevalence_experimental` | 15670 | 15659 |
| `prevalence_world` | 15535 | 15524 |
| `speciality_breast` | 21 | 22 |
| `speciality_cardiovascular` | 21 | 22 |
| `speciality_chromosomal` | 21 | 22 |
| `speciality_connective_tissue` | 21 | 22 |
| `speciality_dermatologic` | 21 | 22 |
| `speciality_ear_nose_throat` | 21 | 22 |
| `speciality_endocrine` | 21 | 22 |
| `speciality_eye_and_adnexa` | 21 | 22 |
| `speciality_gastrointestinal` | 21 | 22 |
| `speciality_hematologic` | 21 | 22 |
| `speciality_immune` | 21 | 22 |
| `speciality_infection` | 21 | 22 |
| `speciality_metabolic` | 21 | 22 |
| `speciality_musculoskeletal` | 21 | 22 |
| `speciality_neoplasm` | 21 | 22 |
| `speciality_neurological` | 21 | 22 |
| `speciality_obstetric` | 21 | 22 |
| `speciality_poisoning_and_toxicity` | 21 | 22 |
| `speciality_psychiatric` | 21 | 22 |
| `speciality_renal_and_urinary` | 21 | 22 |
| `speciality_reproductive` | 21 | 22 |
| `speciality_respiratory` | 21 | 22 |
| `speciality_syndromic` | 21 | 22 |
| `supergroup` | 21 | 22 |
| `synonyms` | 4877 | 5147 |
| `txgnn` | 22 | 23 |
| `unmet_medical_need` | 15535 | 15524 |
