# Release Comparison Report

**New Release:** `v3.0.0-disease_list`

**Base Release:** `v2.0.11-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v3.0.0/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v2.0.11/ec-disease-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 501

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:1060220` | Split cord malformation type i |
| `MONDO:0975897` | Post 5-alpha-reductase inhibitors treatment syndrome |
| `MONDO:0700323` | Systemic lupus erythematosus related to c4a |
| `MONDO:0979310` | Lymphoepithelial cyst of the pancreas |
| `MONDO:0100641` | Chemotherapy-induced neuropathy |
| `MONDO:0980699` | Neurodevelopmental disorder with growth impairment, quadriparesis, and poor or absent speech |
| `MONDO:0979354` | Sickle cell disease due to hemoglobin s and a non-s/non-c hemoglobin variant |
| `MONDO:0980935` | Microcephaly, progressive, with simplified gyral pattern and cerebellar hypoplasia |
| `MONDO:1060176` | Systemic lupus erythematosus related to c1s |
| `MONDO:0978310` | Adenomatoid tumour of the peritoneum |

### Removed Rows
**Total:** 1

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0016061` | Immunodeficiency with factor h anomaly |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `harrisons_view` | 7 |
| `is_benign_tumour` | 15474 |
| `is_glucose_dysfunction` | 15501 |
| `is_infectious_disease` | 1588 |
| `is_malignant_cancer` | 15475 |
| `is_psychiatric_disease` | 15491 |
| `mondo_top_grouping` | 3 |
| `name` | 1 |
| `synonyms` | 30 |
| `unmet_medical_need` | 16 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0958106` | Congenital insensitivity to pain syndrome, mars... | `syndromic_disease\|hereditary_disease` | `syndromic_disease\|hereditary_disease\|nervous_system_disorder` |
| `MONDO:0019237` | Inborn disorder of pyridoxine metabolism | `hereditary_disease\|metabolic_disease` | `hereditary_disease\|metabolic_disease\|nutritional_disorder` |
| `MONDO:0009459` | Channelopathy-associated congenital insensitivi... | `syndromic_disease\|hereditary_disease` | `syndromic_disease\|hereditary_disease\|nervous_system_disorder` |
| `MONDO:0012407` | Pyridoxal phosphate-responsive seizures | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder\|nutritional_disorder` |
| `MONDO:0009945` | Pyridoxine-dependent epilepsy | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder\|nutritional_disorder` |

#### `is_benign_tumour`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004136` | Ovarian endometrioid cystadenoma | `True` | `*None*` |
| `MONDO:0012154` | Myopia 6 | `False` | `*None*` |
| `MONDO:0024645` | Retroperitoneal neoplasm | `False` | `*None*` |
| `MONDO:0017077` | Myelocystocele | `False` | `*None*` |
| `MONDO:0859565` | Atrioventricular septal defect | `False` | `*None*` |

#### `is_glucose_dysfunction`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004136` | Ovarian endometrioid cystadenoma | `False` | `*None*` |
| `MONDO:0012154` | Myopia 6 | `False` | `*None*` |
| `MONDO:0024645` | Retroperitoneal neoplasm | `False` | `*None*` |
| `MONDO:0017077` | Myelocystocele | `False` | `*None*` |
| `MONDO:0859565` | Atrioventricular septal defect | `False` | `*None*` |

#### `is_infectious_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0000295` | Acanthocephaliasis | `True` | `False` |
| `MONDO:0006806` | Intermediate uveitis | `True` | `False` |
| `MONDO:0018769` | Isosporiasis | `True` | `False` |
| `MONDO:0002536` | Skin papilloma | `True` | `False` |
| `MONDO:0005968` | Sporotrichosis | `True` | `False` |

#### `is_malignant_cancer`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004136` | Ovarian endometrioid cystadenoma | `False` | `*None*` |
| `MONDO:0012154` | Myopia 6 | `False` | `*None*` |
| `MONDO:0024645` | Retroperitoneal neoplasm | `True` | `*None*` |
| `MONDO:0017077` | Myelocystocele | `False` | `*None*` |
| `MONDO:0859565` | Atrioventricular septal defect | `False` | `*None*` |

#### `is_psychiatric_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004136` | Ovarian endometrioid cystadenoma | `False` | `*None*` |
| `MONDO:0012154` | Myopia 6 | `False` | `*None*` |
| `MONDO:0024645` | Retroperitoneal neoplasm | `False` | `*None*` |
| `MONDO:0017077` | Myelocystocele | `False` | `*None*` |
| `MONDO:0859565` | Atrioventricular septal defect | `False` | `*None*` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0010590` | Fg syndrome 1 | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |
| `MONDO:0032485` | Intellectual developmental disorder 61 | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |
| `MONDO:0100000` | Med12-related intellectual disability syndrome | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0011010` | `Matthew-wood syndrome` | `Microphthalmia, syndromic 9` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0012634` | Craniofacial dysplasia - osteopenia syndrome | `['Hamamy syndrome']` | `['IRX5-related craniofacial dysostosis with osteopenia, intellectual disability, and dental anoma...` |
| `MONDO:0009459` | Channelopathy-associated congenital insensitivi... | `['insensitivity to pain, congenital' 'channelopathy-associated CIP']` | `['indifference to pain, congenital, autosomal recessive'
 'congenital insensitivity to pain with ...` |
| `MONDO:0014536` | Thrombocytopenia 5 | `['thrombocytopenia type 5' 'thrombocytopenia caused by mutation in ETV6'
 'thrombocytopenia 5' 'E...` | `['thrombocytopenia caused by mutation in ETV6' 'thrombocytopenia 5'
 'ETV6-related thrombocytopen...` |
| `MONDO:0014386` | Platelet-type bleeding disorder 18 | `['platelet-type bleeding disorder 18'
 'inherited bleeding disorder, platelet-type caused by muta...` | `['platelet-type bleeding disorder 18'
 'inherited bleeding disorder, platelet-type caused by muta...` |
| `MONDO:0011103` | Autosomal dominant nonsyndromic hearing loss 3a | `[]` | `['GJB2-related autosomal dominant nonsyndromic hearing loss'
 'GJB2-AD NSHL']` |

#### `unmet_medical_need`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:1010182` | Premenstrual dysphoric disorder | `nan` | `15.75` |
| `MONDO:0009756` | Niemann-pick disease type a | `nan` | `24.5` |
| `MONDO:0018982` | Niemann-pick disease type c | `nan` | `22.25` |
| `MONDO:0021761` | Acral dysostosis dyserythropoiesis syndrome | `21.5` | `nan` |
| `MONDO:0005567` | Substance withdrawal syndrome | `18.5` | `nan` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22107 | 22607 |
| `anatomical_name` | 21976 | 22476 |
| `benign_malignant` | 22043 | 22529 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 114 | 15980 |
| `is_glucose_dysfunction` | 24 | 15891 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 114 | 15981 |
| `is_psychiatric_disease` | 114 | 15997 |
| `level` | 1868 | 1868 |
| `mondo_top_grouping` | 24 | 27 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22732 | 23232 |
| `precancerous` | 22 | 22 |
| `prevalence_experimental` | 15653 | 16153 |
| `prevalence_world` | 15518 | 16018 |
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
| `strategically_viable_assigned_by` | 22 | 22 |
| `supergroup` | 22 | 22 |
| `synonyms` | 22 | 22 |
| `txgnn` | 24 | 425 |
| `unmet_medical_need` | 15518 | 15891 |
