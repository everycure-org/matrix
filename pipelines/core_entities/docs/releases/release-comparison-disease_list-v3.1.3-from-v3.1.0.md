# Release Comparison Report

**New Release:** `v3.1.3-disease_list`

**Base Release:** `v3.1.0-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v3.1.3/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v3.1.0/ec-disease-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 14

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0012144` | Waardenburg syndrome type 2d |
| `MONDO:0035551` | Cathepsin a-related arteriopathy-strokes-leukoencephalopathy |
| `MONDO:0976228` | Immunodeficiency 132b |
| `MONDO:0859002` | Intellectual disability-early-onset cataract-microcephaly syndrome |
| `MONDO:0850095` | X-linked severe syndromic thoracic aortic aneurysm and dissection |
| `MONDO:0030528` | Immunodeficiency 93 and hypertrophic cardiomyopathy |
| `MONDO:0850059` | Hereditary persistence of fetal hemoglobin-intellectual disability syndrome |
| `MONDO:0975834` | Immunodeficiency 128 |
| `MONDO:0022070` | Cantu sanchez-corona hernandez syndrome |
| `MONDO:0035694` | Combined immunodeficiency due to rela haploinsufficiency |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `anatomical_deformity` | 1 |
| `core` | 1 |
| `deleted` | 1 |
| `harrisons_view` | 155 |
| `is_benign_tumour` | 1 |
| `is_glucose_dysfunction` | 1 |
| `is_infectious_disease` | 1 |
| `is_malignant_cancer` | 1 |
| `is_psychiatric_disease` | 1 |
| `level` | 1 |
| `mondo_top_grouping` | 56 |
| `mondo_txgnn` | 1 |
| `name` | 27 |
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
| `strategically_viable` | 3 |
| `strategically_viable_assigned_by` | 2 |
| `supergroup` | 1 |
| `synonyms` | 123 |
| `txgnn` | 1 |
| `unmet_medical_need` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `anatomical_deformity`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `core`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `deleted`
*(Full comparison of all changed values)*

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `True` |

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0030374` | Whim syndrome 2 | `hereditary_disease` | `hereditary_disease\|immune_system_disorder\|hematologic_disorder` |
| `MONDO:0002577` | Extrahepatic bile duct rhabdomyosarcoma | `digestive_system_disorder\|endocrine_system_disorder\|cancer_or_benign_tumor` | `digestive_system_disorder\|cancer_or_benign_tumor` |
| `MONDO:0033492` | Coffin-siris syndrome 6 | `hereditary_disease\|nervous_system_disorder` | `musculoskeletal_system_disorder\|syndromic_disease\|hereditary_disease\|disorder_of_development_or_m...` |
| `MONDO:0010772` | Leber optic atrophy and dystonia | `musculoskeletal_system_disorder\|hereditary_disease\|cardiovascular_disorder\|metabolic_disease\|nerv...` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder\|disorder_of_development_or_morphogen...` |
| `MONDO:0000374` | Bile duct carcinoma in situ | `digestive_system_disorder\|endocrine_system_disorder\|cancer_or_benign_tumor` | `digestive_system_disorder\|cancer_or_benign_tumor` |

#### `is_benign_tumour`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `is_glucose_dysfunction`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `is_infectious_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `is_malignant_cancer`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `is_psychiatric_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `level`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `clinically_recognized` | `*None*` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0030374` | Whim syndrome 2 | `disease_by_etiologic_mechanism` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0033492` | Coffin-siris syndrome 6 | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |
| `MONDO:0013483` | Obesity, hyperphagia, and developmental delay | `disease_by_etiologic_mechanism` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0030991` | Bile acid conjugation defect 1 | `disease_by_etiologic_mechanism` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process` |
| `MONDO:0007099` | Familial visceral amyloidosis | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |

#### `mondo_txgnn`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `other` | `*None*` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0008090` | `Cyclic hematopoiesis` | `Cyclic neutropenia` |
| `MONDO:0017389` | `Tetrahydrobiopterin-responsive hyperphenylalaninemia/phenylketonuria` | `Tetrahydrobiopterin-responsive phenylketonuria` |
| `MONDO:0100089` | `Gata1-related x-linked cytopenia` | `Gata1-related cytopenia` |
| `MONDO:0013805` | `Intellectual disability, autosomal dominant 13` | `Cortical dysplasia, complex, with other brain malformations 13` |
| `MONDO:0030919` | `Intellectual disability, autosomal dominant 53` | `Intellectual developmental disorder, autosomal dominant 53` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `*None*` | `MONDO:0007526` |

#### `precancerous`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_breast`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_cardiovascular`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_chromosomal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_connective_tissue`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `True` | `*None*` |

#### `speciality_dermatologic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_ear_nose_throat`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_endocrine`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_eye_and_adnexa`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_gastrointestinal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_hematologic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_immune`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_infection`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_metabolic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_musculoskeletal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `True` | `*None*` |

#### `speciality_neoplasm`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_neurological`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_obstetric`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_poisoning_and_toxicity`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_psychiatric`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_renal_and_urinary`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_reproductive`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_respiratory`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `speciality_syndromic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `True` | `*None*` |

#### `strategically_viable`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0030308` | Immunodeficiency 82 with systemic inflammation | `False` | `True` |
| `MONDO:0044318` | Jansen de Vries syndrome | `False` | `True` |
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `False` | `*None*` |

#### `strategically_viable_assigned_by`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0030308` | Immunodeficiency 82 with systemic inflammation | `not assigned` | `human` |
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `not assigned` | `*None*` |

#### `supergroup`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `NNNI` | `*None*` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0010787` | Kearns-sayre syndrome | `['Kearns-Sayre syndrome']` | `['ophthalmoplegia, progressive external, with ragged-Red fibres'
 'ophthalmoplegia, progressive e...` |
| `MONDO:0011933` | Alg2-congenital disorder of glycosylation | `['mannosyltransferase 2 deficiency'
 'congenital disorder of glycosylation, type Ii'
 'congenital...` | `['mannosyltransferase 2 deficiency'
 'congenital disorder of glycosylation, type Ii'
 'congenital...` |
| `MONDO:0011670` | Ehlers-danlos syndrome due to tenascin-x defici... | `['Ehlers-Danlos syndrome, classic-like, 1'
 'Ehlers-Danlos syndrome, classic-like type'
 'Ehlers-...` | `['Ehlers-Danlos syndrome, classic-like, 1'
 'Ehlers-Danlos syndrome due to tenascin-X deficiency']` |
| `MONDO:0011142` | Ehlers-danlos syndrome, musculocontractural type | `['musculocontractural Ehlers-Danlos syndrome'
 'adducted thumbs-arthrogryposis syndrome, Dundar t...` | `['musculocontractural Ehlers-Danlos syndrome' 'musculocontractural EDS'
 'mcEDS' 'adducted thumb-...` |
| `MONDO:0013349` | Alg11-congenital disorder of glycosylation | `['congenital disorder of glycosylation type Ip'
 'congenital disorder of glycosylation type 1p'
 ...` | `['congenital disorder of glycosylation type Ip'
 'congenital disorder of glycosylation type 1p'
 ...` |

#### `txgnn`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `inflammatory_disease\|metabolic_disorder` | `*None*` |

#### `unmet_medical_need`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0034021` | obsolete spondylodysplastic Ehlers-Danlos syndrome | `21.5` | `nan` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 23 |
| `anatomical_id` | 22602 | 22589 |
| `anatomical_name` | 22471 | 22458 |
| `benign_malignant` | 22524 | 22511 |
| `core` | 22 | 23 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 56 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 15975 | 15965 |
| `is_glucose_dysfunction` | 15886 | 15876 |
| `is_infectious_disease` | 22 | 23 |
| `is_malignant_cancer` | 15976 | 15966 |
| `is_psychiatric_disease` | 15992 | 15982 |
| `level` | 1868 | 1869 |
| `mondo_top_grouping` | 27 | 28 |
| `mondo_txgnn` | 22 | 23 |
| `name` | 0 | 0 |
| `new_id` | 23227 | 23212 |
| `precancerous` | 22 | 23 |
| `speciality_breast` | 22 | 23 |
| `speciality_cardiovascular` | 22 | 23 |
| `speciality_chromosomal` | 22 | 23 |
| `speciality_connective_tissue` | 22 | 23 |
| `speciality_dermatologic` | 22 | 23 |
| `speciality_ear_nose_throat` | 22 | 23 |
| `speciality_endocrine` | 22 | 23 |
| `speciality_eye_and_adnexa` | 22 | 23 |
| `speciality_gastrointestinal` | 22 | 23 |
| `speciality_hematologic` | 22 | 23 |
| `speciality_immune` | 22 | 23 |
| `speciality_infection` | 22 | 23 |
| `speciality_metabolic` | 22 | 23 |
| `speciality_musculoskeletal` | 22 | 23 |
| `speciality_neoplasm` | 22 | 23 |
| `speciality_neurological` | 22 | 23 |
| `speciality_obstetric` | 22 | 23 |
| `speciality_poisoning_and_toxicity` | 22 | 23 |
| `speciality_psychiatric` | 22 | 23 |
| `speciality_renal_and_urinary` | 22 | 23 |
| `speciality_reproductive` | 22 | 23 |
| `speciality_respiratory` | 22 | 23 |
| `speciality_syndromic` | 22 | 23 |
| `strategically_viable` | 22 | 23 |
| `strategically_viable_assigned_by` | 22 | 23 |
| `supergroup` | 22 | 23 |
| `synonyms` | 22 | 23 |
| `txgnn` | 425 | 426 |
| `unmet_medical_need` | 15886 | 15876 |
