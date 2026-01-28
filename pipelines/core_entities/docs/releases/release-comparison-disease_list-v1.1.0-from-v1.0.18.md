# Release Comparison Report

**New Release:** `v1.1.0-disease_list`

**Base Release:** `v1.0.18-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.1.0/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.0.18/ec-disease-list.parquet`

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
| `MONDO:0017770` | Robinow-like syndrome |
| `MONDO:0025303` | Anaplasmosis |
| `MONDO:0006518` | Sporadic creutzfeld jacob disease |
| `MONDO:0800460` | Asah1-related disorders |
| `MONDO:0001859` | Algoneurodystrophy |
| `MONDO:0035122` | Grin2b-related developmental delay, intellectual disability and autism spectrum disorder |
| `MONDO:0000638` | Benign glioma |
| `MONDO:0957221` | Spastic paraplegia 70, autosomal recessive |
| `MONDO:0020474` | Cheirospondyloenchondromatosis |
| `MONDO:0018773` | Autosomal dominant distal axonal motor neuropathy-myofibrillar myopathy syndrome |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `anatomical_deformity` | 3 |
| `core` | 3 |
| `deleted` | 3 |
| `harrisons_view` | 12047 |
| `is_benign_tumour` | 15709 |
| `is_glucose_dysfunction` | 15577 |
| `is_infectious_disease` | 1927 |
| `is_malignant_cancer` | 15596 |
| `is_psychiatric_disease` | 16649 |
| `level` | 4 |
| `mondo_top_grouping` | 12791 |
| `mondo_txgnn` | 11946 |
| `name` | 16 |
| `new_id` | 1 |
| `precancerous` | 3 |
| `prevalence_experimental` | 2 |
| `prevalence_world` | 2 |
| `speciality_breast` | 3 |
| `speciality_cardiovascular` | 3 |
| `speciality_chromosomal` | 3 |
| `speciality_connective_tissue` | 3 |
| `speciality_dermatologic` | 3 |
| `speciality_ear_nose_throat` | 3 |
| `speciality_endocrine` | 3 |
| `speciality_eye_and_adnexa` | 3 |
| `speciality_gastrointestinal` | 3 |
| `speciality_hematologic` | 3 |
| `speciality_immune` | 3 |
| `speciality_infection` | 3 |
| `speciality_metabolic` | 3 |
| `speciality_musculoskeletal` | 3 |
| `speciality_neoplasm` | 3 |
| `speciality_neurological` | 3 |
| `speciality_obstetric` | 3 |
| `speciality_poisoning_and_toxicity` | 3 |
| `speciality_psychiatric` | 3 |
| `speciality_renal_and_urinary` | 3 |
| `speciality_reproductive` | 3 |
| `speciality_respiratory` | 3 |
| `speciality_syndromic` | 3 |
| `supergroup` | 3 |
| `synonyms` | 11059 |
| `txgnn` | 4 |
| `unmet_medical_need` | 2 |

### Examples by Column

*Up to 5 examples per column*

#### `anatomical_deformity`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `core`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `True` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `True` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `deleted`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `True` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `True` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `True` |

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007284` | Cataract 20 multiple types | `disorder_of_visual_system\|hereditary_disease` | `hereditary_disease\|disorder_of_visual_system` |
| `MONDO:0024463` | Ovarian dysgenesis 1 | `reproductive_system_disorder\|hereditary_disease\|endocrine_system_disorder` | `hereditary_disease\|reproductive_system_disorder\|endocrine_system_disorder` |
| `MONDO:0002800` | Thrombophlebitis | `inflammatory_disease\|cardiovascular_disorder` | `cardiovascular_disorder\|inflammatory_disease` |
| `MONDO:0971034` | Thyroid gland cribriform morular carcinoma | `cancer_or_benign_tumor\|endocrine_system_disorder` | `endocrine_system_disorder\|cancer_or_benign_tumor` |
| `MONDO:0013748` | Ventricular septal defect 2 | `hereditary_disease\|disorder_of_development_or_morphogenesis\|cardiovascular_disorder` | `hereditary_disease\|cardiovascular_disorder\|disorder_of_development_or_morphogenesis` |

#### `is_benign_tumour`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007284` | Cataract 20 multiple types | `*None*` | `False` |
| `MONDO:0024463` | Ovarian dysgenesis 1 | `*None*` | `False` |
| `MONDO:0003671` | Septal myocardial infarction | `*None*` | `False` |
| `MONDO:0851102` | Pulmonary artery disease | `*None*` | `False` |
| `MONDO:0017597` | T-cell/histiocyte rich large b cell lymphoma | `*None*` | `False` |

#### `is_glucose_dysfunction`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007284` | Cataract 20 multiple types | `*None*` | `False` |
| `MONDO:0024463` | Ovarian dysgenesis 1 | `*None*` | `False` |
| `MONDO:0003671` | Septal myocardial infarction | `*None*` | `False` |
| `MONDO:0851102` | Pulmonary artery disease | `*None*` | `False` |
| `MONDO:0017597` | T-cell/histiocyte rich large b cell lymphoma | `*None*` | `False` |

#### `is_infectious_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0957561` | Encephalitis, acute, infection-induced, suscept... | `False` | `True` |
| `MONDO:0017879` | Hantavirus pulmonary syndrome | `False` | `True` |
| `MONDO:0004485` | Interstitial myocarditis | `False` | `True` |
| `MONDO:0024389` | Anaerobic bacteria infectious disease | `False` | `True` |
| `MONDO:0005831` | Lymph node tuberculosis | `False` | `True` |

#### `is_malignant_cancer`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007284` | Cataract 20 multiple types | `*None*` | `False` |
| `MONDO:0024463` | Ovarian dysgenesis 1 | `*None*` | `False` |
| `MONDO:0003671` | Septal myocardial infarction | `*None*` | `False` |
| `MONDO:0851102` | Pulmonary artery disease | `*None*` | `False` |
| `MONDO:0017597` | T-cell/histiocyte rich large b cell lymphoma | `*None*` | `True` |

#### `is_psychiatric_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007284` | Cataract 20 multiple types | `*None*` | `False` |
| `MONDO:0024463` | Ovarian dysgenesis 1 | `*None*` | `False` |
| `MONDO:0003671` | Septal myocardial infarction | `*None*` | `False` |
| `MONDO:0851102` | Pulmonary artery disease | `*None*` | `False` |
| `MONDO:0017597` | T-cell/histiocyte rich large b cell lymphoma | `*None*` | `False` |

#### `level`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `clinically_recognized` | `*None*` |
| `MONDO:0022650` | Cardiomyopathy diabetes deafness | `clinically_recognized` | `subgroup` |
| `MONDO:0010785` | Maternally-inherited diabetes and deafness | `subgroup` | `clinically_recognized` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `clinically_recognized` | `*None*` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007284` | Cataract 20 multiple types | `disorder_of_visual_system\|disorder_of_orbital_region\|hereditary_disease` | `disorder_of_visual_system\|hereditary_disease\|disorder_of_orbital_region` |
| `MONDO:0024463` | Ovarian dysgenesis 1 | `hereditary_disease\|endocrine_system_disorder\|reproductive_system_disorder` | `endocrine_system_disorder\|reproductive_system_disorder\|hereditary_disease` |
| `MONDO:0851102` | Pulmonary artery disease | `cardiovascular_disorder\|respiratory_system_disorder` | `respiratory_system_disorder\|cardiovascular_disorder` |
| `MONDO:0017597` | T-cell/histiocyte rich large b cell lymphoma | `cancer_or_benign_tumor\|immune_system_disorder\|hematologic_disorder` | `cancer_or_benign_tumor\|hematologic_disorder\|immune_system_disorder` |
| `MONDO:0002800` | Thrombophlebitis | `cardiovascular_disorder\|inflammatory_disease` | `inflammatory_disease\|cardiovascular_disorder` |

#### `mondo_txgnn`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0024463` | Ovarian dysgenesis 1 | `endocrine_system_disorder` | `other` |
| `MONDO:0003671` | Septal myocardial infarction | `cardiovascular_disorder` | `other` |
| `MONDO:0851102` | Pulmonary artery disease | `cardiovascular_disorder` | `other` |
| `MONDO:0017597` | T-cell/histiocyte rich large b cell lymphoma | `cancer_or_benign_tumor` | `other` |
| `MONDO:0002800` | Thrombophlebitis | `cardiovascular_disorder` | `other` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0014213` | `Intellectual disability-feeding difficulties-developmental delay-microcephaly syndrome` | `Ctcf-related neurodevelopmental disorder` |
| `MONDO:0970950` | `Rothmund-thomson syndrome, type 4` | `Rothmund-thomson syndrome type 4` |
| `MONDO:0004742` | `Primary cerebellar degeneration` | `obsolete primary cerebellar degeneration` |
| `MONDO:0014347` | `Rothmund-thomson syndrome, type 3` | `Rothmund-thomson syndrome type 3` |
| `MONDO:0015597` | `Pustulosis palmaris et plantaris` | `Palmoplantar pustulosis` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `*None*` | `MONDO:0001999` |

#### `precancerous`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `prevalence_experimental`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `True` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |

#### `prevalence_world`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `1-9 in 100,000` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `<1 in 100,000` | `*None*` |

#### `speciality_breast`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_cardiovascular`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `True` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_chromosomal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_connective_tissue`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_dermatologic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_ear_nose_throat`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_endocrine`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_eye_and_adnexa`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_gastrointestinal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_hematologic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_immune`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_infection`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `True` | `*None*` |

#### `speciality_metabolic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_musculoskeletal`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_neoplasm`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_neurological`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `True` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_obstetric`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_poisoning_and_toxicity`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_psychiatric`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_renal_and_urinary`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_reproductive`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `speciality_respiratory`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `True` | `*None*` |

#### `speciality_syndromic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `False` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `False` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `False` | `*None*` |

#### `supergroup`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `NNNI` | `*None*` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `NNNI` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `infection` | `*None*` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007284` | Cataract 20 multiple types | `cataract (disease) caused by mutation in CRYGS; CRYGS cataract (disease); CTRCT20` | `cataract (disease) caused by mutation in CRYGS; CTRCT20; CRYGS cataract (disease)` |
| `MONDO:0017597` | T-cell/histiocyte rich large b cell lymphoma | `T-cell/histiocyte-rich large B-cell lymphoma; T-cell rich/histiocyte-rich large B-cell lymphoma; ...` | `THRLBCL; T-cell/histiocyte-rich large B-cell lymphoma; T-cell/histiocyte rich lymphoma; T-cell ri...` |
| `MONDO:0002800` | Thrombophlebitis | `superficial thrombophlebitis of leg; thrombophlebitis of superficial veins of lower extremity; ph...` | `thrombophlebitis of superficial veins of lower extremity; thrombophlebitis of a superficial leg v...` |
| `MONDO:0016907` | Partial deletion of the long arm of chromosome 8 | `partial monosomy of the long arm of chromosome 8; partial deletion of chromosome 8q; partial mono...` | `partial monosomy of the long arm of chromosome 8; partial monosomy of chromosome 8q; partial dele...` |
| `MONDO:0013748` | Ventricular septal defect 2 | `CITED2 ventricular septal defect (disease); ventricular septal defect (disease) caused by mutatio...` | `ventricular septal defect type 2; ventricular septal defect 2; ventricular septal defect (disease...` |

#### `txgnn`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `cardiovascular_disorder\|neurodegenerative_disease` | `*None*` |
| `MONDO:0859254` | Hepatorenocardiac degenerative fibrosis | `hepatorenocardiác_degenerative_fibrosis_seems_to_be_related_to_liver__so_also__hepatorenocardiác_...` | `hepatorenocardi√°c_degenerative_fibrosis_seems_to_be_related_to_liver__so_also__hepatorenocardi√°...` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `cardiovascular_disorder\|inflammatory_disease` | `*None*` |
| `MONDO:0800029` | obsolete interstitial lung disease 2 | `inflammatory_disease\|autoimmune_diseases` | `*None*` |

#### `unmet_medical_need`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0004742` | obsolete primary cerebellar degeneration | `22.0` | `nan` |
| `MONDO:0017147` | obsolete idiopathic pulmonary arterial hyperten... | `22.5` | `nan` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 19 | 22 |
| `anatomical_id` | 22143 | 22130 |
| `anatomical_name` | 22012 | 21999 |
| `benign_malignant` | 22079 | 22066 |
| `core` | 19 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 53 | 56 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 15544 | 112 |
| `is_glucose_dysfunction` | 15544 | 22 |
| `is_infectious_disease` | 19 | 22 |
| `is_malignant_cancer` | 15544 | 112 |
| `is_psychiatric_disease` | 15544 | 112 |
| `level` | 1871 | 1871 |
| `mondo_top_grouping` | 63 | 67 |
| `mondo_txgnn` | 19 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22770 | 22756 |
| `precancerous` | 19 | 22 |
| `prevalence_experimental` | 15680 | 15673 |
| `prevalence_world` | 15545 | 15538 |
| `speciality_breast` | 19 | 22 |
| `speciality_cardiovascular` | 19 | 22 |
| `speciality_chromosomal` | 19 | 22 |
| `speciality_connective_tissue` | 19 | 22 |
| `speciality_dermatologic` | 19 | 22 |
| `speciality_ear_nose_throat` | 19 | 22 |
| `speciality_endocrine` | 19 | 22 |
| `speciality_eye_and_adnexa` | 19 | 22 |
| `speciality_gastrointestinal` | 19 | 22 |
| `speciality_hematologic` | 19 | 22 |
| `speciality_immune` | 19 | 22 |
| `speciality_infection` | 19 | 22 |
| `speciality_metabolic` | 19 | 22 |
| `speciality_musculoskeletal` | 19 | 22 |
| `speciality_neoplasm` | 19 | 22 |
| `speciality_neurological` | 19 | 22 |
| `speciality_obstetric` | 19 | 22 |
| `speciality_poisoning_and_toxicity` | 19 | 22 |
| `speciality_psychiatric` | 19 | 22 |
| `speciality_renal_and_urinary` | 19 | 22 |
| `speciality_reproductive` | 19 | 22 |
| `speciality_respiratory` | 19 | 22 |
| `speciality_syndromic` | 19 | 22 |
| `supergroup` | 19 | 22 |
| `synonyms` | 4893 | 4882 |
| `txgnn` | 19 | 22 |
| `unmet_medical_need` | 15545 | 15538 |
