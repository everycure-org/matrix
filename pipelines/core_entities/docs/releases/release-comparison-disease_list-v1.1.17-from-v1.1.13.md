# Release Comparison Report

**New Release:** `v1.1.17-disease_list`

**Base Release:** `v1.1.13-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.1.17/03_primary/release/ec-disease-list.parquet`

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
| `MONDO:0035660` | Gnao1-related developmental delay-seizures-movement disorder spectrum |
| `MONDO:0023243` | Glass-chapman-hockley syndrome |
| `MONDO:0004914` | Celiac artery stenosis from compression by median arcuate ligament of diaphragm |
| `MONDO:0010553` | Charcot-marie-tooth peroneal muscular atrophy and friedreich ataxia, combined |
| `MONDO:0013935` | Usher syndrome type 1j |
| `MONDO:0035774` | Nrxn1-related severe neurodevelopmental disorder-motor stereotypies-chronic constipation-sleep-wake cycle disturbance |
| `MONDO:0700044` | Tubb2a-related tubulinopathy |
| `MONDO:0035661` | Traf7-associated heart defect-digital anomalies-facial dysmorphism-motor and speech delay syndrome |
| `MONDO:0035529` | Infantile-onset pulmonary alveolar proteinosis-hypogammaglobulinemia |
| `MONDO:0034820` | Cleft lip and palate-craniofacial dysmorphism-congenital heart defect-hearing loss syndrome |

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
| `name` | 15 |
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
| `MONDO:0003469` | Epithelioid cell synovial sarcoma | `cancer_or_benign_tumor` | `hereditary_disease\|cancer_or_benign_tumor` |
| `MONDO:0044113` | Bullous systemic lupus erythematosus | `connective_tissue_disorder\|immune_system_disorder` | `hereditary_disease\|connective_tissue_disorder\|immune_system_disorder` |
| `MONDO:0005067` | Monophasic synovial sarcoma | `cancer_or_benign_tumor` | `hereditary_disease\|cancer_or_benign_tumor` |
| `MONDO:0010434` | Synovial sarcoma | `cancer_or_benign_tumor` | `hereditary_disease\|cancer_or_benign_tumor` |
| `MONDO:0013690` | Pitt-hopkins-like syndrome 2 | `hereditary_disease\|disorder_of_development_or_morphogenesis` | `hereditary_disease\|nervous_system_disorder\|disorder_of_development_or_morphogenesis` |

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
| `MONDO:0010725` | X-linked retinoschisis | `disorder_of_visual_system\|nervous_system_disorder\|hereditary_disease\|disorder_of_orbital_region` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0010869` | Motor neuron disease with dementia and ophthalm... | `nervous_system_disorder\|hereditary_disease` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0014669` | Cone-rod dystrophy 21 | `disorder_of_visual_system\|nervous_system_disorder\|hereditary_disease\|disorder_of_orbital_region` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0015447` | Differentiated thyroid carcinoma | `cancer_or_benign_tumor\|endocrine_system_disorder` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0017387` | Epithelioid sarcoma | `cancer_or_benign_tumor` | `disease_by_etiologic_mechanism` |

#### `mondo_txgnn`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020300` | obsolete autosomal dominant nocturnal frontal l... | `other` | `*None*` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0008340` | `Ptosis, hereditary congenital, 1` | `Congenital ptosis` |
| `MONDO:0014877` | `Myopathy, distal, 5` | `Adenylosuccinate Synthetase 1 (ADSS1) Myopathy` |
| `MONDO:0023124` | `Familial pulmonary arterial hypertension leucopenia and atrial septal defect` | `Dursun syndrome` |
| `MONDO:0011236` | `Hyperinsulinism due to glucokinase deficiency` | `Hyperinsulinemic hypoglycemia, familial, 3` |
| `MONDO:0013773` | `Porencephaly 2` | `Brain small vessel disease 2a, autosomal dominant` |

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
| `MONDO:0004620` | Hodgkin's lymphoma, lymphocytic depletion | `lymphocyte-depleted classical Hodgkin lymphoma; classic Hodgkin lymphoma, lymphocyte-depleted typ...` | `lymphocyte-depleted classical Hodgkin lymphoma; classic Hodgkin lymphoma, lymphocyte-depleted typ...` |
| `MONDO:0007572` | Primary familial polycythemia due to epo recept... | `primary familial and congenital polycythemia; primary congenital erythrocytosis; familial polycyt...` | `primary familial and congenital polycythemia; primary congenital erythrocytosis; familial polycyt...` |
| `MONDO:0005459` | Human african trypanosomiasis | `sleeping sickness; African trypanosomiasis; African sleeping sickness; Africam sleeping sickness` | `sleeping sickness; African trypanosomiasis; Africam sleeping sickness` |
| `MONDO:0007667` | Subependymoma | `subependymoma; subependymal glioma; subependymal astrocytoma; mixed subependymoma-ependymoma; WHO...` | `subependymoma; subependymal glioma; subependymal astrocytoma; WHO grade I ependymal tumour; WHO g...` |
| `MONDO:0019499` | Turner syndrome | `monosomy X syndrome; karyotype 45, X; gonadal dysgenesis - Turner; XO syndrome; 45X syndrome; 45,...` | `karyotype 45, X; 45X syndrome; 45,X0 syndrome; 45,X/46,XX syndrome; 45,X syndrome; 45,X gonadal d...` |

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
