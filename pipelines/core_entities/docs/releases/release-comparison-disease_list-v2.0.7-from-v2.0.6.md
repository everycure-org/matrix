# Release Comparison Report

**New Release:** `v2.0.7-disease_list`

**Base Release:** `v2.0.6-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v2.0.7/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v2.0.6/ec-disease-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 6

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0958129` | Coq7-related distal hereditary motor neuropathy |
| `MONDO:0005719` | Coronavinae infectious disease |
| `MONDO:0014866` | Charcot-marie-tooth disease axonal type 2t |
| `MONDO:0008284` | Polyposis of gastric fundus without polyposis coli |
| `MONDO:0100025` | Epilepsy of infancy with migrating focal seizures |
| `MONDO:0859244` | Phosphoribosylaminoimidazole carboxylase deficiency |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `harrisons_view` | 16 |
| `mondo_top_grouping` | 15 |
| `name` | 5 |
| `synonyms` | 89 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0957481` | Idiopathic pregnancy-associated osteoporosis | `musculoskeletal_system_disorder\|hereditary_disease\|disorder_of_development_or_morphogenesis\|obste...` | `musculoskeletal_system_disorder\|obstetric_disorder` |
| `MONDO:0000079` | Nephrolithiasis/osteoporosis, hypophosphatemic | `musculoskeletal_system_disorder\|hereditary_disease\|disorder_of_development_or_morphogenesis` | `musculoskeletal_system_disorder\|hereditary_disease` |
| `MONDO:0957874` | Neuronopathy, distal hereditary motor, autosoma... | `hereditary_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder\|disorder_of_development_or_morphogen...` |
| `MONDO:0018315` | X-linked osteoporosis with fractures | `musculoskeletal_system_disorder\|hereditary_disease\|disorder_of_development_or_morphogenesis` | `musculoskeletal_system_disorder` |
| `MONDO:0005298` | Osteoporosis | `musculoskeletal_system_disorder\|hereditary_disease\|disorder_of_development_or_morphogenesis` | `musculoskeletal_system_disorder` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0957481` | Idiopathic pregnancy-associated osteoporosis | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` |
| `MONDO:0000079` | Nephrolithiasis/osteoporosis, hypophosphatemic | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` |
| `MONDO:0957874` | Neuronopathy, distal hereditary motor, autosoma... | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |
| `MONDO:0018315` | X-linked osteoporosis with fractures | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` | `disease_by_body_system_or_component` |
| `MONDO:0005298` | Osteoporosis | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` | `disease_by_body_system_or_component` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0007100` | `Familial amyloid neuropathy` | `Hereditary transthyretin amyloidosis` |
| `MONDO:0014252` | `Familial hypobetalipoproteinemia 1` | `Familial defective apolipoprotein B-100` |
| `MONDO:0020040` | `46,xy disorder of sex development` | `46 xy differences of sex development` |
| `MONDO:0100058` | `Hypervalinemia and hyperleucine-isoleucinemia` | `Branched-chain aminotransferase deficiency` |
| `MONDO:0012611` | `Polyhydramnios, megalencephaly, and symptomatic epilepsy` | `PMSE syndrome` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0016057` | Isolated encephalocele | `['Encephalocele']` | `[]` |
| `MONDO:0005103` | Well-differentiated liposarcoma | `['well-differentiated liposarcoma'
 'well differentiated liposarcoma of deep soft tissue'
 'well ...` | `['well-differentiated liposarcoma' 'WDLS']` |
| `MONDO:0009948` | Pyropoikilocytosis, hereditary | `['pyropoikilocytosis, hereditary' 'pyropoikilocytosis' 'Hypophosphatasia']` | `['pyropoikilocytosis, hereditary' 'pyropoikilocytosis']` |
| `MONDO:0008056` | Myotonic dystrophy type 1 | `['myotonic dystrophy type 1'
 'myotonic dystrophy caused by mutation in DMPK' 'dystrophia myotoni...` | `['myotonic dystrophy type 1'
 'myotonic dystrophy caused by mutation in DMPK' 'dystrophia myotoni...` |
| `MONDO:0005100` | Systemic sclerosis | `['systemic sclerosis' 'systemic scleroderma' 'systemic Scleroderma'
 'diffuse sclerosis' 'diffuse...` | `['systemic sclerosis' 'systemic scleroderma' 'systemic Scleroderma'
 'Systemic Scleroderma' 'Scle...` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22113 | 22107 |
| `anatomical_name` | 21982 | 21976 |
| `benign_malignant` | 22049 | 22043 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 114 | 114 |
| `is_glucose_dysfunction` | 24 | 24 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 114 | 114 |
| `is_psychiatric_disease` | 114 | 114 |
| `level` | 1869 | 1868 |
| `mondo_top_grouping` | 24 | 24 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22738 | 22732 |
| `precancerous` | 22 | 22 |
| `prevalence_experimental` | 15658 | 15653 |
| `prevalence_world` | 15523 | 15518 |
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
| `txgnn` | 24 | 24 |
| `unmet_medical_need` | 15523 | 15518 |
