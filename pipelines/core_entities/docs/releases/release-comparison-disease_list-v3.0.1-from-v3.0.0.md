# Release Comparison Report

**New Release:** `v3.0.1-disease_list`

**Base Release:** `v3.0.0-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v3.0.1/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v3.0.0/ec-disease-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
- `prevalence_experimental`
- `prevalence_world`

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 3

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0600011` | Mild hypophosphatasia |
| `MONDO:0021834` | Akaba hayasaka syndrome |
| `MONDO:0971094` | Cardiac anomalies-short stature-joint hypermobility-facial dysmorphism syndrome due to tab2 mutation |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `harrisons_view` | 259 |
| `is_infectious_disease` | 1588 |
| `mondo_top_grouping` | 18 |
| `name` | 15 |
| `strategically_viable` | 108 |
| `strategically_viable_assigned_by` | 345 |
| `synonyms` | 43 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0033304` | Nonsyndromic deafness, y-linked | `auditory_system_disorder\|hereditary_disease\|nervous_system_disorder` | `auditory_system_disorder\|hereditary_disease` |
| `MONDO:0032917` | Hearing loss, autosomal dominant 76 | `auditory_system_disorder\|hereditary_disease\|nervous_system_disorder` | `auditory_system_disorder\|hereditary_disease` |
| `MONDO:0957825` | Hearing loss, autosomal recessive 121 | `auditory_system_disorder\|hereditary_disease\|nervous_system_disorder` | `auditory_system_disorder\|hereditary_disease` |
| `MONDO:0033259` | Hearing loss, autosomal dominant 72 | `auditory_system_disorder\|hereditary_disease\|nervous_system_disorder` | `auditory_system_disorder\|hereditary_disease` |
| `MONDO:0001729` | Active cochlear meniere disease | `auditory_system_disorder\|hereditary_disease\|nervous_system_disorder\|otorhinolaryngologic_disease` | `auditory_system_disorder\|hereditary_disease\|otorhinolaryngologic_disease` |

#### `is_infectious_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0023147` | Fetal parainfluenza virus type 3 syndrome | `True` | `False` |
| `MONDO:0043836` | Tuberculosis, spinal | `True` | `False` |
| `MONDO:0005724` | Cryptococcosis | `True` | `False` |
| `MONDO:0600003` | Bacterial hemorrhagic fever | `True` | `False` |
| `MONDO:0957790` | Immune dysregulation, autoimmunity, and autoinf... | `True` | `False` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0000669` | Color agnosia | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0000660` | Akinetopsia | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0000685` | Visual agnosia | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0000675` | Pain agnosia | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0000683` | Topographical agnosia | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0012824` | `Hypomyelinating leukodystrophy 4` | `Leukodystrophy, hypomyelinating, 4` |
| `MONDO:0014732` | `Hypomyelinating leukodystrophy 12` | `Leukodystrophy, hypomyelinating, 12` |
| `MONDO:0014506` | `Hypomyelinating leukodystrophy 9` | `Leukodystrophy, hypomyelinating, 9` |
| `MONDO:0030714` | `Osteogenesis imperfecta, iia 22` | `Osteogenesis imperfecta, type xxii` |
| `MONDO:0008006` | `Mobius syndrome` | `Moebius syndrome` |

#### `strategically_viable`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007417` | Darier disease | `False` | `True` |
| `MONDO:0008788` | Irida syndrome | `False` | `True` |
| `MONDO:0100339` | Friedreich ataxia | `False` | `True` |
| `MONDO:0006412` | Rosai-Dorfman disease | `False` | `True` |
| `MONDO:0012276` | Generalized epilepsy-paroxysmal dyskinesia synd... | `False` | `True` |

#### `strategically_viable_assigned_by`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0032943` | Neurodevelopmental disorder with relative macro... | `not assigned` | `human` |
| `MONDO:0007417` | Darier disease | `not assigned` | `human` |
| `MONDO:0010428` | Chromosome xp11.23-p11.22 duplication syndrome | `not assigned` | `human` |
| `MONDO:0007863` | Kleine-levin syndrome | `not assigned` | `human` |
| `MONDO:0020466` | Monosomy x | `not assigned` | `human` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0012758` | Prostate cancer, hereditary, 13 | `['prostate cancer, hereditary, type 13' 'prostate cancer, hereditary, 13'
 'familial prostate can...` | `['prostate cancer, hereditary, type 13' 'prostate cancer, hereditary, 13']` |
| `MONDO:0012824` | Leukodystrophy, hypomyelinating, 4 | `['mitochondrial HSP60 chaperonopathy'
 'leukodystrophy, hypomyelinating, type 4'
 'leukodystrophy...` | `['mitochondrial HSP60 chaperonopathy'
 'leukodystrophy, hypomyelinating, type 4'
 'leukodystrophy...` |
| `MONDO:0006412` | Rosai-Dorfman disease | `['sinus histiocytosis with massive lymphadenopathy' 'SHML'
 'Rosaï-Dorfman-Destombes disease' 'Ro...` | `['sinus histiocytosis with massive lymphadenopathy' 'SHML'
 'Rosaï-Dorfman-Destombes disease' 'Ro...` |
| `MONDO:0032730` | Leukodystrophy, hypomyelinating, 18 | `[]` | `['hypomyelinating leukodystrophy 18' 'HLD18'
 'DEGS1-related hypomyelinating leukodystrophy' 'DEG...` |
| `MONDO:0011136` | Quebec platelet disorder | `['factor V Quebec' 'Quebec platelet disorder' 'BDPLT5']` | `['factor V Quebec' 'Quebec platelet disorder' 'QPD' 'BDPLT5']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22607 | 22604 |
| `anatomical_name` | 22476 | 22473 |
| `benign_malignant` | 22529 | 22526 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 15980 | 15977 |
| `is_glucose_dysfunction` | 15891 | 15888 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 15981 | 15978 |
| `is_psychiatric_disease` | 15997 | 15994 |
| `level` | 1868 | 1868 |
| `mondo_top_grouping` | 27 | 27 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 23232 | 23229 |
| `precancerous` | 22 | 22 |
| `prevalence_experimental` | 16153 | N/A |
| `prevalence_world` | 16018 | N/A |
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
| `txgnn` | 425 | 425 |
| `unmet_medical_need` | 15891 | 15888 |
