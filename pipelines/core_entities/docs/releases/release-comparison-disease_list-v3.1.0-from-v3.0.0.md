# Release Comparison Report

**New Release:** `v3.1.0-disease_list`

**Base Release:** `v3.0.0-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v3.1.0/03_primary/release/ec-disease-list.parquet`

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
**Total:** 5

**Examples (up to 10):**

| ID | Name |
|----|------|
| `MONDO:0019398` | Desmin-related myopathy with mallory body-like inclusions |
| `MONDO:0971094` | Cardiac anomalies-short stature-joint hypermobility-facial dysmorphism syndrome due to tab2 mutation |
| `MONDO:0011271` | Rigid spine muscular dystrophy 1 |
| `MONDO:0021834` | Akaba hayasaka syndrome |
| `MONDO:0600011` | Mild hypophosphatasia |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `harrisons_view` | 304 |
| `is_infectious_disease` | 1588 |
| `mondo_top_grouping` | 57 |
| `name` | 20 |
| `strategically_viable` | 109 |
| `strategically_viable_assigned_by` | 345 |
| `synonyms` | 145 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0013537` | Autosomal recessive nonsyndromic hearing loss 29 | `auditory_system_disorder\|hereditary_disease\|nervous_system_disorder` | `auditory_system_disorder\|hereditary_disease` |
| `MONDO:0019950` | Congenital muscular dystrophy | `musculoskeletal_system_disorder\|hereditary_disease\|nervous_system_disorder` | `musculoskeletal_system_disorder\|nervous_system_disorder` |
| `MONDO:0001021` | Ametropic amblyopia | `nervous_system_disorder\|disorder_of_visual_system` | `disorder_of_visual_system` |
| `MONDO:0017936` | Benign samaritan congenital myopathy | `musculoskeletal_system_disorder\|hereditary_disease` | `musculoskeletal_system_disorder` |
| `MONDO:0015794` | Antenatal multiminicore disease with arthrogryp... | `hereditary_disease\|nervous_system_disorder` | `musculoskeletal_system_disorder\|hereditary_disease\|nervous_system_disorder` |

#### `is_infectious_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0023143` | Fetal enterovirus syndrome | `True` | `False` |
| `MONDO:0000252` | Inflammatory diarrhea | `True` | `False` |
| `MONDO:0000878` | Cytomegalovirus retinitis | `True` | `False` |
| `MONDO:0015908` | Chromomycosis | `True` | `False` |
| `MONDO:0003765` | Adult leptomeningeal melanoma | `True` | `False` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0019950` | Congenital muscular dystrophy | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0017936` | Benign samaritan congenital myopathy | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0020121` | Muscular dystrophy | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0019952` | Congenital myopathy | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |
| `MONDO:0000686` | Alexia without agraphia | `disease_by_developmental_or_physiological_process\|disease_by_body_system_or_component` | `disease_by_body_system_or_component` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0014732` | `Hypomyelinating leukodystrophy 12` | `Leukodystrophy, hypomyelinating, 12` |
| `MONDO:0014666` | `Hypomyelinating leukodystrophy 11` | `Leukodystrophy, hypomyelinating, 11` |
| `MONDO:0012824` | `Hypomyelinating leukodystrophy 4` | `Leukodystrophy, hypomyelinating, 4` |
| `MONDO:0010829` | `Carasil syndrome` | `Cerebral arteriopathy, autosomal recessive, with subcortical infarcts and leukoencephalopathy 2` |
| `MONDO:0011479` | `Postural orthostatic tachycardia syndrome` | `Postural orthostatic tachycardia syndrome due to net deficiency` |

#### `strategically_viable`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0018975` | Neurofibromatosis type 1 | `False` | `True` |
| `MONDO:0014567` | Glutamate pyruvate transaminase 2 deficiency | `False` | `True` |
| `MONDO:0011871` | Niemann-pick disease type b | `False` | `True` |
| `MONDO:0008641` | Retinal vasculopathy with cerebral leukoencepha... | `False` | `True` |
| `MONDO:0100135` | Dravet syndrome | `False` | `True` |

#### `strategically_viable_assigned_by`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0016512` | Kabuki syndrome | `not assigned` | `human` |
| `MONDO:0019342` | Seckel syndrome | `not assigned` | `human` |
| `MONDO:0009185` | Amelocerebrohypohidrotic syndrome | `not assigned` | `human` |
| `MONDO:0018248` | Intellectual disability-seizures-macrocephaly-o... | `not assigned` | `human` |
| `MONDO:0008999` | Cohen syndrome | `not assigned` | `human` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0014830` | Platelet-type bleeding disorder 20 | `['inherited bleeding disorder, platelet-type caused by mutation in SLFN14'
 'bleeding disorder, p...` | `['inherited bleeding disorder, platelet-type caused by mutation in SLFN14'
 'bleeding disorder, p...` |
| `MONDO:0031332` | Glanzmann thrombasthenia 1 | `['thrombasthenia of Glanzmann and Naegeli' 'thrombasthenia'
 'platelet glycoprotein IIb-IIIa defi...` | `['thrombasthenia of Glanzmann and Naegeli' 'thrombasthenia'
 'platelet glycoprotein IIb-IIIa defi...` |
| `MONDO:0012541` | Deafness with labyrinthine aplasia, microtia, a... | `['microdontia-type I microtia-deafness syndrome'
 'deafness, congenital with inner ear agenesis, ...` | `['microdontia-type I microtia-deafness syndrome'
 'deafness, congenital with inner ear agenesis, ...` |
| `MONDO:0008748` | Hermansky-pudlak syndrome 1 | `['Hermansky-Pudlak syndrome type 1'
 'Hermansky-Pudlak syndrome caused by mutation in HPS1'
 'Her...` | `['reticuloendothelial cells' 'delta storage pool disease'
 'albinism with hemorrhagic diathesis a...` |
| `MONDO:0005761` | Filarial elephantiasis | `['eyelid elephantiasis']` | `['eyelid elephantiasis' 'Lymphatic Filariasis']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22607 | 22602 |
| `anatomical_name` | 22476 | 22471 |
| `benign_malignant` | 22529 | 22524 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 15980 | 15975 |
| `is_glucose_dysfunction` | 15891 | 15886 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 15981 | 15976 |
| `is_psychiatric_disease` | 15997 | 15992 |
| `level` | 1868 | 1868 |
| `mondo_top_grouping` | 27 | 27 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 23232 | 23227 |
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
| `unmet_medical_need` | 15891 | 15886 |
