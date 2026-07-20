# Release Comparison Report

**New Release:** `v2.0.13-disease_list`

**Base Release:** `v2.0.11-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v2.0.13/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v2.0.11/ec-disease-list.parquet`

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
| `MONDO:0016061` | Immunodeficiency with factor h anomaly |

## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `harrisons_view` | 7 |
| `mondo_top_grouping` | 3 |
| `name` | 1 |
| `synonyms` | 30 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `harrisons_view`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0020741` | Pyridoxine-dependent epilepsy caused by aldh7a1... | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder\|nutritional_disorder` |
| `MONDO:0958106` | Congenital insensitivity to pain syndrome, mars... | `syndromic_disease\|hereditary_disease` | `syndromic_disease\|hereditary_disease\|nervous_system_disorder` |
| `MONDO:0009459` | Channelopathy-associated congenital insensitivi... | `syndromic_disease\|hereditary_disease` | `syndromic_disease\|hereditary_disease\|nervous_system_disorder` |
| `MONDO:0019237` | Inborn disorder of pyridoxine metabolism | `hereditary_disease\|metabolic_disease` | `hereditary_disease\|metabolic_disease\|nutritional_disorder` |
| `MONDO:0009945` | Pyridoxine-dependent epilepsy | `hereditary_disease\|metabolic_disease\|nervous_system_disorder` | `hereditary_disease\|metabolic_disease\|nervous_system_disorder\|nutritional_disorder` |

#### `mondo_top_grouping`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0100000` | Med12-related intellectual disability syndrome | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |
| `MONDO:0010590` | Fg syndrome 1 | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |
| `MONDO:0032485` | Intellectual developmental disorder 61 | `disease_by_etiologic_mechanism\|disease_by_body_system_or_component` | `disease_by_etiologic_mechanism\|disease_by_developmental_or_physiological_process\|disease_by_body_...` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0011010` | `Matthew-wood syndrome` | `Microphthalmia, syndromic 9` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0015912` | Macrothrombocytopenia and granulocyte inclusion... | `['macrothrombocytopenia progressive deafness'
 'macrothrombocytopenia and progressive sensorineur...` | `['macrothrombocytopenia progressive deafness'
 'macrothrombocytopenia and progressive sensorineur...` |
| `MONDO:0014536` | Thrombocytopenia 5 | `['thrombocytopenia type 5' 'thrombocytopenia caused by mutation in ETV6'
 'thrombocytopenia 5' 'E...` | `['thrombocytopenia caused by mutation in ETV6' 'thrombocytopenia 5'
 'ETV6-related thrombocytopen...` |
| `MONDO:0012350` | Complement factor h deficiency | `['complement factor H deficiency']` | `['immunodeficiency with factor H anomaly' 'factor H deficiency'
 'complement factor H deficiency'...` |
| `MONDO:0005892` | Otitis media with effusion | `['serous otitis Media' 'secretory otitis Media' 'OME']` | `['OME']` |
| `MONDO:0859307` | Cleidocranial dysplasia 2 | `[]` | `['CCD2' 'CBFB-related cleidocranial dysplasia']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22107 | 22106 |
| `anatomical_name` | 21976 | 21975 |
| `benign_malignant` | 22043 | 22042 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 114 | 114 |
| `is_glucose_dysfunction` | 24 | 24 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 114 | 114 |
| `is_psychiatric_disease` | 114 | 114 |
| `level` | 1868 | 1868 |
| `mondo_top_grouping` | 24 | 24 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22732 | 22731 |
| `precancerous` | 22 | 22 |
| `prevalence_experimental` | 15653 | 15652 |
| `prevalence_world` | 15518 | 15517 |
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
| `unmet_medical_need` | 15518 | 15517 |
