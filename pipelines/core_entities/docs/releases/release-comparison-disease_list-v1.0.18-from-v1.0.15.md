# Release Comparison Report

**New Release:** `v1.0.18-disease_list`

**Base Release:** `v1.0.15-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.0.18/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.0.15/ec-disease-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `is_benign_tumour` | 15718 |
| `is_glucose_dysfunction` | 15585 |
| `is_infectious_disease` | 1927 |
| `is_malignant_cancer` | 15605 |
| `is_psychiatric_disease` | 16659 |

### Examples by Column

*Up to 5 examples per column*

#### `is_benign_tumour`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0010273` | Lymphoma, hodgkin, x-linked pseudoautosomal | `False` | `*None*` |
| `MONDO:0007456` | Diarrhea, glucose-stimulated secretory, with co... | `False` | `*None*` |
| `MONDO:0004199` | Vulvar keratinizing squamous cell carcinoma | `False` | `*None*` |
| `MONDO:0008515` | Syndactyly type 4 | `False` | `*None*` |
| `MONDO:0010427` | Syndromic x-linked intellectual disability raym... | `False` | `*None*` |

#### `is_glucose_dysfunction`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0010273` | Lymphoma, hodgkin, x-linked pseudoautosomal | `False` | `*None*` |
| `MONDO:0007456` | Diarrhea, glucose-stimulated secretory, with co... | `False` | `*None*` |
| `MONDO:0004199` | Vulvar keratinizing squamous cell carcinoma | `False` | `*None*` |
| `MONDO:0008515` | Syndactyly type 4 | `False` | `*None*` |
| `MONDO:0010427` | Syndromic x-linked intellectual disability raym... | `False` | `*None*` |

#### `is_infectious_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0007456` | Diarrhea, glucose-stimulated secretory, with co... | `True` | `False` |
| `MONDO:0005135` | Parasitic infectious disease | `True` | `False` |
| `MONDO:0017342` | Epstein-barr virus-related tumor | `True` | `False` |
| `MONDO:0024777` | Immunodeficiency 98 with autoinflammation, x-li... | `True` | `False` |
| `MONDO:0009951` | Radiculoneuropathy, fatal neonatal | `True` | `False` |

#### `is_malignant_cancer`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0010273` | Lymphoma, hodgkin, x-linked pseudoautosomal | `True` | `*None*` |
| `MONDO:0007456` | Diarrhea, glucose-stimulated secretory, with co... | `False` | `*None*` |
| `MONDO:0004199` | Vulvar keratinizing squamous cell carcinoma | `True` | `*None*` |
| `MONDO:0008515` | Syndactyly type 4 | `False` | `*None*` |
| `MONDO:0010427` | Syndromic x-linked intellectual disability raym... | `False` | `*None*` |

#### `is_psychiatric_disease`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0010273` | Lymphoma, hodgkin, x-linked pseudoautosomal | `False` | `*None*` |
| `MONDO:0007456` | Diarrhea, glucose-stimulated secretory, with co... | `False` | `*None*` |
| `MONDO:0004199` | Vulvar keratinizing squamous cell carcinoma | `False` | `*None*` |
| `MONDO:0008515` | Syndactyly type 4 | `False` | `*None*` |
| `MONDO:0010427` | Syndromic x-linked intellectual disability raym... | `True` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 19 | 19 |
| `anatomical_id` | 22143 | 22143 |
| `anatomical_name` | 22012 | 22012 |
| `benign_malignant` | 22079 | 22079 |
| `core` | 19 | 19 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 53 | 53 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 110 | 15544 |
| `is_glucose_dysfunction` | 19 | 15544 |
| `is_infectious_disease` | 19 | 19 |
| `is_malignant_cancer` | 110 | 15544 |
| `is_psychiatric_disease` | 110 | 15544 |
| `level` | 1871 | 1871 |
| `mondo_top_grouping` | 63 | 63 |
| `mondo_txgnn` | 19 | 19 |
| `name` | 0 | 0 |
| `new_id` | 22770 | 22770 |
| `precancerous` | 19 | 19 |
| `prevalence_experimental` | 15680 | 15680 |
| `prevalence_world` | 15545 | 15545 |
| `speciality_breast` | 19 | 19 |
| `speciality_cardiovascular` | 19 | 19 |
| `speciality_chromosomal` | 19 | 19 |
| `speciality_connective_tissue` | 19 | 19 |
| `speciality_dermatologic` | 19 | 19 |
| `speciality_ear_nose_throat` | 19 | 19 |
| `speciality_endocrine` | 19 | 19 |
| `speciality_eye_and_adnexa` | 19 | 19 |
| `speciality_gastrointestinal` | 19 | 19 |
| `speciality_hematologic` | 19 | 19 |
| `speciality_immune` | 19 | 19 |
| `speciality_infection` | 19 | 19 |
| `speciality_metabolic` | 19 | 19 |
| `speciality_musculoskeletal` | 19 | 19 |
| `speciality_neoplasm` | 19 | 19 |
| `speciality_neurological` | 19 | 19 |
| `speciality_obstetric` | 19 | 19 |
| `speciality_poisoning_and_toxicity` | 19 | 19 |
| `speciality_psychiatric` | 19 | 19 |
| `speciality_renal_and_urinary` | 19 | 19 |
| `speciality_reproductive` | 19 | 19 |
| `speciality_respiratory` | 19 | 19 |
| `speciality_syndromic` | 19 | 19 |
| `supergroup` | 19 | 19 |
| `synonyms` | 4893 | 4893 |
| `txgnn` | 19 | 19 |
| `unmet_medical_need` | 15545 | 15545 |
