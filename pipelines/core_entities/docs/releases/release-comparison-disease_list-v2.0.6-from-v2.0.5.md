# Release Comparison Report

**New Release:** `v2.0.6-disease_list`

**Base Release:** `v2.0.5-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v2.0.6/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v2.0.5/ec-disease-list.parquet`

## Column Changes

### Added Columns
- `strategically_viable_assigned_by`

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
| `strategically_viable` | 19 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `strategically_viable`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `MONDO:0009796` | Ornithine aminotransferase deficiency | `False` | `True` |
| `MONDO:0014252` | Familial hypobetalipoproteinemia 1 | `False` | `True` |
| `MONDO:0013178` | Congenital muscular dystrophy due to lmna mutation | `False` | `True` |
| `MONDO:0100058` | Hypervalinemia and hyperleucine-isoleucinemia | `False` | `True` |
| `MONDO:0019169` | Pyruvate dehydrogenase deficiency | `False` | `True` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `anatomical_deformity` | 22 | 22 |
| `anatomical_id` | 22113 | 22113 |
| `anatomical_name` | 21982 | 21982 |
| `benign_malignant` | 22049 | 22049 |
| `core` | 22 | 22 |
| `deleted` | 0 | 0 |
| `harrisons_view` | 55 | 55 |
| `id` | 0 | 0 |
| `is_benign_tumour` | 114 | 114 |
| `is_glucose_dysfunction` | 24 | 24 |
| `is_infectious_disease` | 22 | 22 |
| `is_malignant_cancer` | 114 | 114 |
| `is_psychiatric_disease` | 114 | 114 |
| `level` | 1869 | 1869 |
| `mondo_top_grouping` | 24 | 24 |
| `mondo_txgnn` | 22 | 22 |
| `name` | 0 | 0 |
| `new_id` | 22738 | 22738 |
| `precancerous` | 22 | 22 |
| `prevalence_experimental` | 15658 | 15658 |
| `prevalence_world` | 15523 | 15523 |
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
| `strategically_viable_assigned_by` | N/A | 22 |
| `supergroup` | 22 | 22 |
| `synonyms` | 22 | 22 |
| `txgnn` | 24 | 24 |
| `unmet_medical_need` | 15523 | 15523 |
