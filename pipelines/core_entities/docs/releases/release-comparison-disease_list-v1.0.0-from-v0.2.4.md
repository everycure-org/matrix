# Release Comparison Report

**New Release:** `v1.0.0-disease_list`

**Base Release:** `v0.2.4-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.0.0/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v0.2.4/ec-disease-list.parquet`

## Column Changes

### Added Columns
- `deleted`
- `is_benign_tumour`
- `is_malignant_cancer`
- `is_psychiatric_disease`
- `prevalence_experimental`
- `prevalence_world`

### Removed Columns
- `obsolete`

## Row Changes

### Added Rows
**Total:** 0


### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `anatomical_id` | 19 |
| `anatomical_name` | 19 |
| `benign_malignant` | 19 |
| `core` | 5 |
| `harrisons_view` | 53 |
| `level` | 24 |
| `mondo_top_grouping` | 63 |
| `mondo_txgnn` | 19 |
| `new_id` | 22763 |
| `supergroup` | 19 |
| `synonyms` | 4893 |
| `txgnn` | 19 |
| `unmet_medical_need` | 15545 |

### Examples by Column

*Up to 5 examples per column*

#### `anatomical_id`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `UMLS:C1334669` | `` | `*None*` |
| `UMLS:C1334668` | `` | `*None*` |
| `UMLS:C1334676` | `` | `*None*` |

#### `anatomical_name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `UMLS:C1334669` | `` | `*None*` |
| `UMLS:C1334668` | `` | `*None*` |
| `UMLS:C1334676` | `` | `*None*` |

#### `benign_malignant`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `UMLS:C1334669` | `` | `*None*` |
| `UMLS:C1334668` | `` | `*None*` |
| `UMLS:C1334676` | `` | `*None*` |

#### `core`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0009756` | `False` | `True` |
| `MONDO:0011871` | `False` | `True` |
| `MONDO:0018982` | `False` | `True` |
| `MONDO:0004369` | `True` | `False` |
| `MONDO:0001982` | `True` | `False` |

#### `harrisons_view`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0005151` | `` | `*None*` |
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `MONDO:0024458` | `` | `*None*` |
| `MONDO:0002118` | `` | `*None*` |

#### `level`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0009756` | `subgroup` | `clinically_recognized` |
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `UMLS:C1334669` | `` | `*None*` |
| `MONDO:0011871` | `subgroup` | `clinically_recognized` |

#### `mondo_top_grouping`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0005151` | `` | `*None*` |
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `MONDO:0024458` | `` | `*None*` |
| `MONDO:0100366` | `` | `*None*` |

#### `mondo_txgnn`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `UMLS:C1334669` | `` | `*None*` |
| `UMLS:C1334668` | `` | `*None*` |
| `UMLS:C1334676` | `` | `*None*` |

#### `new_id`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0019857` | `` | `*None*` |
| `MONDO:0010207` | `` | `*None*` |
| `MONDO:0009550` | `` | `*None*` |
| `MONDO:0018202` | `` | `*None*` |
| `MONDO:0056803` | `` | `*None*` |

#### `supergroup`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `UMLS:C1334669` | `` | `*None*` |
| `UMLS:C1334668` | `` | `*None*` |
| `UMLS:C1334676` | `` | `*None*` |

#### `synonyms`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0019857` | `` | `*None*` |
| `MONDO:0018202` | `` | `*None*` |
| `MONDO:0859301` | `` | `*None*` |
| `MONDO:0859231` | `` | `*None*` |
| `MONDO:0005965` | `` | `*None*` |

#### `txgnn`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `UMLS:C1334672` | `` | `*None*` |
| `MONDO:0000412` | `` | `*None*` |
| `UMLS:C1334669` | `` | `*None*` |
| `UMLS:C1334668` | `` | `*None*` |
| `UMLS:C1334676` | `` | `*None*` |

#### `unmet_medical_need`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0019857` | `` | `*None*` |
| `MONDO:0009550` | `` | `*None*` |
| `MONDO:0018202` | `` | `*None*` |
| `MONDO:0003889` | `` | `*None*` |
| `MONDO:0002405` | `` | `*None*` |