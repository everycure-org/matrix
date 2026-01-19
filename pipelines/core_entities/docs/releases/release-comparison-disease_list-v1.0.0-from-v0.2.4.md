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

## Commits

51 commits between 22/10/2025 and 07/11/2025 from the following authors: Jacques Vergine, Li-Vern Teo, everycure

- [daff2131](https://github.com/everycure-org/core-entities/commit/daff213161885f89137c190b487fd66e486dedff): Replace disease obsolete by deleted & handle None (#100) (Jacques Vergine)
- [93dddc7b](https://github.com/everycure-org/core-entities/commit/93dddc7b06cf527a12ba97bc14dfa0a5cd7740b9): Add disease categories to release (#98) (Jacques Vergine)
- [bff5ef49](https://github.com/everycure-org/core-entities/commit/bff5ef49370deef3ee53008b2060423bba70ef8f): Include disease prevalence in release (#97) (Jacques Vergine)
- [34843343](https://github.com/everycure-org/core-entities/commit/348433432b9ac1eb4f81a2af9e9caab3f84afb50): Update TSV to parquet for disease pipelines (#96) (Jacques Vergine)
- [51520103](https://github.com/everycure-org/core-entities/commit/515201037269d4fd1f8c4d6a9c25d2fdaf68f8eb): Update readme (#94) (Jacques Vergine)
- [39a11a33](https://github.com/everycure-org/core-entities/commit/39a11a335dce0da6b74810598d73df92314f21c9): Update tags from drug to drug_list and disease to disease_list (Jacques Vergine)
- [56f662b9](https://github.com/everycure-org/core-entities/commit/56f662b9d8206859222b1751bcc6c277abfdb7b9): Update release version (Jacques Vergine)
- [48dd5238](https://github.com/everycure-org/core-entities/commit/48dd5238a6f2847d35edc1e77fec0a8c8d5ff3e3): Change run_llm_pipeline to run drug and disease list pipelines (Jacques Vergine)
- [8829814d](https://github.com/everycure-org/core-entities/commit/8829814d2321fc50ac2d849863d88e8e09fad1e7): Revert "Merge branch 'main' of github.com:everycure-org/core-entities" (Jacques Vergine)
- [2c8122a3](https://github.com/everycure-org/core-entities/commit/2c8122a3b88046da4f7e389dcff3b0fb298cdbed): Add author next to commit (Jacques Vergine)
- [fec0fc36](https://github.com/everycure-org/core-entities/commit/fec0fc363f12dd1d7371951164950118592e91d0): Release v0.2.1 notes for drug (#93) (everycure)
- [bcc5fad7](https://github.com/everycure-org/core-entities/commit/bcc5fad74db46da5f413dfc8eadb9dc14df1b6b7): Fetch tags when creating release PR (Jacques Vergine)
- [eeb81786](https://github.com/everycure-org/core-entities/commit/eeb817860c1b7298219bcf5d428ec2b8126a059e): Not run CI on draft PRs (Jacques Vergine)
- [038e792d](https://github.com/everycure-org/core-entities/commit/038e792d29110a569a2dff3852b770a8cbab21f7): Fix typo in create release PR (Jacques Vergine)
- [52e833ce](https://github.com/everycure-org/core-entities/commit/52e833ce3d1f24d65c94eb1be2dd80610087b106): Add post PR action (Jacques Vergine)
- [f5222f28](https://github.com/everycure-org/core-entities/commit/f5222f289983ae83f7d98fa7345a97b93a4ffa45): Move release notes to specific folder (Jacques Vergine)
- [a5d0aad4](https://github.com/everycure-org/core-entities/commit/a5d0aad4c4e2b4431a07286ec18d32dd2c926f79): Trigger create release pr after release dev (Jacques Vergine)
- [32781928](https://github.com/everycure-org/core-entities/commit/327819284d481edd903ac924c77883ed77d44dc9): Add \ after draft (Jacques Vergine)
- [d50fc80e](https://github.com/everycure-org/core-entities/commit/d50fc80e40ab03607b82e709fe7e62e06925933c): Install gh (Jacques Vergine)
- [a7776064](https://github.com/everycure-org/core-entities/commit/a7776064bad9e89526872f34928510102e78560b): Configure git releasebot (Jacques Vergine)
- [5f1a1a7c](https://github.com/everycure-org/core-entities/commit/5f1a1a7c011cde5b57d209c5723c60555a42bafc): Typo gs:// (Jacques Vergine)
- [6ccca23a](https://github.com/everycure-org/core-entities/commit/6ccca23a3ba48a02cb85056656948a459ed31b0e): Fix branch out step (Jacques Vergine)
- [cc3827bb](https://github.com/everycure-org/core-entities/commit/cc3827bb57a8f7080157e6395b94ce6782e6272f): Compare releases and create release PR (Jacques Vergine)
- [63973ca1](https://github.com/everycure-org/core-entities/commit/63973ca13636fcad4e098ec011cff5288b4b4f6f): Checkout, branch out and get latest public pipeline release (Jacques Vergine)
- [8c845723](https://github.com/everycure-org/core-entities/commit/8c845723ebcaf57704f80d31c295955ca1cf17ca): Authenticate to google and setup dev env (Jacques Vergine)
- [37f16ba8](https://github.com/everycure-org/core-entities/commit/37f16ba861559ad089b0b28a257f337ec2012a2d): Remove ignore errors on gcloud storage ls (Jacques Vergine)
- [c0391eae](https://github.com/everycure-org/core-entities/commit/c0391eaef1c20906bdd3f24be8e205788489814f): Improve release process and documentation for drug and disease (#88) (Jacques Vergine)
- [801c52bb](https://github.com/everycure-org/core-entities/commit/801c52bb357508391b74b8e60e38689fc66c35b0): Add action to create release PR with dummy steps (Jacques Vergine)
- [04a307b9](https://github.com/everycure-org/core-entities/commit/04a307b99a383a9b1358d8f4bd63401bdd7a3345): Include additional columns in drug list (aggregated_with, is_analgesic,is_cardiovascular) (#87) (Jacques Vergine)
- [a9aa9a96](https://github.com/everycure-org/core-entities/commit/a9aa9a967e8cacab9dd43234834b0f6ba66b0cd5): Update NN (#86) (Jacques Vergine)
- [bae80c5f](https://github.com/everycure-org/core-entities/commit/bae80c5f14e3ed73bfd96b5c0c494276c61146fe): EC drug list pipeline (#83) (Jacques Vergine)
- [457ac624](https://github.com/everycure-org/core-entities/commit/457ac624154a7eeff0ce7fb295cdb15437dc48c5): Update run llm action (Jacques Vergine)
- [0e68e08d](https://github.com/everycure-org/core-entities/commit/0e68e08d40a2ff551d187df471ef70bc0130e83b): Remove debug (Jacques Vergine)
- [424fc483](https://github.com/everycure-org/core-entities/commit/424fc4831daed08efb4720a6177f72f4e2523820): Install gcloud (Jacques Vergine)
- [cd81b233](https://github.com/everycure-org/core-entities/commit/cd81b2333d3824e57c18a0e118d7fa38c67a84ee): Remove " (Jacques Vergine)
- [293d9649](https://github.com/everycure-org/core-entities/commit/293d96499417b3d42eaa700242164885671b14d7): Add echo (Jacques Vergine)
- [da3d27b7](https://github.com/everycure-org/core-entities/commit/da3d27b713427f5b1f46ca14a178b25f9a80517a): Don't suppress stderr (Jacques Vergine)
- [bd4e6b60](https://github.com/everycure-org/core-entities/commit/bd4e6b60223bf6cf57a7edafef80b1f1d4d762f8): Remove curly braces around BUCKET_PATH (Jacques Vergine)
- [9f5c9582](https://github.com/everycure-org/core-entities/commit/9f5c9582d860a9f0bbf5c0e74f068750e2069655): Rename dummy action (Jacques Vergine)
- [4a7ae775](https://github.com/everycure-org/core-entities/commit/4a7ae77561f433347c36db07eb8a4a337dc5c34d): Add dummy action to test version bump (Jacques Vergine)
- [6e04e462](https://github.com/everycure-org/core-entities/commit/6e04e4623e59b371c7177136dd72c796a6ee12a4): Revert "Improve run llm actions versioning parameter (#85)" (Jacques Vergine)
- [b0dacbbf](https://github.com/everycure-org/core-entities/commit/b0dacbbf4b17650ab00cade26644b272a2801e15): Revert "Update base LLM version" (Jacques Vergine)
- [46fb8aa2](https://github.com/everycure-org/core-entities/commit/46fb8aa250a7ba17c8adf6900c2a897dfc560773): Update base LLM version (Jacques Vergine)
- [0d7f97cd](https://github.com/everycure-org/core-entities/commit/0d7f97cd6b974b5b2acf44dad350239d98861491): Improve run llm actions versioning parameter (#85) (Jacques Vergine)
- [dfea8dae](https://github.com/everycure-org/core-entities/commit/dfea8dae9200a727d53e7c20a279d79ed215241c): Revert "Add version bump type to run llm pipeline (#84)" (Jacques Vergine)
- [fafd369d](https://github.com/everycure-org/core-entities/commit/fafd369d6f436632b68686bd794fec503f335d09): Reapply "Merge pull request #82 from everycure-org/fix/run-disease-pipeline-for-all-crd" (Jacques Vergine)
- [f70a1216](https://github.com/everycure-org/core-entities/commit/f70a12161b591722ee4b824f4830c951022f17fc): Revert "Merge pull request #82 from everycure-org/fix/run-disease-pipeline-for-all-crd" (Jacques Vergine)
- [0189e9e0](https://github.com/everycure-org/core-entities/commit/0189e9e04f53e668df192adb9c209aae1f471466): Add version bump type to run llm pipeline (#84) (Jacques Vergine)
- [8d8b614c](https://github.com/everycure-org/core-entities/commit/8d8b614c9d41501c82327f38d0b070b7c1f3a7a9): reroute none labels to use disease-primary protocol instead; added experimental tag to note when this happens (Li-Vern Teo)
- [cb9678d4](https://github.com/everycure-org/core-entities/commit/cb9678d4a6b898c0664ead12ec6714630ebf60b6): update-disease-label-version (Li-Vern Teo)
- [71eab019](https://github.com/everycure-org/core-entities/commit/71eab019f420d963366abc2fcfd24a642feb9bca): Tidy up for disease_labels investigation (#80) (Jacques Vergine)
