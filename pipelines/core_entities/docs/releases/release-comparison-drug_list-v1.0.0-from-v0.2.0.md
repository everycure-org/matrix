# Release Comparison Report

**New Release:** `v1.0.0-drug_list`

**Base Release:** `v0.2.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.0.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v0.2.0/ec-drug-list.parquet`

## Column Changes

### Added Columns
- `aggregated_with`
- `deleted`
- `deleted_reason`
- `is_analgesic`
- `is_cardiovascular`
- `is_cell_therapy`
- `new_id`
- `synonyms`

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 2

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01805` | Clenbuterol |
| `EC:01806` | Taurine |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 364 |
| `atc_level_2` | 477 |
| `atc_level_3` | 522 |
| `atc_level_4` | 670 |
| `atc_level_5` | 1499 |
| `atc_main` | 503 |
| `drugbank_id` | 15 |
| `l1_label` | 364 |
| `l2_label` | 477 |
| `l3_label` | 503 |
| `l4_label` | 670 |
| `l5_label` | 1499 |
| `smiles` | 331 |
| `translator_id` | 31 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `L` | `G` |
| `EC:00867` | Ketoconazole | `S` | `J` |
| `EC:00464` | Dequalinium | `M` | `G` |
| `EC:00686` | Flavoxate | `C` | `G` |
| `EC:00691` | Fluconazole | `S` | `J` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `L02` | `G03` |
| `EC:00867` | Ketoconazole | `S03` | `J02` |
| `EC:00464` | Dequalinium | `M01` | `G01` |
| `EC:00686` | Flavoxate | `C05` | `G04` |
| `EC:00691` | Fluconazole | `S03` | `J02` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `L02A` | `G03F` |
| `EC:00867` | Ketoconazole | `S03A` | `J02A` |
| `EC:00020` | Acoramidis | `` | `*None*` |
| `EC:00464` | Dequalinium | `M01C` | `G01A` |
| `EC:00686` | Flavoxate | `C05C` | `G04B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `L02AB` | `G03FB` |
| `EC:00867` | Ketoconazole | `S03AA` | `J02AB` |
| `EC:00020` | Acoramidis | `` | `*None*` |
| `EC:00464` | Dequalinium | `M01CA` | `G01AC` |
| `EC:01121` | Nemolizumab | `L04AG` | `L04AC` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `` | `*None*` |
| `EC:00810` | Idelalisib | `` | `*None*` |
| `EC:00137` | Avanafil | `` | `*None*` |
| `EC:00867` | Ketoconazole | `` | `*None*` |
| `EC:00706` | Flurazepam | `` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `L02AB` | `G03FB` |
| `EC:00867` | Ketoconazole | `S03AA` | `J02AB` |
| `EC:00464` | Dequalinium | `M01CA` | `G01AC` |
| `EC:01121` | Nemolizumab | `L04AG` | `L04AC` |
| `EC:00686` | Flavoxate | `C05CA` | `G04BD` |

#### `drugbank_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01788` | Zopapogene imadenovec | `` | `*None*` |
| `EC:01620` | Thymoglobulin | `` | `*None*` |
| `EC:00838` | Intravesical bacillus calmette-guerin | `` | `*None*` |
| `EC:00382` | Coagulation factor viii | `` | `*None*` |
| `EC:00615` | Ergonovine | `` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `antineoplastic and immunomodulating agents` | `genito urinary system and sex hormones` |
| `EC:00867` | Ketoconazole | `sensory organs` | `antiinfectives for systemic use` |
| `EC:00464` | Dequalinium | `musculo-skeletal system` | `genito urinary system and sex hormones` |
| `EC:00686` | Flavoxate | `cardiovascular system` | `genito urinary system and sex hormones` |
| `EC:00691` | Fluconazole | `sensory organs` | `antiinfectives for systemic use` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `endocrine therapy` | `sex hormones and modulators of the genital system` |
| `EC:00867` | Ketoconazole | `ophthalmological and otological preparations` | `antimycotics for systemic use` |
| `EC:00464` | Dequalinium | `antiinflammatory and antirheumatic products` | `gynecological antiinfectives and antiseptics` |
| `EC:00686` | Flavoxate | `vasoprotectives` | `urologicals` |
| `EC:00691` | Fluconazole | `ophthalmological and otological preparations` | `antimycotics for systemic use` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `hormones and related agents` | `progestogens and estrogens in combination` |
| `EC:00867` | Ketoconazole | `antiinfectives` | `antimycotics for systemic use` |
| `EC:00020` | Acoramidis | `` | `*None*` |
| `EC:00464` | Dequalinium | `specific antirheumatic agents` | `antiinfectives and antiseptics, excl. combinations with corticosteroids` |
| `EC:00686` | Flavoxate | `capillary stabilizing agents` | `urologicals` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `progestogens` | `progestogens and estrogens, sequential preparations` |
| `EC:00867` | Ketoconazole | `antiinfectives` | `imidazole derivatives` |
| `EC:00020` | Acoramidis | `` | `*None*` |
| `EC:00464` | Dequalinium | `quinolines` | `quinoline derivatives` |
| `EC:01121` | Nemolizumab | `monoclonal antibodies` | `interleukin inhibitors` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01804` | Dydrogesterone | `` | `*None*` |
| `EC:00810` | Idelalisib | `` | `*None*` |
| `EC:00137` | Avanafil | `` | `*None*` |
| `EC:00867` | Ketoconazole | `` | `*None*` |
| `EC:00706` | Flurazepam | `` | `*None*` |

#### `smiles`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00576` | Elranatamab | `` | `*None*` |
| `EC:01251` | Pembrolizumab | `` | `*None*` |
| `EC:00198` | Bevacizumab | `` | `*None*` |
| `EC:00238` | Bulevirtide | `` | `*None*` |
| `EC:01121` | Nemolizumab | `` | `*None*` |

#### `translator_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01302` | Plasminogen | `DRUGBANK:DB16701` | `UMLS:C0032140` |
| `EC:00823` | Inbakicept | `DRUGBANK:DB16479` | `UMLS:C5439902` |
| `EC:01254` | Pemivibart | `DRUGBANK:DB18720` | `UMLS:C5908975` |
| `EC:00383` | Coagulation factor x | `DRUGBANK:DB13148` | `UMLS:C2826078` |
| `EC:01659` | Toripalimab | `UNII:8JXN261VVA` | `UMLS:C4329337` |

## Commits

35 commits between 30/10/2025 and 07/11/2025 from the following authors: Jacques Vergine, everycure

- [24ca8c30](https://everycure@github.com/everycure-org/core-entities/commit/24ca8c30b64551736a4fb3721ba16394cc8eec04): Include additional columns in drug list release (#102) (Jacques Vergine)
- [baa536ea](https://everycure@github.com/everycure-org/core-entities/commit/baa536eac0d615d4d5ae32b65f898b301a3003b3): Fix release file path in release note (Jacques Vergine)
- [8e0b58f7](https://everycure@github.com/everycure-org/core-entities/commit/8e0b58f7224b1f998014e3a43ee07a75da5ba9fe): Update BQ source file from TSV to parquet (Jacques Vergine)
- [1e4c68cd](https://everycure@github.com/everycure-org/core-entities/commit/1e4c68cd3393319e1a5fc5d49c17f5d45fa87a9b): Release/v1.0.0-disease_list (#101) (everycure)
- [9c824f3b](https://everycure@github.com/everycure-org/core-entities/commit/9c824f3b15cd8bf9875af8060dfe214db84803f2): Update _list and -list (Jacques Vergine)
- [daff2131](https://everycure@github.com/everycure-org/core-entities/commit/daff213161885f89137c190b487fd66e486dedff): Replace disease obsolete by deleted & handle None (#100) (Jacques Vergine)
- [93dddc7b](https://everycure@github.com/everycure-org/core-entities/commit/93dddc7b06cf527a12ba97bc14dfa0a5cd7740b9): Add disease categories to release (#98) (Jacques Vergine)
- [bff5ef49](https://everycure@github.com/everycure-org/core-entities/commit/bff5ef49370deef3ee53008b2060423bba70ef8f): Include disease prevalence in release (#97) (Jacques Vergine)
- [34843343](https://everycure@github.com/everycure-org/core-entities/commit/348433432b9ac1eb4f81a2af9e9caab3f84afb50): Update TSV to parquet for disease pipelines (#96) (Jacques Vergine)
- [51520103](https://everycure@github.com/everycure-org/core-entities/commit/515201037269d4fd1f8c4d6a9c25d2fdaf68f8eb): Update readme (#94) (Jacques Vergine)
- [39a11a33](https://everycure@github.com/everycure-org/core-entities/commit/39a11a335dce0da6b74810598d73df92314f21c9): Update tags from drug to drug_list and disease to disease_list (Jacques Vergine)
- [56f662b9](https://everycure@github.com/everycure-org/core-entities/commit/56f662b9d8206859222b1751bcc6c277abfdb7b9): Update release version (Jacques Vergine)
- [48dd5238](https://everycure@github.com/everycure-org/core-entities/commit/48dd5238a6f2847d35edc1e77fec0a8c8d5ff3e3): Change run_llm_pipeline to run drug and disease list pipelines (Jacques Vergine)
- [8829814d](https://everycure@github.com/everycure-org/core-entities/commit/8829814d2321fc50ac2d849863d88e8e09fad1e7): Revert "Merge branch 'main' of github.com:everycure-org/core-entities" (Jacques Vergine)
- [2c8122a3](https://everycure@github.com/everycure-org/core-entities/commit/2c8122a3b88046da4f7e389dcff3b0fb298cdbed): Add author next to commit (Jacques Vergine)
- [fec0fc36](https://everycure@github.com/everycure-org/core-entities/commit/fec0fc363f12dd1d7371951164950118592e91d0): Release v0.2.1 notes for drug (#93) (everycure)
- [bcc5fad7](https://everycure@github.com/everycure-org/core-entities/commit/bcc5fad74db46da5f413dfc8eadb9dc14df1b6b7): Fetch tags when creating release PR (Jacques Vergine)
- [eeb81786](https://everycure@github.com/everycure-org/core-entities/commit/eeb817860c1b7298219bcf5d428ec2b8126a059e): Not run CI on draft PRs (Jacques Vergine)
- [038e792d](https://everycure@github.com/everycure-org/core-entities/commit/038e792d29110a569a2dff3852b770a8cbab21f7): Fix typo in create release PR (Jacques Vergine)
- [52e833ce](https://everycure@github.com/everycure-org/core-entities/commit/52e833ce3d1f24d65c94eb1be2dd80610087b106): Add post PR action (Jacques Vergine)
- [f5222f28](https://everycure@github.com/everycure-org/core-entities/commit/f5222f289983ae83f7d98fa7345a97b93a4ffa45): Move release notes to specific folder (Jacques Vergine)
- [a5d0aad4](https://everycure@github.com/everycure-org/core-entities/commit/a5d0aad4c4e2b4431a07286ec18d32dd2c926f79): Trigger create release pr after release dev (Jacques Vergine)
- [32781928](https://everycure@github.com/everycure-org/core-entities/commit/327819284d481edd903ac924c77883ed77d44dc9): Add \ after draft (Jacques Vergine)
- [d50fc80e](https://everycure@github.com/everycure-org/core-entities/commit/d50fc80e40ab03607b82e709fe7e62e06925933c): Install gh (Jacques Vergine)
- [a7776064](https://everycure@github.com/everycure-org/core-entities/commit/a7776064bad9e89526872f34928510102e78560b): Configure git releasebot (Jacques Vergine)
- [5f1a1a7c](https://everycure@github.com/everycure-org/core-entities/commit/5f1a1a7c011cde5b57d209c5723c60555a42bafc): Typo gs:// (Jacques Vergine)
- [6ccca23a](https://everycure@github.com/everycure-org/core-entities/commit/6ccca23a3ba48a02cb85056656948a459ed31b0e): Fix branch out step (Jacques Vergine)
- [cc3827bb](https://everycure@github.com/everycure-org/core-entities/commit/cc3827bb57a8f7080157e6395b94ce6782e6272f): Compare releases and create release PR (Jacques Vergine)
- [63973ca1](https://everycure@github.com/everycure-org/core-entities/commit/63973ca13636fcad4e098ec011cff5288b4b4f6f): Checkout, branch out and get latest public pipeline release (Jacques Vergine)
- [8c845723](https://everycure@github.com/everycure-org/core-entities/commit/8c845723ebcaf57704f80d31c295955ca1cf17ca): Authenticate to google and setup dev env (Jacques Vergine)
- [37f16ba8](https://everycure@github.com/everycure-org/core-entities/commit/37f16ba861559ad089b0b28a257f337ec2012a2d): Remove ignore errors on gcloud storage ls (Jacques Vergine)
- [c0391eae](https://everycure@github.com/everycure-org/core-entities/commit/c0391eaef1c20906bdd3f24be8e205788489814f): Improve release process and documentation for drug and disease (#88) (Jacques Vergine)
- [801c52bb](https://everycure@github.com/everycure-org/core-entities/commit/801c52bb357508391b74b8e60e38689fc66c35b0): Add action to create release PR with dummy steps (Jacques Vergine)
- [04a307b9](https://everycure@github.com/everycure-org/core-entities/commit/04a307b99a383a9b1358d8f4bd63401bdd7a3345): Include additional columns in drug list (aggregated_with, is_analgesic,is_cardiovascular) (#87) (Jacques Vergine)
- [a9aa9a96](https://everycure@github.com/everycure-org/core-entities/commit/a9aa9a967e8cacab9dd43234834b0f6ba66b0cd5): Update NN (#86) (Jacques Vergine)
