# Release Comparison Report

**New Release:** `v1.0.2-drug_list`

**Base Release:** `v1.0.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.0.2/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.0.0/ec-drug-list.parquet`

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
| `deleted` | 1 |
| `deleted_reason` | 1803 |
| `is_cardiovascular` | 1 |
| `new_id` | 1803 |
| `synonyms` | 2 |

### Examples by Column

*Up to 5 examples per column*

#### `deleted`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00775` | Haem arginate | `False` | `True` |

#### `deleted_reason`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00023` | Adagrasib | `` | `*None*` |
| `EC:00876` | Lamivudine | `` | `*None*` |
| `EC:01357` | Propylhexedrine | `` | `*None*` |
| `EC:01029` | Methyl aminolevulinate | `` | `*None*` |
| `EC:01049` | Midodrine | `` | `*None*` |

#### `is_cardiovascular`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00348` | Ciprofibrate | `False` | `True` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00023` | Adagrasib | `` | `*None*` |
| `EC:00876` | Lamivudine | `` | `*None*` |
| `EC:01357` | Propylhexedrine | `` | `*None*` |
| `EC:01029` | Methyl aminolevulinate | `` | `*None*` |
| `EC:01049` | Midodrine | `` | `*None*` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01043` | Metyrosine | `['']` | `['Metirosine']` |
| `EC:00779` | Hemin | `['']` | `['Haem arginate, heme arginate']` |

## Commits

21 commits between 07/11/2025 and 14/11/2025 from the following authors: Jacques Vergine, everycure, may-lim

- [5c6aa007](https://everycure@github.com/everycure-org/core-entities/commit/5c6aa007f6a47e8de78fe173a76d602febebd9fb): Fix isna .all (Jacques Vergine)
- [2155ec61](https://everycure@github.com/everycure-org/core-entities/commit/2155ec61ffad11508eeb50e5e018b8e30c66fceb): Update release PR action name (Jacques Vergine)
- [59c68f3b](https://everycure@github.com/everycure-org/core-entities/commit/59c68f3be9ae461a0fb48ef02690d9ea9aca7e33): Fix run pipeline matrix (Jacques Vergine)
- [a212b785](https://everycure@github.com/everycure-org/core-entities/commit/a212b785e8545ed6674072c051d08a680bdeab68): Update actions pre-move (Jacques Vergine)
- [d599ad12](https://everycure@github.com/everycure-org/core-entities/commit/d599ad120601f63579b95a6ab47c3ad2f36dc78e): Remove unused curation utils (Jacques Vergine)
- [8796baaf](https://everycure@github.com/everycure-org/core-entities/commit/8796baaf646edc6172472ff4deb6bf4df3887fea): Refactor matrix dependencies (BQ dataset and inject) (#109) (Jacques Vergine)
- [454f309d](https://everycure@github.com/everycure-org/core-entities/commit/454f309d49684dc0efdcb760ce8e4f4d826623d4): Update new_id (#110) (Jacques Vergine)
- [a73eab59](https://everycure@github.com/everycure-org/core-entities/commit/a73eab59e3c489ed1f42521c449fa8165a2bdce2): Update run_pipeline to run every two weeks for drug and disease lists (#108) (Jacques Vergine)
- [c43cd7f8](https://everycure@github.com/everycure-org/core-entities/commit/c43cd7f84a175f401bb562646c4b155cabeb861f): Fixing tabs (may-lim)
- [cb14a3c5](https://everycure@github.com/everycure-org/core-entities/commit/cb14a3c56a0e4a09da4c9cd48190da3113622444): Update disease_name_patch.tsv (may-lim)
- [947cf463](https://everycure@github.com/everycure-org/core-entities/commit/947cf4638305ef40bd4667046e1c9ec7e3ef03c5): Update disease_name_patch.tsv (may-lim)
- [42e28eef](https://everycure@github.com/everycure-org/core-entities/commit/42e28eef8d6db6f3d581b1ee25646c5860948b3b): Update disease_name_patch.tsv (may-lim)
- [cbce1a8e](https://everycure@github.com/everycure-org/core-entities/commit/cbce1a8eccbd838598f8554f41421eb9494dc151): Update README.md (Jacques Vergine)
- [c406e377](https://everycure@github.com/everycure-org/core-entities/commit/c406e3775e52566b4bc6af2065ee4386790b30ca): Release v1.0.1 notes for disease_list (#106) (everycure)
- [18be2acc](https://everycure@github.com/everycure-org/core-entities/commit/18be2acc12ce76aea8a4046e7dd9339d649d0422): Add is_infectious_disease (Jacques Vergine)
- [e8bd22b1](https://everycure@github.com/everycure-org/core-entities/commit/e8bd22b1780619dc20d99378f19116bc3fb13578): Add publish pipelines to write disease and drug lists to gitops (#104) (Jacques Vergine)
- [0f39867c](https://everycure@github.com/everycure-org/core-entities/commit/0f39867c3dd8439687a7abe879f4f997bcd463f4): Remove external data catalog (Jacques Vergine)
- [f21fa9f6](https://everycure@github.com/everycure-org/core-entities/commit/f21fa9f64836a596f01c51d4a20b297b72b579cf): Save Node norm and Nameres versions (#105) (Jacques Vergine)
- [94ec9500](https://everycure@github.com/everycure-org/core-entities/commit/94ec95008f4ccb5b66019fb7381d3b71386bc760): Fix disease list v1.0.0 release notes (Jacques Vergine)
- [32e3ca16](https://everycure@github.com/everycure-org/core-entities/commit/32e3ca16b19b845039a19887ce33ecbd368a5f29): Release v1.0.0 notes for drug_list (#103) (everycure)
- [5bea9bb6](https://everycure@github.com/everycure-org/core-entities/commit/5bea9bb632e0a402d38f27f6a069a172598d295a): Update readme with minor logic (Jacques Vergine)
