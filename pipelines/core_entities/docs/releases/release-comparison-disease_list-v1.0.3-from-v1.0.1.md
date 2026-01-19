# Release Comparison Report

**New Release:** `v1.0.3-disease_list`

**Base Release:** `v1.0.1-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.0.3/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.0.1/ec-disease-list.parquet`

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
| `name` | 18 |

### Examples by Column

*Up to 5 examples per column*

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `MONDO:0001099` | `Lactocele` | `Galactocele` |
| `MONDO:0017227` | `Autoimmune pancreatitis type 1` | `Lymphoplasmacytic sclerosing pancreatitis` |
| `MONDO:0016583` | `Familial intestinal malrotation-facial anomalies syndrome` | `Stalker-Chitayat syndrome` |
| `MONDO:0023650` | `Littoral cell angioma of the spleen` | `Littoral cell angioma` |
| `MONDO:0017407` | `Deficiency in anterior pituitary function - variable immunodeficiency syndrome` | `David syndrome` |

## Commits

14 commits between 10/11/2025 and 14/11/2025 from the following authors: Jacques Vergine, everycure, may-lim

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
