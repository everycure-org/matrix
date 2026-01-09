# Release Comparison Report

**New Release:** `v1.2.9-drug_list`

**Base Release:** `v1.2.5-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.9/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.5/ec-drug-list.parquet`

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
| `atc_level_1` | 309 |
| `atc_level_2` | 398 |
| `atc_level_3` | 411 |
| `atc_level_4` | 426 |
| `atc_level_5` | 16 |
| `atc_main` | 449 |
| `l1_label` | 309 |
| `l2_label` | 398 |
| `l3_label` | 392 |
| `l4_label` | 426 |
| `l5_label` | 16 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `C` | `S` |
| `EC:01613` | Thiamine | `V` | `B` |
| `EC:01553` | Tafenoquine | `M` | `P` |
| `EC:00741` | Gemcitabine | `L` | `V` |
| `EC:00556` | Econazole | `J` | `S` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `C07` | `S01` |
| `EC:01613` | Thiamine | `V01` | `B03` |
| `EC:01553` | Tafenoquine | `M01` | `P01` |
| `EC:00271` | Capreomycin | `S03` | `S01` |
| `EC:00741` | Gemcitabine | `L01` | `V06` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `C07A` | `S01E` |
| `EC:01613` | Thiamine | `V01A` | `B03B` |
| `EC:01553` | Tafenoquine | `M01C` | `P01B` |
| `EC:00271` | Capreomycin | `S03A` | `S01A` |
| `EC:00741` | Gemcitabine | `L01B` | `V06D` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `C07AB` | `S01ED` |
| `EC:01613` | Thiamine | `V01AA` | `B03BA` |
| `EC:01553` | Tafenoquine | `M01CA` | `P01BA` |
| `EC:00271` | Capreomycin | `S03AA` | `S01AA` |
| `EC:00741` | Gemcitabine | `L01BC` | `V06DC` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01613` | Thiamine | `V01AA08` | `B03BA01` |
| `EC:00732` | Gabapentin | `N02BF01` | `N03AG03` |
| `EC:01365` | Pyridoxine | `V01AA08` | `B03BA01` |
| `EC:00493` | Dicloxacillin | `D11AC08` | `S01AA19` |
| `EC:01097` | Nafcillin | `D11AC08` | `S01AA19` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `C07AB` | `S01ED` |
| `EC:01613` | Thiamine | `V01AA08` | `B03BA01` |
| `EC:01553` | Tafenoquine | `M01CA` | `P01BA` |
| `EC:00271` | Capreomycin | `S03AA` | `S01AA` |
| `EC:00741` | Gemcitabine | `L01BC` | `V06DC` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `cardiovascular system` | `sensory organs` |
| `EC:01613` | Thiamine | `various` | `blood and blood forming organs` |
| `EC:01553` | Tafenoquine | `musculo-skeletal system` | `antiparasitic products, insecticides and repellents` |
| `EC:00741` | Gemcitabine | `antineoplastic and immunomodulating agents` | `various` |
| `EC:00556` | Econazole | `antiinfectives for systemic use` | `sensory organs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `beta blocking agents` | `ophthalmologicals` |
| `EC:01613` | Thiamine | `allergens` | `antianemic preparations` |
| `EC:01553` | Tafenoquine | `antiinflammatory and antirheumatic products` | `antiprotozoals` |
| `EC:00271` | Capreomycin | `ophthalmological and otological preparations` | `ophthalmologicals` |
| `EC:00741` | Gemcitabine | `antineoplastic agents` | `general nutrients` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `beta blocking agents` | `antiglaucoma preparations and miotics` |
| `EC:01613` | Thiamine | `allergens` | `vitamin b12 and folic acid` |
| `EC:01553` | Tafenoquine | `specific antirheumatic agents` | `antimalarials` |
| `EC:00741` | Gemcitabine | `antimetabolites` | `other nutrients` |
| `EC:00732` | Gabapentin | `other analgesics and antipyretics` | `antiepileptics` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01039` | Metoprolol | `beta blocking agents, selective` | `beta blocking agents` |
| `EC:01613` | Thiamine | `allergen extracts` | `vitamin b12 (cyanocobalamin and analogues)` |
| `EC:01553` | Tafenoquine | `quinolines` | `aminoquinolines` |
| `EC:00271` | Capreomycin | `antiinfectives` | `antibiotics` |
| `EC:00741` | Gemcitabine | `pyrimidine analogues` | `carbohydrates` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01613` | Thiamine | `food` | `cyanocobalamin` |
| `EC:00732` | Gabapentin | `gabapentin` | `aminobutyric acid` |
| `EC:01365` | Pyridoxine | `food` | `cyanocobalamin` |
| `EC:00493` | Dicloxacillin | `sulfur compounds` | `ampicillin` |
| `EC:01097` | Nafcillin | `sulfur compounds` | `ampicillin` |
