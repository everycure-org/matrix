# Release Comparison Report

**New Release:** `v1.2.5-drug_list`

**Base Release:** `v1.2.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.5/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.0/ec-drug-list.parquet`

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
| `therapeutic_area` | 462 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `L` | `B` |
| `EC:01782` | Zolbetuximab | `J` | `L` |
| `EC:01777` | Zileuton | `R` | `N` |
| `EC:01477` | Senna | `V` | `A` |
| `EC:00609` | Eptinezumab | `L` | `N` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `L04` | `B06` |
| `EC:01782` | Zolbetuximab | `J06` | `L01` |
| `EC:01777` | Zileuton | `R03` | `N01` |
| `EC:01477` | Senna | `V06` | `A06` |
| `EC:00609` | Eptinezumab | `L04` | `N02` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `L04A` | `B06A` |
| `EC:01782` | Zolbetuximab | `J06B` | `L01F` |
| `EC:01777` | Zileuton | `R03D` | `N01B` |
| `EC:01477` | Senna | `V06D` | `A06A` |
| `EC:00609` | Eptinezumab | `L04A` | `N02C` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `L04AG` | `B06AC` |
| `EC:01777` | Zileuton | `R03DC` | `N01BB` |
| `EC:01477` | Senna | `V06DC` | `A06AB` |
| `EC:00609` | Eptinezumab | `L04AG` | `N02CD` |
| `EC:00529` | Doravirine | `S01AD` | `J05AG` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01276` | Phenoxymethylpenicillin | `S01AA19` | `D11AC08` |
| `EC:00732` | Gabapentin | `N03AG03` | `N02BF01` |
| `EC:01202` | Oxacillin | `S01AA19` | `D11AC08` |
| `EC:00084` | Ampicillin | `D11AC08` | `S01AA19` |
| `EC:01097` | Nafcillin | `S01AA19` | `D11AC08` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `L04AG` | `B06AC` |
| `EC:01782` | Zolbetuximab | `J06B` | `L01F` |
| `EC:01777` | Zileuton | `R03DC` | `N01BB` |
| `EC:01477` | Senna | `V06DC` | `A06AB` |
| `EC:00609` | Eptinezumab | `L04AG` | `N02CD` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `antineoplastic and immunomodulating agents` | `blood and blood forming organs` |
| `EC:01782` | Zolbetuximab | `antiinfectives for systemic use` | `antineoplastic and immunomodulating agents` |
| `EC:01777` | Zileuton | `respiratory system` | `nervous system` |
| `EC:01477` | Senna | `various` | `alimentary tract and metabolism` |
| `EC:00609` | Eptinezumab | `antineoplastic and immunomodulating agents` | `nervous system` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `immunosuppressants` | `other hematological agents` |
| `EC:01782` | Zolbetuximab | `immune sera and immunoglobulins` | `antineoplastic agents` |
| `EC:01777` | Zileuton | `drugs for obstructive airway diseases` | `anesthetics` |
| `EC:01477` | Senna | `general nutrients` | `drugs for constipation` |
| `EC:00609` | Eptinezumab | `immunosuppressants` | `analgesics` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `immunosuppressants` | `other hematological agents` |
| `EC:01782` | Zolbetuximab | `immunoglobulins` | `monoclonal antibodies and antibody drug conjugates` |
| `EC:01777` | Zileuton | `other systemic drugs for obstructive airway diseases` | `anesthetics, local` |
| `EC:01477` | Senna | `other nutrients` | `drugs for constipation` |
| `EC:00609` | Eptinezumab | `immunosuppressants` | `antimigraine preparations` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00878` | Lanadelumab | `monoclonal antibodies` | `drugs used in hereditary angioedema` |
| `EC:01777` | Zileuton | `leukotriene receptor antagonists` | `amides` |
| `EC:01477` | Senna | `carbohydrates` | `contact laxatives` |
| `EC:00609` | Eptinezumab | `monoclonal antibodies` | `calcitonin gene-related peptide (cgrp) antagonists` |
| `EC:00529` | Doravirine | `antivirals` | `non-nucleoside reverse transcriptase inhibitors` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01276` | Phenoxymethylpenicillin | `ampicillin` | `sulfur compounds` |
| `EC:00732` | Gabapentin | `aminobutyric acid` | `gabapentin` |
| `EC:01202` | Oxacillin | `ampicillin` | `sulfur compounds` |
| `EC:00084` | Ampicillin | `sulfur compounds` | `ampicillin` |
| `EC:01097` | Nafcillin | `ampicillin` | `sulfur compounds` |

#### `therapeutic_area`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00239` | Bumetanide | `Renal \| Cardiovascular` | `Renal; Cardiovascular` |
| `EC:01364` | Pyridostigmine | `PNS` | `Peripheral nervous system` |
| `EC:01024` | Methotrexate | `Immune \| Traditional cancer therapy` | `Immune; Traditional cancer therapy` |
| `EC:01617` | Thiothixene | `CNS` | `Central nervous system` |
| `EC:00055` | Alprazolam | `CNS` | `Central nervous system` |

## Commits

5 commits between 01/12/2025 and 19/12/2025 from the following authors: Jacques Vergine, everycure

- [2a196db0](https://everycure@github.com/everycure-org/core-entities/commit/2a196db0239dc1768d72eaa63baa98b4690f529d): Add latest table on publish (#131) (Jacques Vergine)
- [61df4ebe](https://everycure@github.com/everycure-org/core-entities/commit/61df4ebe0bf510cb19cf62c6dd9ab2edd7f184c4): Remove reviews linked with "other" disease from dangling reviews logic (Jacques Vergine)
- [c51cfeff](https://everycure@github.com/everycure-org/core-entities/commit/c51cfeff209f0a7e58aca4d862da3f5b0bfc4a65): Get orchard reviews to DBT (Jacques Vergine)
- [1483f894](https://everycure@github.com/everycure-org/core-entities/commit/1483f894ef237d33278220d0149fc734ed8065be): Update release PR text (Jacques Vergine)
- [681838e5](https://everycure@github.com/everycure-org/core-entities/commit/681838e561cc57d7997c1202833a33a28ba45176): Release/v1.2.0-drug_list (#128) (everycure)
