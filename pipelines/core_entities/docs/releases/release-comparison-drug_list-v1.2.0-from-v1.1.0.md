# Release Comparison Report

**New Release:** `v1.2.0-drug_list`

**Base Release:** `v1.1.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.1.0/ec-drug-list.parquet`

## Summary
* Two new drugs
* Curated `drug_class` column
* Three new WIP columns: `therapeutic_area`, `drug_function` and `drug_target`
* Renaming of two drugs causing changes in their translator_id and drugbank_id

## Column Changes

### Added Columns
- `drug_function`
- `drug_target`
- `therapeutic_area`

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 2

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01812` | Ibudilast |
| `EC:01811` | Nandrolone decanoate |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 1 |
| `atc_level_2` | 1 |
| `atc_level_3` | 1 |
| `atc_level_4` | 1 |
| `atc_main` | 1 |
| `deleted` | 2 |
| `deleted_reason` | 2 |
| `drug_class` | 492 |
| `drugbank_id` | 1 |
| `is_analgesic` | 3 |
| `l1_label` | 1 |
| `l2_label` | 1 |
| `l3_label` | 1 |
| `l4_label` | 1 |
| `name` | 2 |
| `new_id` | 1 |
| `smiles` | 1 |
| `synonyms` | 3 |
| `translator_id` | 1 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `D` | `S` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `D03` | `S01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `D03A` | `S01B` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `*None*` | `S01BC` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `D03A` | `S01BC` |

#### `deleted`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01479` | Serdexmethylphenidate | `False` | `True` |
| `EC:01697` | Trolamine salicylate | `False` | `True` |

#### `deleted_reason`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01479` | Serdexmethylphenidate | `*None*` | `aggregated with EC:00484 Dexmethylphenidate as it's the prodrug` |
| `EC:01697` | Trolamine salicylate | `*None*` | `cosmetic product` |

#### `drug_class`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01148` | Nitrazepam | `*None*` | `Benzodiazepine` |
| `EC:01024` | Methotrexate | `Antifolate` | `Folate antimetabolite` |
| `EC:01447` | Rupatadine | `Histamine H1 agonist, second gen` | `Histamine H1 antagonist` |
| `EC:00907` | Levamisole | `Nicotinic receptor agonist` | `Antihelminthic` |
| `EC:00633` | Estrone | `Estrogen derivative` | `Estrogen agonist` |

#### `drugbank_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `DB13747` | `DB11079` |

#### `is_analgesic`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01026` | Methoxyflurane | `True` | `False` |
| `EC:00508` | Dimethyl sulfoxide | `False` | `True` |
| `EC:01697` | Trolamine salicylate | `True` | `False` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `dermatologicals` | `sensory organs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `preparations for treatment of wounds and ulcers` | `ophthalmologicals` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `cicatrizants` | `antiinflammatory agents` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `*None*` | `antiinflammatory agents, non-steroids` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `EC:01697` | `Trolamine` | `Trolamine salicylate` |
| `EC:00775` | `Haem arginate` | `Heme arginate` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01479` | Serdexmethylphenidate | `*None*` | `EC:00484` |

#### `smiles`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01697` | Trolamine salicylate | `OCCN(CCO)CCO` | `OCCN(CCO)CCO.OC(=O)C1=CC=CC=C1O` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01227` | Pamabrom | `['']` | `['Bromotheophylline' '8-bromotheophylline']` |
| `EC:00224` | Brexanolone | `['']` | `['Brexanolone']` |
| `EC:01697` | Trolamine salicylate | `['Salicylate']` | `['']` |

#### `translator_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00775` | Heme arginate | `PUBCHEM.COMPOUND:135564839` | `UNII:R1B526117P` |

## Commits

12 commits between 26/11/2025 and 01/12/2025 from the following authors: Jacques Vergine, everycure

- [5d7cd06b](https://everycure@github.com/everycure-org/core-entities/commit/5d7cd06b4157949cb53132a4c2c09c1f7831080d): Add drug_class, therapeutic_area, drug_function and drug_target columns (#127) (Jacques Vergine)
- [ab83de37](https://everycure@github.com/everycure-org/core-entities/commit/ab83de37c54e646f91a58f8e6dd3f82d2be63a55): Release v1.0.6 notes for disease_list (#124) (everycure)
- [d0580d89](https://everycure@github.com/everycure-org/core-entities/commit/d0580d895e83d40e1221250963aa47c33db3409c): Fix None in remapping columns (Jacques Vergine)
- [345c4466](https://everycure@github.com/everycure-org/core-entities/commit/345c446646b0368b7da8e4c1a370da8157fa9fc4): Set release_version variables earlier in pipeline (Jacques Vergine)
- [c59b03b4](https://everycure@github.com/everycure-org/core-entities/commit/c59b03b4235699c167a6a54124057fc2444abe62): Add logs for env variables in action (Jacques Vergine)
- [72f363d1](https://everycure@github.com/everycure-org/core-entities/commit/72f363d1a3a1172775eac29ad5122feb5c9176c9): Fix other PIPELINE_NAME dug (Jacques Vergine)
- [22231f59](https://everycure@github.com/everycure-org/core-entities/commit/22231f597af8f692400359207f5472f9b274c26e): Fix PIPELINE_NAME env variable (Jacques Vergine)
- [e2c575ff](https://everycure@github.com/everycure-org/core-entities/commit/e2c575ffc2334ea8239a3e1affcc1325a2e166dc): Fix systel -> system typo (Jacques Vergine)
- [c16c247c](https://everycure@github.com/everycure-org/core-entities/commit/c16c247c5ae03fa51819e4dd3e802d3bf18e867c): Fix publish action and pipeline (#122) (Jacques Vergine)
- [41f7f720](https://everycure@github.com/everycure-org/core-entities/commit/41f7f720ee01425c4b2db759d45b409fbe46e320): Add service account in publish action (Jacques Vergine)
- [08cd5339](https://everycure@github.com/everycure-org/core-entities/commit/08cd53398299c1de2278d216456dc37097e353d5): Release v1.1.0 notes for drug_list (#121) (everycure)
- [190c726b](https://everycure@github.com/everycure-org/core-entities/commit/190c726b9a172ef34bd26341f7841b9a3c735441): Fix makefile release gh action name (Jacques Vergine)
