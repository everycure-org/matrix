# Release Comparison Report

**New Release:** `v1.9.0-drug_list`

**Base Release:** `v1.6.29-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.9.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.29/ec-drug-list.parquet`

## Column Changes

### Added Columns
- `gras_usa`

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 108

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01875` | L-threonine |
| `EC:01864` | L-asparagine |
| `EC:01915` | D-pinitol |
| `EC:01916` | Glucuronolactone |
| `EC:01872` | L-isoleucine |
| `EC:01953` | Arachidonic acid |
| `EC:01889` | Dimethylglycine |
| `EC:01932` | Selenomethionine |
| `EC:01917` | L-arabinose |
| `EC:01874` | L-proline |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `approved_usa` | 5 |
| `atc_level_1` | 436 |
| `atc_level_2` | 438 |
| `atc_level_3` | 440 |
| `atc_level_4` | 441 |
| `atc_level_5` | 441 |
| `atc_main` | 441 |
| `drug_class` | 1 |
| `drug_function` | 1 |
| `drug_target` | 5 |
| `is_fda_generic_drug` | 2 |
| `l1_label` | 436 |
| `l2_label` | 438 |
| `l3_label` | 440 |
| `l4_label` | 440 |
| `l5_label` | 407 |
| `synonyms` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `approved_usa`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01806` | Taurine | `APPROVED` | `NOT_APPROVED` |
| `EC:01856` | Alpha-ketoglutarate | `APPROVED` | `NOT_APPROVED` |
| `EC:01821` | Beta-hydroxybutyrate | `APPROVED` | `NOT_APPROVED` |
| `EC:01823` | Trehalose | `APPROVED` | `NOT_APPROVED` |
| `EC:01824` | D-ribose | `APPROVED` | `NOT_APPROVED` |

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `V` | `S` |
| `EC:00577` | Eltrombopag | `*None*` | `B` |
| `EC:00344` | Cimetidine | `A` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `N` |
| `EC:00360` | Clindamycin | `*None*` | `J` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `V03` | `S01` |
| `EC:00577` | Eltrombopag | `*None*` | `B02` |
| `EC:00344` | Cimetidine | `A02` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `N06` |
| `EC:00360` | Clindamycin | `*None*` | `J01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `V03A` | `S01E` |
| `EC:00577` | Eltrombopag | `*None*` | `B02B` |
| `EC:00344` | Cimetidine | `A02B` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `N06B` |
| `EC:00360` | Clindamycin | `*None*` | `J01F` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `V03AB` | `S01EB` |
| `EC:00577` | Eltrombopag | `*None*` | `B02BX` |
| `EC:00344` | Cimetidine | `A02BA` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `N06BA` |
| `EC:00360` | Clindamycin | `*None*` | `J01FF` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `V03AB19` | `S01EB05` |
| `EC:00577` | Eltrombopag | `*None*` | `B02BX05` |
| `EC:00344` | Cimetidine | `A02BA01` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `N06BA04` |
| `EC:00360` | Clindamycin | `*None*` | `J01FF01` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `V03AB19` | `S01EB05` |
| `EC:00577` | Eltrombopag | `*None*` | `B02BX05` |
| `EC:00344` | Cimetidine | `A02BA01` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `N06BA04` |
| `EC:00360` | Clindamycin | `*None*` | `J01FF01` |

#### `drug_class`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01856` | Alpha-ketoglutarate | `Metabolic supplement` | `Dicarboxylic acid` |

#### `drug_function`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01856` | Alpha-ketoglutarate | `Metabolic substrate (TCA intermediate)` | `TCA-cycle intermediate` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01856` | Alpha-ketoglutarate | `TCA cycle intermediate; amino acid metabolism` | `Anaplerotic carbon and nitrogen metabolism intermediate` |
| `EC:00773` | Guanfacine | `Alpha2-Adrenergic agonist` | `Alpha-2 adrenergic agonist` |
| `EC:00942` | Lofexidine | `Alpha2-Adrenergic Agonist` | `Alpha-2 adrenergic agonist` |
| `EC:00374` | Clonidine | `Alpha2-Adrenergic Agonist` | `Alpha-2 adrenergic agonist` |
| `EC:01030` | Methyldopa | `Alpha2-Adrenergic Agonist` | `Alpha-2 adrenergic agonist` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00498` | Difelikefalin | `False` | `True` |
| `EC:01391` | Relugolix | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `Various drug classes in atc` | `Sensory organ drugs` |
| `EC:00577` | Eltrombopag | `*None*` | `Blood and blood forming organ drugs` |
| `EC:00344` | Cimetidine | `Alimentary tract and metabolism drugs` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `Nervous system drugs` |
| `EC:00360` | Clindamycin | `*None*` | `Antiinfectives for systemic use` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `All other therapeutic products` | `Ophthalmologicals` |
| `EC:00577` | Eltrombopag | `*None*` | `Antihemorrhagics` |
| `EC:00344` | Cimetidine | `Drugs for acid related disorders` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `Psychoanaleptics` |
| `EC:00360` | Clindamycin | `*None*` | `Antibacterials for systemic use` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `All other therapeutic products` | `Antiglaucoma preparations and miotics` |
| `EC:00577` | Eltrombopag | `*None*` | `Vitamin k and other hemostatics` |
| `EC:00344` | Cimetidine | `Drugs for peptic ulcer and gastro-oesophageal reflux disease (gord)` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `Psychostimulants, agents used for adhd and nootropics` |
| `EC:00360` | Clindamycin | `*None*` | `Macrolides, lincosamides and streptogramins` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01281` | Physostigmine | `Antidotes` | `Parasympathomimetics, antiglaucoma preparations and miotics` |
| `EC:00577` | Eltrombopag | `*None*` | `Other systemic hemostatics in atc` |
| `EC:00344` | Cimetidine | `H2-receptor antagonists` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `Centrally acting sympathomimetics` |
| `EC:00360` | Clindamycin | `*None*` | `Lincosamides` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00577` | Eltrombopag | `*None*` | `Eltrombopag` |
| `EC:00344` | Cimetidine | `Cimetidine` | `*None*` |
| `EC:01034` | Methylphenidate | `*None*` | `Methylphenidate` |
| `EC:00360` | Clindamycin | `*None*` | `Clindamycin` |
| `EC:00292` | Cefadroxil | `*None*` | `Cefadroxil` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01856` | Alpha-ketoglutarate | `['Α-ketoglutaric acid' '2-oxoglutarate' '2-ketoglutarate'
 '2-oxopentanedioate']` | `['Α-ketoglutaric acid' '2-oxoglutarate' '2-ketoglutarate'
 '2-oxopentanedioate' 'A-kg, α-kg, αkg']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 506 | 604 |
| `atc_level_2` | 506 | 604 |
| `atc_level_3` | 506 | 604 |
| `atc_level_4` | 506 | 604 |
| `atc_level_5` | 506 | 604 |
| `atc_main` | 506 | 604 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1818 | 1926 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 19 | 55 |
| `gras_usa` | N/A | 0 |
| `id` | 0 | 0 |
| `is_analgesic` | 0 | 0 |
| `is_antimicrobial` | 0 | 0 |
| `is_antipsychotic` | 0 | 0 |
| `is_cardiovascular` | 0 | 0 |
| `is_cell_therapy` | 0 | 0 |
| `is_chemotherapy` | 0 | 0 |
| `is_fda_generic_drug` | 0 | 0 |
| `is_glucose_regulator` | 0 | 0 |
| `is_sedative` | 0 | 0 |
| `is_steroid` | 0 | 0 |
| `l1_label` | 506 | 604 |
| `l2_label` | 506 | 604 |
| `l3_label` | 506 | 604 |
| `l4_label` | 513 | 612 |
| `l5_label` | 554 | 652 |
| `name` | 0 | 0 |
| `new_id` | 1820 | 1928 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
