# Release Comparison Report

**New Release:** `v1.2.14-drug_list`

**Base Release:** `v1.2.9-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.2.14/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.2.9/ec-drug-list.parquet`

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
| `atc_level_1` | 1100 |
| `atc_level_2` | 1205 |
| `atc_level_3` | 1231 |
| `atc_level_4` | 1338 |
| `atc_level_5` | 1394 |
| `atc_main` | 1757 |
| `l1_label` | 1768 |
| `l2_label` | 1755 |
| `l3_label` | 1746 |
| `l4_label` | 1717 |
| `l5_label` | 1354 |
| `synonyms` | 26 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `V` | `N` |
| `EC:01362` | Pyrantel pamoate | `N` | `*None*` |
| `EC:00296` | Cefiderocol | `D` | `J` |
| `EC:01811` | Nandrolone decanoate | `G` | `*None*` |
| `EC:01039` | Metoprolol | `S` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `V10` | `N07` |
| `EC:01362` | Pyrantel pamoate | `N02` | `*None*` |
| `EC:00296` | Cefiderocol | `D11` | `J01` |
| `EC:01811` | Nandrolone decanoate | `G03` | `*None*` |
| `EC:01039` | Metoprolol | `S01` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `V10A` | `N07X` |
| `EC:00296` | Cefiderocol | `D11A` | `J01D` |
| `EC:01811` | Nandrolone decanoate | `G03B` | `*None*` |
| `EC:01039` | Metoprolol | `S01E` | `*None*` |
| `EC:00625` | Esmolol | `S01E` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `*None*` | `N07XX` |
| `EC:00296` | Cefiderocol | `D11AC` | `J01DI` |
| `EC:01811` | Nandrolone decanoate | `G03BB` | `*None*` |
| `EC:01039` | Metoprolol | `S01ED` | `*None*` |
| `EC:00024` | Adalimumab | `L04AG` | `L04AB` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `*None*` | `N07XX08` |
| `EC:00296` | Cefiderocol | `D11AC08` | `J01DI04` |
| `EC:01811` | Nandrolone decanoate | `G03BB02` | `*None*` |
| `EC:00398` | Copanlisib | `*None*` | `L01EM02` |
| `EC:00024` | Adalimumab | `*None*` | `L04AB04` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `V10A` | `N07XX08` |
| `EC:01362` | Pyrantel pamoate | `N02` | `*None*` |
| `EC:00296` | Cefiderocol | `D11AC08` | `J01DI04` |
| `EC:01811` | Nandrolone decanoate | `G03BB02` | `*None*` |
| `EC:01039` | Metoprolol | `S01ED` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `various` | `Nervous system drugs` |
| `EC:01362` | Pyrantel pamoate | `nervous system` | `*None*` |
| `EC:00296` | Cefiderocol | `dermatologicals` | `Antiinfectives for systemic use` |
| `EC:01811` | Nandrolone decanoate | `genito urinary system and sex hormones` | `*None*` |
| `EC:01039` | Metoprolol | `sensory organs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `therapeutic radiopharmaceuticals` | `Other nervous system drugs in atc` |
| `EC:01362` | Pyrantel pamoate | `analgesics` | `*None*` |
| `EC:00296` | Cefiderocol | `other dermatological preparations` | `Antibacterials for systemic use` |
| `EC:01811` | Nandrolone decanoate | `sex hormones and modulators of the genital system` | `*None*` |
| `EC:01039` | Metoprolol | `ophthalmologicals` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `antiinflammatory agents` | `Other nervous system drugs in atc` |
| `EC:00296` | Cefiderocol | `other dermatological preparations` | `Other beta-lactam antibacterials in atc` |
| `EC:01811` | Nandrolone decanoate | `androgens` | `*None*` |
| `EC:01039` | Metoprolol | `antiglaucoma preparations and miotics` | `*None*` |
| `EC:00398` | Copanlisib | `protein kinase inhibitors` | `Protein kinase inhibitors, antineoplastic and immunomodulating agents` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `*None*` | `Other nervous system drugs in atc` |
| `EC:00296` | Cefiderocol | `medicated shampoos` | `Other cephalosporins and penems in atc` |
| `EC:01811` | Nandrolone decanoate | `5-androstanon (3) derivatives` | `*None*` |
| `EC:01039` | Metoprolol | `beta blocking agents` | `*None*` |
| `EC:00398` | Copanlisib | `phosphatidylinositol-3-kinase (pi3k) inhibitors` | `Phosphatidylinositol-3-kinase (pi3k) inhibitors` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01551` | Tafamidis | `*None*` | `Tafamidis` |
| `EC:00296` | Cefiderocol | `sulfur compounds` | `Cefiderocol` |
| `EC:01811` | Nandrolone decanoate | `androstanolone` | `*None*` |
| `EC:00398` | Copanlisib | `*None*` | `Copanlisib` |
| `EC:00024` | Adalimumab | `*None*` | `Adalimumab` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00359` | Clidinium | `['']` | `['Clidinium and psycholeptics']` |
| `EC:00834` | Insulin human | `['Insulin regular' 'Insulin']` | `['Insulin regular' 'Insulin' 'Insulin (human)']` |
| `EC:00477` | Deutivacaftor | `['']` | `['Deutivacaftor, tezacaftor and vanzacaftor']` |
| `EC:01367` | Pyrithione | `['Zinc pyrithione']` | `['Zinc pyrithione' 'Pyrithione zinc']` |
| `EC:01374` | Quinupristin | `['']` | `['Quinupristin/dalfopristin']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 58 | 512 |
| `atc_level_2` | 83 | 512 |
| `atc_level_3` | 115 | 512 |
| `atc_level_4` | 247 | 512 |
| `atc_level_5` | 1489 | 512 |
| `atc_main` | 58 | 512 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1805 | 1805 |
| `drug_class` | 1 | 1 |
| `drug_function` | 17 | 17 |
| `drug_target` | 23 | 23 |
| `drugbank_id` | 15 | 15 |
| `id` | 0 | 0 |
| `is_analgesic` | 0 | 0 |
| `is_antimicrobial` | 0 | 0 |
| `is_antipsychotic` | 0 | 0 |
| `is_cardiovascular` | 0 | 0 |
| `is_cell_therapy` | 0 | 0 |
| `is_chemotherapy` | 0 | 0 |
| `is_glucose_regulator` | 0 | 0 |
| `is_sedative` | 0 | 0 |
| `is_steroid` | 0 | 0 |
| `l1_label` | 58 | 512 |
| `l2_label` | 83 | 512 |
| `l3_label` | 115 | 512 |
| `l4_label` | 247 | 521 |
| `l5_label` | 1489 | 565 |
| `name` | 0 | 0 |
| `new_id` | 1806 | 1806 |
| `smiles` | 331 | 331 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
