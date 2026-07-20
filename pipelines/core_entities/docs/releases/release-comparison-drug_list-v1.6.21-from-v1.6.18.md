# Release Comparison Report

**New Release:** `v1.6.21-drug_list`

**Base Release:** `v1.6.18-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.6.21/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.6.18/ec-drug-list.parquet`

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
| `atc_level_1` | 427 |
| `atc_level_2` | 429 |
| `atc_level_3` | 430 |
| `atc_level_4` | 433 |
| `atc_level_5` | 433 |
| `atc_main` | 433 |
| `drug_target` | 1 |
| `is_fda_generic_drug` | 3 |
| `l1_label` | 427 |
| `l2_label` | 429 |
| `l3_label` | 430 |
| `l4_label` | 431 |
| `l5_label` | 399 |
| `synonyms` | 1 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `L` | `*None*` |
| `EC:00064` | Amifostine | `V` | `*None*` |
| `EC:01047` | Miconazole | `D` | `J` |
| `EC:00148` | Azelaic acid | `*None*` | `D` |
| `EC:00955` | Lotilaner | `*None*` | `S` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `L02` | `*None*` |
| `EC:00064` | Amifostine | `V03` | `*None*` |
| `EC:01047` | Miconazole | `D01` | `J02` |
| `EC:00148` | Azelaic acid | `*None*` | `D10` |
| `EC:00955` | Lotilaner | `*None*` | `S01` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `L02B` | `*None*` |
| `EC:00064` | Amifostine | `V03A` | `*None*` |
| `EC:01047` | Miconazole | `D01A` | `J02A` |
| `EC:00148` | Azelaic acid | `*None*` | `D10A` |
| `EC:00955` | Lotilaner | `*None*` | `S01A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `L02BG` | `*None*` |
| `EC:00064` | Amifostine | `V03AF` | `*None*` |
| `EC:01047` | Miconazole | `D01AC` | `J02AB` |
| `EC:00148` | Azelaic acid | `*None*` | `D10AX` |
| `EC:00955` | Lotilaner | `*None*` | `S01AX` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `L02BG04` | `*None*` |
| `EC:00064` | Amifostine | `V03AF05` | `*None*` |
| `EC:01047` | Miconazole | `D01AC02` | `J02AB01` |
| `EC:00148` | Azelaic acid | `*None*` | `D10AX03` |
| `EC:00955` | Lotilaner | `*None*` | `S01AX25` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `L02BG04` | `*None*` |
| `EC:00064` | Amifostine | `V03AF05` | `*None*` |
| `EC:01047` | Miconazole | `D01AC02` | `J02AB01` |
| `EC:00148` | Azelaic acid | `*None*` | `D10AX03` |
| `EC:00955` | Lotilaner | `*None*` | `S01AX25` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01567` | Taurolidine | `Taruine derivative` | `Taurine derivative` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00031` | Afatinib | `False` | `True` |
| `EC:00508` | Dimethyl sulfoxide | `False` | `True` |
| `EC:00814` | Iloprost | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00064` | Amifostine | `Various drug classes in atc` | `*None*` |
| `EC:01047` | Miconazole | `Dermatologicals` | `Antiinfectives for systemic use` |
| `EC:00148` | Azelaic acid | `*None*` | `Dermatologicals` |
| `EC:00955` | Lotilaner | `*None*` | `Sensory organ drugs` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `Endocrine therapy antineoplastic and immunomodulating agents` | `*None*` |
| `EC:00064` | Amifostine | `All other therapeutic products` | `*None*` |
| `EC:01047` | Miconazole | `Antifungals for dermatological use` | `Antimycotics for systemic use` |
| `EC:00148` | Azelaic acid | `*None*` | `Anti-acne preparations` |
| `EC:00955` | Lotilaner | `*None*` | `Ophthalmologicals` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `Hormone antagonists and related agents` | `*None*` |
| `EC:00064` | Amifostine | `All other therapeutic products` | `*None*` |
| `EC:01047` | Miconazole | `Antifungals for topical use` | `Antimycotics for systemic use` |
| `EC:00148` | Azelaic acid | `*None*` | `Anti-acne preparations for topical use` |
| `EC:00955` | Lotilaner | `*None*` | `Antiinfective ophthalmologics` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `Aromatase inhibitors` | `*None*` |
| `EC:00064` | Amifostine | `Detoxifying agents for antineoplastic treatment` | `*None*` |
| `EC:01047` | Miconazole | `Imidazole and triazole derivatives, topical antifungals` | `Imidazole derivatives, antimycotic for systemic use` |
| `EC:00148` | Azelaic acid | `*None*` | `Other anti-acne preparations for topical use in atc` |
| `EC:00955` | Lotilaner | `*None*` | `Other antiinfectives in atc` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00902` | Letrozole | `Letrozole` | `*None*` |
| `EC:00064` | Amifostine | `Amifostine` | `*None*` |
| `EC:00148` | Azelaic acid | `*None*` | `Azelaic acid` |
| `EC:01222` | Paliperidone | `Paliperidone` | `*None*` |
| `EC:01020` | Methenamine | `*None*` | `Methenamine` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01711` | Ursodeoxycholic acid | `['Ursodiol']` | `['Ursodiol' 'Udca']` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 487 | 503 |
| `atc_level_2` | 487 | 503 |
| `atc_level_3` | 487 | 503 |
| `atc_level_4` | 487 | 503 |
| `atc_level_5` | 487 | 503 |
| `atc_main` | 487 | 503 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1816 | 1816 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 18 | 18 |
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
| `l1_label` | 487 | 503 |
| `l2_label` | 487 | 503 |
| `l3_label` | 487 | 503 |
| `l4_label` | 495 | 513 |
| `l5_label` | 535 | 562 |
| `name` | 0 | 0 |
| `new_id` | 1818 | 1818 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
