# Release Comparison Report

**New Release:** `v1.3.1-drug_list`

**Base Release:** `v1.3.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.3.1/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.3.0/ec-drug-list.parquet`

## Column Changes

### Added Columns
*No columns added*

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 1

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01819` | Ambroxol |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 275 |
| `atc_level_2` | 279 |
| `atc_level_3` | 280 |
| `atc_level_4` | 282 |
| `atc_level_5` | 282 |
| `atc_main` | 282 |
| `l1_label` | 275 |
| `l2_label` | 279 |
| `l3_label` | 280 |
| `l4_label` | 281 |
| `l5_label` | 249 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `N` | `*None*` |
| `EC:01744` | Viloxazine | `N` | `*None*` |
| `EC:00923` | Lidocaine | `N` | `D` |
| `EC:01176` | Oliceridine | `N` | `*None*` |
| `EC:00692` | Flucytosine | `J` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `N06` | `*None*` |
| `EC:01744` | Viloxazine | `N06` | `*None*` |
| `EC:00923` | Lidocaine | `N01` | `D04` |
| `EC:01176` | Oliceridine | `N02` | `*None*` |
| `EC:00692` | Flucytosine | `J02` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `N06D` | `*None*` |
| `EC:01744` | Viloxazine | `N06A` | `*None*` |
| `EC:00923` | Lidocaine | `N01B` | `D04A` |
| `EC:01176` | Oliceridine | `N02A` | `*None*` |
| `EC:00692` | Flucytosine | `J02A` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `N06DA` | `*None*` |
| `EC:01744` | Viloxazine | `N06AX` | `*None*` |
| `EC:00923` | Lidocaine | `N01BB` | `D04AB` |
| `EC:01176` | Oliceridine | `N02AX` | `*None*` |
| `EC:00692` | Flucytosine | `J02AX` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `N06DA06` | `*None*` |
| `EC:01744` | Viloxazine | `N06AX09` | `*None*` |
| `EC:00923` | Lidocaine | `N01BB02` | `D04AB01` |
| `EC:01176` | Oliceridine | `N02AX07` | `*None*` |
| `EC:00692` | Flucytosine | `J02AX01` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `N06DA06` | `*None*` |
| `EC:01744` | Viloxazine | `N06AX09` | `*None*` |
| `EC:00923` | Lidocaine | `N01BB02` | `D04AB01` |
| `EC:01176` | Oliceridine | `N02AX07` | `*None*` |
| `EC:00692` | Flucytosine | `J02AX01` | `*None*` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `Nervous system drugs` | `*None*` |
| `EC:01744` | Viloxazine | `Nervous system drugs` | `*None*` |
| `EC:00923` | Lidocaine | `Nervous system drugs` | `Dermatologicals` |
| `EC:01176` | Oliceridine | `Nervous system drugs` | `*None*` |
| `EC:00692` | Flucytosine | `Antiinfectives for systemic use` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `Psychoanaleptics` | `*None*` |
| `EC:01744` | Viloxazine | `Psychoanaleptics` | `*None*` |
| `EC:00923` | Lidocaine | `Anesthetics` | `Antipruritics, incl. antihistamines, anesthetics, etc.` |
| `EC:01176` | Oliceridine | `Analgesics` | `*None*` |
| `EC:00692` | Flucytosine | `Antimycotics for systemic use` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `Anti-dementia drugs` | `*None*` |
| `EC:01744` | Viloxazine | `Antidepressants` | `*None*` |
| `EC:00923` | Lidocaine | `Local anesthetics` | `Antipruritics, incl. antihistamines, anesthetics, etc.` |
| `EC:01176` | Oliceridine | `Opioid analgesics` | `*None*` |
| `EC:00692` | Flucytosine | `Antimycotics for systemic use` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00177` | Benzgalantamine | `Anticholinesterase anti-dementia drugs` | `*None*` |
| `EC:01744` | Viloxazine | `Other antidepressants in atc` | `*None*` |
| `EC:00923` | Lidocaine | `Amide local anesthetics` | `Anesthetics for topical use` |
| `EC:01176` | Oliceridine | `Other opioids in atc` | `*None*` |
| `EC:00692` | Flucytosine | `Other antimycotics for systemic use in atc` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01744` | Viloxazine | `Viloxazine` | `*None*` |
| `EC:01176` | Oliceridine | `Oliceridine` | `*None*` |
| `EC:00692` | Flucytosine | `Flucytosine` | `*None*` |
| `EC:00035` | Albendazole | `Albendazole` | `*None*` |
| `EC:01412` | Rifapentine | `Rifapentine` | `*None*` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 243 | 507 |
| `atc_level_2` | 243 | 507 |
| `atc_level_3` | 243 | 507 |
| `atc_level_4` | 243 | 507 |
| `atc_level_5` | 243 | 507 |
| `atc_main` | 243 | 507 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1810 | 1811 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 24 | 25 |
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
| `l1_label` | 243 | 507 |
| `l2_label` | 243 | 507 |
| `l3_label` | 243 | 507 |
| `l4_label` | 253 | 516 |
| `l5_label` | 303 | 552 |
| `name` | 0 | 0 |
| `new_id` | 1812 | 1813 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
