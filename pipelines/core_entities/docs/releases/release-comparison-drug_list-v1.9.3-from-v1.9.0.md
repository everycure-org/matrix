# Release Comparison Report

**New Release:** `v1.9.3-drug_list`

**Base Release:** `v1.9.0-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.9.3/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.9.0/ec-drug-list.parquet`

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
| `EC:01966` | Fasudil |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `approved_usa` | 2 |
| `atc_level_1` | 386 |
| `atc_level_2` | 390 |
| `atc_level_3` | 393 |
| `atc_level_4` | 393 |
| `atc_level_5` | 393 |
| `atc_main` | 393 |
| `drug_target` | 2 |
| `drugbank_id` | 1 |
| `is_fda_generic_drug` | 9 |
| `l1_label` | 386 |
| `l2_label` | 390 |
| `l3_label` | 393 |
| `l4_label` | 389 |
| `l5_label` | 355 |
| `name` | 1 |
| `synonyms` | 3 |
| `translator_id` | 172 |

### Examples by Column

*Up to 5 examples per column; full comparison for `deleted` column*

#### `approved_usa`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01234` | Papaverine | `APPROVED` | `NOT_APPROVED` |
| `EC:01858` | Guanabenz | `APPROVED` | `DISCONTINUED` |

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `L` |
| `EC:01759` | Voriconazole | `J` | `*None*` |
| `EC:01555` | Tagraxofusp | `L` | `*None*` |
| `EC:01762` | Vosoritide | `M` | `*None*` |
| `EC:00546` | Dulaglutide | `A` | `*None*` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `L02` |
| `EC:01759` | Voriconazole | `J02` | `*None*` |
| `EC:01555` | Tagraxofusp | `L01` | `*None*` |
| `EC:01762` | Vosoritide | `M05` | `*None*` |
| `EC:00546` | Dulaglutide | `A10` | `*None*` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `L02B` |
| `EC:01759` | Voriconazole | `J02A` | `*None*` |
| `EC:01555` | Tagraxofusp | `L01X` | `*None*` |
| `EC:01762` | Vosoritide | `M05B` | `*None*` |
| `EC:00546` | Dulaglutide | `A10B` | `*None*` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `L02BB` |
| `EC:01759` | Voriconazole | `J02AC` | `*None*` |
| `EC:01555` | Tagraxofusp | `L01XX` | `*None*` |
| `EC:01762` | Vosoritide | `M05BX` | `*None*` |
| `EC:00546` | Dulaglutide | `A10BJ` | `*None*` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `L02BB04` |
| `EC:01759` | Voriconazole | `J02AC03` | `*None*` |
| `EC:01555` | Tagraxofusp | `L01XX67` | `*None*` |
| `EC:01762` | Vosoritide | `M05BX07` | `*None*` |
| `EC:00546` | Dulaglutide | `A10BJ05` | `*None*` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `L02BB04` |
| `EC:01759` | Voriconazole | `J02AC03` | `*None*` |
| `EC:01555` | Tagraxofusp | `L01XX67` | `*None*` |
| `EC:01762` | Vosoritide | `M05BX07` | `*None*` |
| `EC:00546` | Dulaglutide | `A10BJ05` | `*None*` |

#### `drug_target`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01126` | Netarsudil | `Rho Kinase Inhibitor` | `Rho kinase inhibitor` |
| `EC:00167` | Belumosudil | `kinase inhibitor (ROCK2)` | `ROCK2 inhibitor` |

#### `drugbank_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00094` | Rho(d) immune globulin | `DB11597` | `*None*` |

#### `is_fda_generic_drug`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01818` | Remibrutinib | `False` | `True` |
| `EC:00145` | Axitinib | `False` | `True` |
| `EC:00926` | Linaclotide | `False` | `True` |
| `EC:00465` | Desflurane | `False` | `True` |
| `EC:00802` | Ibrutinib | `False` | `True` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `Antineoplastic and immunomodulating agents` |
| `EC:01759` | Voriconazole | `Antiinfectives for systemic use` | `*None*` |
| `EC:01555` | Tagraxofusp | `Antineoplastic and immunomodulating agents` | `*None*` |
| `EC:01762` | Vosoritide | `Musculo-skeletal system drugs` | `*None*` |
| `EC:00546` | Dulaglutide | `Alimentary tract and metabolism drugs` | `*None*` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `Endocrine therapy antineoplastic and immunomodulating agents` |
| `EC:01759` | Voriconazole | `Antimycotics for systemic use` | `*None*` |
| `EC:01555` | Tagraxofusp | `Antineoplastic agents` | `*None*` |
| `EC:01762` | Vosoritide | `Drugs for treatment of bone diseases` | `*None*` |
| `EC:00546` | Dulaglutide | `Drugs used in diabetes` | `*None*` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `Hormone antagonists and related agents` |
| `EC:01759` | Voriconazole | `Antimycotics for systemic use` | `*None*` |
| `EC:01555` | Tagraxofusp | `Other antineoplastic agents in atc` | `*None*` |
| `EC:01762` | Vosoritide | `Drugs affecting bone structure and mineralization` | `*None*` |
| `EC:00546` | Dulaglutide | `Blood glucose lowering drugs, excl. insulins` | `*None*` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `Anti-androgen hormone antagonists and related agents` |
| `EC:01759` | Voriconazole | `Triazole and tetrazole derivatives` | `*None*` |
| `EC:01555` | Tagraxofusp | `Other antineoplastic agents in atc` | `*None*` |
| `EC:01762` | Vosoritide | `Other drugs affecting bone structure and mineralization in atc` | `*None*` |
| `EC:00546` | Dulaglutide | `Glucagon-like peptide-1 (glp-1) analogues` | `*None*` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00597` | Enzalutamide | `*None*` | `Enzalutamide` |
| `EC:01759` | Voriconazole | `Voriconazole` | `*None*` |
| `EC:01555` | Tagraxofusp | `Tagraxofusp` | `*None*` |
| `EC:01762` | Vosoritide | `Vosoritide` | `*None*` |
| `EC:00546` | Dulaglutide | `Dulaglutide` | `*None*` |

#### `name`

| ID | Old Value | New Value |
|----|-----------|-----------|
| `EC:00094` | `Anti-d immunoglobulin` | `Rho(d) immune globulin` |

#### `synonyms`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00409` | Cyanocobalamin | `['Vitamin b12']` | `['Vitamin b12' 'Methylcobalamin']` |
| `EC:00094` | Rho(d) immune globulin | `['Human rho(d) immune globulin' 'Anti-d (rh0) immunoglobulin']` | `['Anti-d (rh0) immunoglobulin' 'Anti-d immunoglobulin']` |
| `EC:00904` | Leuprolide | `['Leuprorelin acetate']` | `['Leuprorelin acetate' 'Leuprorelin']` |

#### `translator_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01901` | Oxaloacetate | `CHEBI:30744` | `CHEBI:16452` |
| `EC:00110` | Arsenic trioxide | `PUBCHEM.COMPOUND:261004` | `UNII:S7V92P67HO` |
| `EC:00059` | Alvimopan | `CHEBI:135686` | `UNII:677C126AET` |
| `EC:01384` | Ranitidine | `CHEBI:8776` | `CHEBI:92246` |
| `EC:01907` | Fructose | `CHEBI:28757` | `CHEBI:43703` |

## Null Values per Column

| Column | Base Release Null Count | New Release Null Count |
|--------|-------------------------|------------------------|
| `aggregated_with` | 0 | 0 |
| `approved_usa` | 0 | 0 |
| `atc_level_1` | 604 | 578 |
| `atc_level_2` | 604 | 578 |
| `atc_level_3` | 604 | 578 |
| `atc_level_4` | 604 | 578 |
| `atc_level_5` | 604 | 578 |
| `atc_main` | 604 | 578 |
| `deleted` | 0 | 0 |
| `deleted_reason` | 1926 | 1927 |
| `drug_class` | 1 | 1 |
| `drug_function` | 18 | 18 |
| `drug_target` | 26 | 26 |
| `drugbank_id` | 55 | 56 |
| `gras_usa` | 0 | 0 |
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
| `l1_label` | 604 | 578 |
| `l2_label` | 604 | 578 |
| `l3_label` | 604 | 578 |
| `l4_label` | 612 | 586 |
| `l5_label` | 652 | 626 |
| `name` | 0 | 0 |
| `new_id` | 1928 | 1929 |
| `synonyms` | 0 | 0 |
| `therapeutic_area` | 0 | 0 |
| `translator_id` | 0 | 0 |
