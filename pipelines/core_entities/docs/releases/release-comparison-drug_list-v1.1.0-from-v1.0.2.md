# Release Comparison Report

**New Release:** `v1.1.0-drug_list`

**Base Release:** `v1.0.2-drug_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/drug_list/v1.1.0/03_primary/release/ec-drug-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/drug_list/v1.0.2/ec-drug-list.parquet`

## Column Changes

### Added Columns
- `drug_class`

### Removed Columns
*No columns removed*

## Row Changes

### Added Rows
**Total:** 4

**Examples (up to 10):**

| ID | Name |
|----|------|
| `EC:01807` | Epigallocatechin gallate |
| `EC:01808` | Elamipretide |
| `EC:01810` | Nimorazole |
| `EC:01809` | Anethole trithione |

### Removed Rows
**Total:** 0


## Value Changes

### Summary by Column

| Column | Number of Changes |
|--------|-------------------|
| `atc_level_1` | 307 |
| `atc_level_2` | 396 |
| `atc_level_3` | 409 |
| `atc_level_4` | 424 |
| `atc_level_5` | 15 |
| `atc_main` | 447 |
| `deleted` | 1 |
| `deleted_reason` | 1 |
| `is_cardiovascular` | 1 |
| `is_cell_therapy` | 1 |
| `is_chemotherapy` | 1 |
| `l1_label` | 307 |
| `l2_label` | 396 |
| `l3_label` | 390 |
| `l4_label` | 424 |
| `l5_label` | 15 |
| `new_id` | 1 |

### Examples by Column

*Up to 5 examples per column*

#### `atc_level_1`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `C` | `S` |
| `EC:00324` | Chlorambucil | `L` | `N` |
| `EC:00252` | Cabotegravir | `J` | `S` |
| `EC:00709` | Fluticasone | `R` | `S` |
| `EC:01115` | Nedocromil | `M` | `R` |

#### `atc_level_2`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `C07` | `S01` |
| `EC:00324` | Chlorambucil | `L01` | `N01` |
| `EC:00252` | Cabotegravir | `J05` | `S03` |
| `EC:00709` | Fluticasone | `R07` | `S03` |
| `EC:00827` | Indomethacin | `M01` | `M02` |

#### `atc_level_3`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `C07A` | `S01E` |
| `EC:00324` | Chlorambucil | `L01A` | `N01A` |
| `EC:00252` | Cabotegravir | `J05A` | `S03A` |
| `EC:00709` | Fluticasone | `R07A` | `S03B` |
| `EC:00827` | Indomethacin | `M01A` | `M02A` |

#### `atc_level_4`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `C07AA` | `S01ED` |
| `EC:01628` | Tildrakizumab | `L04AC` | `L04AG` |
| `EC:00324` | Chlorambucil | `L01AA` | `N01AB` |
| `EC:00252` | Cabotegravir | `J05AJ` | `S03AA` |
| `EC:00709` | Fluticasone | `R07AB` | `S03BA` |

#### `atc_level_5`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01365` | Pyridoxine | `V01AA08` | `B03BA01` |
| `EC:01097` | Nafcillin | `D11AC08` | `S01AA19` |
| `EC:00084` | Ampicillin | `S01AA19` | `D11AC08` |
| `EC:00732` | Gabapentin | `N02BF01` | `N03AG03` |
| `EC:00301` | Cefpodoxime | `D11AC08` | `J01DD13` |

#### `atc_main`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `C07AA` | `S01ED` |
| `EC:01628` | Tildrakizumab | `L04AC` | `L04AG` |
| `EC:00324` | Chlorambucil | `L01AA` | `N01AB` |
| `EC:00252` | Cabotegravir | `J05AJ` | `S03AA` |
| `EC:00709` | Fluticasone | `R07AB` | `S03BA` |

#### `deleted`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00369` | Chlophedianol | `False` | `True` |

#### `deleted_reason`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00369` | Chlophedianol | `*None*` | `Duplicated with clofedanol` |

#### `is_cardiovascular`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01516` | Sotatercept | `True` | `False` |

#### `is_cell_therapy`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00937` | Lisocabtagene maraleucel | `False` | `True` |

#### `is_chemotherapy`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00631` | Estramustine | `True` | `False` |

#### `l1_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `cardiovascular system` | `sensory organs` |
| `EC:00324` | Chlorambucil | `antineoplastic and immunomodulating agents` | `nervous system` |
| `EC:00252` | Cabotegravir | `antiinfectives for systemic use` | `sensory organs` |
| `EC:00709` | Fluticasone | `respiratory system` | `sensory organs` |
| `EC:01115` | Nedocromil | `musculo-skeletal system` | `respiratory system` |

#### `l2_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `beta blocking agents` | `ophthalmologicals` |
| `EC:00324` | Chlorambucil | `antineoplastic agents` | `anesthetics` |
| `EC:00252` | Cabotegravir | `antivirals for systemic use` | `ophthalmological and otological preparations` |
| `EC:00709` | Fluticasone | `other respiratory system products` | `ophthalmological and otological preparations` |
| `EC:00827` | Indomethacin | `antiinflammatory and antirheumatic products` | `topical products for joint and muscular pain` |

#### `l3_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `beta blocking agents` | `antiglaucoma preparations and miotics` |
| `EC:00324` | Chlorambucil | `alkylating agents` | `anesthetics, general` |
| `EC:00252` | Cabotegravir | `direct acting antivirals` | `antiinfectives` |
| `EC:00709` | Fluticasone | `other respiratory system products` | `corticosteroids` |
| `EC:00827` | Indomethacin | `antiinflammatory and antirheumatic products, non-steroids` | `topical products for joint and muscular pain` |

#### `l4_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01356` | Propranolol | `beta blocking agents, non-selective` | `beta blocking agents` |
| `EC:01628` | Tildrakizumab | `interleukin inhibitors` | `monoclonal antibodies` |
| `EC:00324` | Chlorambucil | `nitrogen mustard analogues` | `halogenated hydrocarbons` |
| `EC:00252` | Cabotegravir | `integrase inhibitors` | `antiinfectives` |
| `EC:00709` | Fluticasone | `respiratory stimulants` | `corticosteroids` |

#### `l5_label`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:01365` | Pyridoxine | `food` | `cyanocobalamin` |
| `EC:01097` | Nafcillin | `sulfur compounds` | `ampicillin` |
| `EC:00084` | Ampicillin | `ampicillin` | `sulfur compounds` |
| `EC:00732` | Gabapentin | `gabapentin` | `aminobutyric acid` |
| `EC:00301` | Cefpodoxime | `sulfur compounds` | `cefpodoxime` |

#### `new_id`

| ID | Name | Old Value | New Value |
|----|------|-----------|-----------|
| `EC:00369` | Chlophedianol | `*None*` | `EC:00323` |

## Commits

19 commits between 17/11/2025 and 26/11/2025 from the following authors: Jacques Vergine, Li-Vern Teo, Nelson Alfonso, everycure

- [affff225](https://everycure@github.com/everycure-org/core-entities/commit/affff22563f59b0bc27a44e20e97be0abb128e89): Add drug_class column to drug_list (#120) (Jacques Vergine)
- [3dfc9752](https://everycure@github.com/everycure-org/core-entities/commit/3dfc975217a008e268cb234d6164e2cc80ea34b4): Fix disease list None in string column (#119) (Jacques Vergine)
- [1c9dea3a](https://everycure@github.com/everycure-org/core-entities/commit/1c9dea3ad0964e315e1915b08fed865cb67ca765): Refactor get_llm_response to simplify OpenAI model initialization (Nelson Alfonso)
- [81303d29](https://everycure@github.com/everycure-org/core-entities/commit/81303d29fa963c3d048144aff2fc0309d1fa126e): Refactor code structure for improved readability and maintainability (Nelson Alfonso)
- [de4a9570](https://everycure@github.com/everycure-org/core-entities/commit/de4a9570ce139e0dbf958535278f573f72fe471f): Refactor code structure for improved readability and maintainability (Nelson Alfonso)
- [ab5027f2](https://everycure@github.com/everycure-org/core-entities/commit/ab5027f277291b31f8f22c5fb083827d60d2a497): Implement feature X to enhance user experience and optimize performance (Nelson Alfonso)
- [dbe157dd](https://everycure@github.com/everycure-org/core-entities/commit/dbe157dd2e1d19c53c2afd137ed5b18de2ad4cd7): Update pydantic dependency from pydantic-ai-slim to pydantic-ai (Nelson Alfonso)
- [6b9ffea7](https://everycure@github.com/everycure-org/core-entities/commit/6b9ffea716fecba810b55e8288f1f1a8b618336e): Refactor import order and fix indentation in LiteLLMProvider initialization (Nelson Alfonso)
- [314f144a](https://everycure@github.com/everycure-org/core-entities/commit/314f144a76f73af1dc556dcaf08f506193454acf): Update src/core_entities/utils/llm_utils.py (Nelson Alfonso)
- [0c94a8e5](https://everycure@github.com/everycure-org/core-entities/commit/0c94a8e5a08114bfb09d9744006421326c06322d): Update src/core_entities/utils/llm_utils.py (Nelson Alfonso)
- [7c11ac02](https://everycure@github.com/everycure-org/core-entities/commit/7c11ac0270323e64a30ce3ebeb351ccc10b8fb0f): Implement LiteLLM integration by adding environment variables and updating LLM response handling (Nelson Alfonso)
- [4d5cc8b9](https://everycure@github.com/everycure-org/core-entities/commit/4d5cc8b9dac7ceca04e6de6715effa9492726c1a): added exceptions (Li-Vern Teo)
- [ede8a631](https://everycure@github.com/everycure-org/core-entities/commit/ede8a631a7b7e863bdbbb5307a8c54b631f2489e): Remove v from catalog dataset version (#115) (Jacques Vergine)
- [d7bbe38b](https://everycure@github.com/everycure-org/core-entities/commit/d7bbe38b4a324a93b544a058725a574a9bd78396): added more descriptive comments and removed redundancies (Li-Vern Teo)
- [7f9ba078](https://everycure@github.com/everycure-org/core-entities/commit/7f9ba0783fc4e3470ce3827d2b06703adfff4149): Update publishing to happen automatically on release PR merged (#114) (Jacques Vergine)
- [7ad11c55](https://everycure@github.com/everycure-org/core-entities/commit/7ad11c5565b3edafbba608148d46d5bd7d63893f): Update catalog dataset (Jacques Vergine)
- [8f7b119b](https://everycure@github.com/everycure-org/core-entities/commit/8f7b119b0df5aad703ea0d58cc09a7be224dc010): Release/v1.0.3-disease_list (#112) (everycure)
- [488deed6](https://everycure@github.com/everycure-org/core-entities/commit/488deed65e6fb08014ba9c7cf10f6b3b7bfefe2d): Release/v1.0.2-drug_list (#113) (everycure)
- [b39e766e](https://everycure@github.com/everycure-org/core-entities/commit/b39e766e1863a4b6ea0ba2b09f0f80962b5e3493): working code for drug atc codes (Li-Vern Teo)
