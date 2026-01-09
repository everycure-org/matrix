# Release Comparison Report

**New Release:** `v1.0.6-disease_list`

**Base Release:** `v1.0.3-disease_list`

**New Release File:** `gs://mtrx-us-central1-hub-dev-storage/kedro/data/core-entities/disease_list/v1.0.6/03_primary/release/ec-disease-list.parquet`

**Base Release File:** `gs://data.dev.everycure.org/data/01_RAW/disease_list/v1.0.3/ec-disease-list.parquet`

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

*No value changes detected*

## Commits

29 commits between 17/11/2025 and 27/11/2025 from the following authors: Jacques Vergine, Li-Vern Teo, Nelson Alfonso, everycure

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
