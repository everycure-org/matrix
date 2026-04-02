import asyncio
import logging
import multiprocessing
import re
from functools import partial

import aiohttp
import nest_asyncio
import pandas as pd
import pandera.pandas as pa
from pandas import DataFrame
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm

from core_entities.data.internal.schema.fda_drug_labels import (
    CURATED_DRUG_LIST_FOR_FDA_MATCH_SCHEMA,
    FDA_DRUG_LABELS_BIOSIMILAR_PARQUET_SCHEMA,
    FDA_DRUG_LABELS_BIOSIMILAR_TSV_SCHEMA,
    FDA_DRUG_LABELS_FILTERED_PARQUET_SCHEMA,
    FDA_DRUG_LABELS_FILTERED_TSV_SCHEMA,
    FDA_DRUG_LABELS_FOR_BIOSIMILAR_INPUT_SCHEMA,
    FDA_DRUG_LABELS_FOR_OTC_INPUT_SCHEMA,
    FDA_DRUG_LABELS_OTC_PARQUET_SCHEMA,
    FDA_DRUG_LABELS_OTC_TSV_SCHEMA,
    FDA_DRUG_LABELS_UNFILTERED_SCHEMA,
)
from core_entities.data.internal.schema.fda_drugs import FDA_DRUG_LIST_FOR_MATCHING_SCHEMA, FDA_DRUG_LIST_SCHEMA
from core_entities.pipelines.drug_llm_tags.drug_atc_codes import get_drug_atc_codes
from core_entities.utils.curation_utils import create_search_term_from_curated_drug_list, filter_dataframe_by_columns
from core_entities.utils.fda_drugs_utils import (
    extract_openfda_field,
    extract_product_active_ingredients,
    extract_product_marketing_status,
    filter_fda_rows,
    has_anda_application_number,
    match_drug_to_fda_worker,
    normalize_fda_results_to_dataframe,
)
from core_entities.utils.fda_labels_utils import run_sync as resolve_otc_monograph_labels
from core_entities.utils.fda_otc_monograph_drugs_utils import add_over_the_counter_status_if_needed
from core_entities.utils.python_utils import (
    ensure_python_list,
    ensure_string_list,
)

logger = logging.getLogger(__name__)

nest_asyncio.apply()


# ------------------------------------------------------------
# INGESTION NODES
# ------------------------------------------------------------


def get_boolean_column_schema(column_name: str):
    return pa.Column(
        nullable=False,
        parsers=pa.Parser(lambda col: col.apply(lambda x: True if x == "TRUE" else False)),
        checks=pa.Check(
            lambda col: col.apply(lambda x: x in ["TRUE", "FALSE", True, False]),
            title=f"{column_name} must be True or False",
        ),
    )


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(
            lambda df: df[
                [
                    "id",
                    "name",
                    "synonyms",
                    "aggregated_with",
                    "drug_class",
                    "therapeutic_area",
                    "drug_function",
                    "drug_target",
                    "approved_usa",
                    "is_antipsychotic",
                    "is_sedative",
                    "is_antimicrobial",
                    "is_glucose_regulator",
                    "is_chemotherapy",
                    "is_steroid",
                    "is_analgesic",
                    "is_cardiovascular",
                    "is_cell_therapy",
                    "deleted",
                    "deleted_reason",
                    "new_id",
                    "available_in_combo_with",
                ]
            ]
        ),
        columns={
            "id": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.startswith("EC:") and len(x) == 8 and x[3:].isdigit()),
                        title="id does start with 'EC:', is 8 characters long and only contains numbers after EC",
                    ),
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    ),
                ],
            ),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: "\n" not in x and x.strip() != ""),
                        title="Name must not contain newlines and must not be empty",
                    ),
                    pa.Check(
                        lambda col: len(col.str.lower().unique()) == len(col),
                        title="name must be unique",
                    ),
                ],
            ),
            "synonyms": pa.Column(nullable=True),
            "aggregated_with": pa.Column(nullable=True),
            "drug_class": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: "\n" not in x),
                        title="drug_class must not contain newlines",
                    ),
                ],
            ),
            "therapeutic_area": pa.Column(nullable=True),
            "drug_function": pa.Column(nullable=True),
            "drug_target": pa.Column(nullable=True),
            "approved_usa": pa.Column(
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x in ["APPROVED", "NOT_APPROVED", "DISCONTINUED"]),
                    title="approved_usa must be APPROVED, NOT_APPROVED or DISCONTINUED",
                ),
            ),
            "is_antipsychotic": get_boolean_column_schema("is_antipsychotic"),
            "is_sedative": get_boolean_column_schema("is_sedative"),
            "is_antimicrobial": get_boolean_column_schema("is_antimicrobial"),
            "is_glucose_regulator": get_boolean_column_schema("is_glucose_regulator"),
            "is_chemotherapy": get_boolean_column_schema("is_chemotherapy"),
            "is_steroid": get_boolean_column_schema("is_steroid"),
            "is_analgesic": get_boolean_column_schema("is_analgesic"),
            "is_cardiovascular": get_boolean_column_schema("is_cardiovascular"),
            "is_cell_therapy": get_boolean_column_schema("is_cell_therapy"),
            "deleted": pa.Column(
                nullable=True,
                parsers=pa.Parser(lambda col: col.apply(lambda x: True if x == "TRUE" else False)),
            ),
            "deleted_reason": pa.Column(
                nullable=True,
                parsers=pa.Parser(
                    lambda col: col.apply(lambda x: x if isinstance(x, str) and x.strip() != "" else None)
                ),
            ),
            "new_id": pa.Column(
                nullable=True,
                parsers=pa.Parser(
                    lambda col: col.apply(lambda x: x if isinstance(x, str) and x.strip() != "" else None)
                ),
            ),
            "available_in_combo_with": pa.Column(nullable=True),
        },
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.lower()),
                        title="name must be lowercase",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.strip()),
                        title="name must be stripped (no leading/trailing whitespace)",
                    ),
                    pa.Check(
                        lambda col: len(col.str.lower().unique()) == len(col),
                        title="name must be unique",
                    ),
                ],
            ),
            "synonyms": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: isinstance(x, list)),
                        title="synonyms must be a list",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.lower().strip() == item for item in x)),
                        title="synonyms must contain only stripped lowercase strings",
                    ),
                ],
            ),
            "aggregated_with": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: isinstance(x, list)),
                        title="aggregated_with must be a list",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.strip().capitalize() == item for item in x)),
                        title="aggregated_with must contain only stripped capitalized strings",
                    ),
                ],
            ),
            "drug_class": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_class must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_class must not have trailing whitespace",
                    ),
                ],
            ),
            "therapeutic_area": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="therapeutic_area must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="therapeutic_area must not have trailing whitespace",
                    ),
                ],
            ),
            "drug_function": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_function must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_function must not have trailing whitespace",
                    ),
                ],
            ),
            "drug_target": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_target must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() == x),
                        title="drug_target must not have trailing whitespace",
                    ),
                ],
            ),
            "approved_usa": pa.Column(nullable=False),
            "is_antipsychotic": pa.Column(dtype=bool, nullable=False),
            "is_sedative": pa.Column(dtype=bool, nullable=False),
            "is_antimicrobial": pa.Column(dtype=bool, nullable=False),
            "is_glucose_regulator": pa.Column(dtype=bool, nullable=False),
            "is_chemotherapy": pa.Column(dtype=bool, nullable=False),
            "is_steroid": pa.Column(dtype=bool, nullable=False),
            "is_analgesic": pa.Column(dtype=bool, nullable=False),
            "is_cardiovascular": pa.Column(dtype=bool, nullable=False),
            "is_cell_therapy": pa.Column(dtype=bool, nullable=False),
            "deleted": pa.Column(dtype=bool, nullable=False),
            "deleted_reason": pa.Column(dtype=str, nullable=True),
            "new_id": pa.Column(dtype=str, nullable=True),
            "available_in_combo_with": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: isinstance(x, list)),
                        title="available_in_combo_with must be a list",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.strip().lower() == item for item in x)),
                        title="available_in_combo_with must contain only stripped lowercase strings",
                    ),
                ],
            ),
        },
        unique=["id"],
        strict=True,
    )
)
def ingest_curated_drug_list(curated_drug_list: pd.DataFrame) -> pd.DataFrame:
    curated_drug_list.loc[:, "name"] = curated_drug_list.loc[:, "name"].apply(lambda x: x.lower().strip())

    def parse_string_column(x: str):
        if x.strip() == "":
            return None
        else:
            return x.strip()

    curated_drug_list.loc[:, "drug_class"] = curated_drug_list.loc[:, "drug_class"].apply(parse_string_column)
    curated_drug_list.loc[:, "therapeutic_area"] = curated_drug_list.loc[:, "therapeutic_area"].apply(
        parse_string_column
    )
    curated_drug_list.loc[:, "drug_function"] = curated_drug_list.loc[:, "drug_function"].apply(parse_string_column)
    curated_drug_list.loc[:, "drug_target"] = curated_drug_list.loc[:, "drug_target"].apply(parse_string_column)

    curated_drug_list.loc[:, "synonyms"] = curated_drug_list["synonyms"].apply(
        lambda x: [] if pd.isna(x) else [xx.lower().strip() for xx in x.split(";") if xx.strip() != ""]
    )
    curated_drug_list.loc[:, "aggregated_with"] = curated_drug_list.loc[:, "aggregated_with"].apply(
        lambda x: [] if pd.isna(x) else [xx.strip().capitalize() for xx in x.split(";") if xx.strip() != ""]
    )
    curated_drug_list.loc[:, "available_in_combo_with"] = curated_drug_list.loc[:, "available_in_combo_with"].apply(
        lambda x: [] if pd.isna(x) else [xx.strip().lower() for xx in x.split(";") if xx.strip() != ""]
    )

    return curated_drug_list


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["drugbank_id", "name"]]),
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
        },
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.lower()),
                        title="name must be lowercase",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.strip()),
                        title="name must be stripped (no leading/trailing whitespace)",
                    ),
                ],
            ),
        },
        strict=True,
    )
)
def ingest_drugbank_identifiers(drugbank_identifiers: pd.DataFrame) -> pd.DataFrame:
    drugbank_identifiers.loc[:, "name"] = drugbank_identifiers.loc[:, "name"].apply(lambda x: x.lower().strip())
    return drugbank_identifiers


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["Class ID", "Preferred Label"]]),
        columns={
            "Class ID": pa.Column(nullable=False),
            "Preferred Label": pa.Column(nullable=False),
        },
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "atc_code": pa.Column(nullable=False),
            "atc_label": pa.Column(nullable=False),
        },
        strict=True,
    )
)
def ingest_atc_labels(atc: pd.DataFrame) -> pd.DataFrame:
    atc["atc_code"] = atc["Class ID"].apply(
        lambda x: x.replace("http://purl.bioontology.org/ontology/ATC/", "").upper()
    )
    atc["atc_label"] = atc["Preferred Label"].apply(lambda x: x.lower().capitalize())
    atc = atc.drop(columns=["Class ID", "Preferred Label"])
    return atc


@pa.check_input(
    FDA_DRUG_LIST_SCHEMA,
)
def _validate_fda_drug_list_input(fda_results_df: pd.DataFrame) -> pd.DataFrame:
    return fda_results_df


@pa.check_output(FDA_DRUG_LIST_SCHEMA)
def ingest_fda_drug_json(fda_json: dict) -> pd.DataFrame:
    normalized_df = normalize_fda_results_to_dataframe(fda_json)
    return _validate_fda_drug_list_input(normalized_df)


# ------------------------------------------------------------
# RESOLUTION NODES
# ------------------------------------------------------------


@pa.check_input(
    pa.DataFrameSchema(
        parsers=pa.Parser(lambda df: df[["id", "name"]]),
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.lower().strip() == x),
                        title="name must be lowercase and stripped",
                    ),
                    pa.Check(
                        lambda col: len(col.str.lower().unique()) == len(col),
                        title="name must be unique",
                    ),
                ],
            ),
        },
        unique=["id"],
        strict=True,
    ),
    obj_getter="curated_drug_list",
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "normalization_success": pa.Column(nullable=False),
            "curie": pa.Column(nullable=False),
            "preferred_label": pa.Column(nullable=False),
            "category": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=False),
        },
        unique=["id"],
        strict=True,
    ),
    obj_getter="name_resolved_curies",
)
def resolve_drug_curies(
    curated_drug_list: pd.DataFrame,
    name_resolver_base_url: str,
    name_resolver_path: str,
    accepted_drug_categories: list[str],
    parallelism: int,
) -> pd.DataFrame:
    async def get_name_resolver_version(name_resolver_base_url: str) -> str:
        nr_openapi_json_url = f"{name_resolver_base_url}/openapi.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(nr_openapi_json_url) as response:
                json_response = await response.json()
                version = json_response["info"]["version"]
                return version

    @retry(wait=wait_exponential(multiplier=1, min=4, max=10), stop=stop_after_attempt(5))
    async def resolve_name(id: str, name: str, name_resolver_url: str, session: aiohttp.ClientSession) -> dict:
        try:
            async with session.get(name_resolver_url.format(name=name), timeout=60) as response:
                if response.status != 200:
                    raise Exception(f"Error for name {name}: {response.status}")

                result = await response.json()
                if not result:
                    raise Exception(f"No result for name {name}")

                return {
                    "id": id,
                    "name": name,
                    "normalization_success": True,
                    "output": result,
                }
        except TimeoutError:
            raise Exception(f"Timeout for name: {name}")
        except Exception as e:
            raise Exception(f"Error for name {name}: {str(e)}")

    async def resolve_all_names(
        curated_drug_list: pd.DataFrame,
        name_resolver_url: str,
        parallelism: int,
        max_connections: int,
    ) -> list[dict]:
        semaphore = asyncio.Semaphore(parallelism)

        async def call_name_resolver_with_semaphore(
            id: str, name: str, name_resolver_url: str, session: aiohttp.ClientSession
        ) -> dict:
            async with semaphore:
                return await resolve_name(id, name, name_resolver_url, session)

        # Configure connection pool with increased timeouts
        conn = aiohttp.TCPConnector(limit=max_connections, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)

        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            tasks = [
                call_name_resolver_with_semaphore(row["id"], row["name"], name_resolver_url, session)
                for _, row in curated_drug_list.iterrows()
            ]
            return await tqdm.gather(*tasks)

    def filter_name_resolver_result(name_resolver_result: dict, accepted_drug_categories: list[str]) -> list[dict]:
        if not name_resolver_result["normalization_success"]:
            logger.warning(f"No normalization success for {name_resolver_result['name']}")
            return {
                "id": name_resolver_result["id"],
                "name": name_resolver_result["name"],
                "normalization_success": False,
                "curie": None,
                "preferred_label": None,
                "category": None,
                "synonyms": None,
            }
        else:
            for match in name_resolver_result.get("output", []):
                if len(set(match["types"]) & set(accepted_drug_categories)) > 0:
                    return {
                        "id": name_resolver_result["id"],
                        "name": name_resolver_result["name"],
                        "normalization_success": True,
                        "curie": match["curie"],
                        "preferred_label": match["label"],
                        "category": [match["types"]],
                        "synonyms": [match["synonyms"]],
                    }
                else:
                    continue

            logger.warning(
                f"No match found for {name_resolver_result['name']} in the accepted_drug_categories. Picking the first match."
            )
            return {
                "id": name_resolver_result["id"],
                "name": name_resolver_result["name"],
                "normalization_success": True,
                "curie": name_resolver_result["output"][0]["curie"],
                "preferred_label": name_resolver_result["output"][0]["label"],
                "category": [name_resolver_result["output"][0]["types"]],
                "synonyms": [name_resolver_result["output"][0]["synonyms"]],
            }

    name_resolver_url = f"{name_resolver_base_url}{name_resolver_path}"
    name_resolver_version = asyncio.run(get_name_resolver_version(name_resolver_base_url))

    # ... call Name Resolver
    name_resolver_results = asyncio.run(
        resolve_all_names(curated_drug_list, name_resolver_url, parallelism, parallelism)
    )
    # ... filter name resolver results
    nameres_df = pd.DataFrame(
        [filter_name_resolver_result(result, accepted_drug_categories) for result in name_resolver_results]
    )

    return {
        "name_resolved_curies": nameres_df,
        "name_resolver_version": name_resolver_version,
    }


@pa.check_input(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "normalization_success": pa.Column(nullable=False),
            "curie": pa.Column(
                nullable=False,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: x.strip() != ""),
                    title="curie must not be empty",
                ),
            ),
            "preferred_label": pa.Column(nullable=False),
            "category": pa.Column(nullable=False),
            "synonyms": pa.Column(nullable=False),
        },
        unique=["id"],
        strict=True,
    )
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "normalization_success": pa.Column(nullable=False),
            "curie": pa.Column(nullable=False),
            "preferred_label": pa.Column(nullable=False),
            "category": pa.Column(nullable=False),
        },
        strict=True,
    ),
    obj_getter="normalized_drug_curies",
)
def normalize_drug_curies(drug_curies: pd.DataFrame, node_normalizer_base_url: str, node_normalizer_path: str) -> dict:
    async def get_nodenorm_version(node_normalizer_base_url: str) -> str:
        nn_openapi_json_url = f"{node_normalizer_base_url}/openapi.json"
        async with aiohttp.ClientSession() as session:
            async with session.get(nn_openapi_json_url) as response:
                json_response = await response.json()
                version = json_response["info"]["version"]
                return version

    async def normalize_curies(curies: list[str], node_normalizer_url: str) -> list[dict]:
        # Configure connection pool with increased timeouts
        conn = aiohttp.TCPConnector(limit=20, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)

        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            payload = {
                "curies": curies,
                "conflate": True,
                "description": True,
                "drug_chemical_conflate": False,
            }

            results = []
            async with session.post(node_normalizer_url, json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    for curie, node_data in result.items():
                        if node_data:
                            results.append(
                                {
                                    "original_curie": curie,
                                    "normalized_curie": node_data["id"]["identifier"] if node_data.get("id") else None,
                                    "all_categories": node_data.get("type", None),
                                }
                            )
                        else:
                            results.append(
                                {
                                    "original_id": curie,
                                    "id": None,
                                    "all_categories": None,
                                }
                            )

                    return results

                elif response.status == 502:
                    logger.error("Server overloaded (502). Retrying batch after delay...")
                    raise aiohttp.ClientError("Server overloaded")
                elif response.status == 422:
                    raise Exception("Invalid data (422).")
                else:
                    raise Exception(f"Error status {response.status}")

    node_normalizer_url = f"{node_normalizer_base_url}{node_normalizer_path}"
    curies = drug_curies["curie"].unique().tolist()
    nodenorm_version = asyncio.run(get_nodenorm_version(node_normalizer_base_url))
    normalized_results = asyncio.run(normalize_curies(curies, node_normalizer_url))
    normalized_df = pd.DataFrame(data=normalized_results, columns=["original_curie", "normalized_curie"])[
        ["original_curie", "normalized_curie"]
    ]

    failed_normalization = normalized_df[normalized_df["normalized_curie"].isna()]
    if len(failed_normalization) > 0:
        logger.warning(f"Failed to normalize {len(failed_normalization)} drugs")
        logger.warning(failed_normalization.head(15)["original_curie"])
    normalized_df = normalized_df[normalized_df["normalized_curie"].notna()].rename(columns={"original_curie": "curie"})

    merged_df = pd.merge(drug_curies, normalized_df, on="curie", how="left")
    merged_df["curie"] = merged_df["normalized_curie"].combine_first(merged_df["curie"])
    merged_df = merged_df.drop(columns=["normalized_curie", "synonyms"])

    return {"normalized_drug_curies": merged_df, "nodenorm_version": nodenorm_version}


@pa.check_input(
    pa.DataFrameSchema(
        columns={
            "drugbank_id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
        },
    ),
    obj_getter="drugbank_identifiers",
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "drugbank_id": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: col.apply(lambda x: pd.notna(x)),
                    title="drugbank_id must not be null",
                    ignore_na=False,
                    raise_warning=True,
                ),
            ),
        },
        unique=["id"],
        strict=True,
    )
)
def resolve_drugbank_ids(curated_drug_list: pd.DataFrame, drugbank_identifiers: pd.DataFrame) -> pd.DataFrame:
    drug_name_and_synonyms = curated_drug_list.explode("synonyms")[["id", "name", "synonyms"]]
    merged_df = pd.merge(
        pd.merge(
            drug_name_and_synonyms,
            drugbank_identifiers[["name", "drugbank_id"]].rename(columns={"drugbank_id": "drugbank_id_via_name"}),
            on="name",
            how="left",
        ),
        drugbank_identifiers[["name", "drugbank_id"]].rename(columns={"drugbank_id": "drugbank_id_via_synonym"}),
        left_on="synonyms",
        right_on="name",
        how="left",
    )

    merged_df["drugbank_id"] = merged_df["drugbank_id_via_name"].combine_first(merged_df["drugbank_id_via_synonym"])
    merged_df = merged_df.drop(columns=["drugbank_id_via_name", "drugbank_id_via_synonym", "name_y"]).rename(
        columns={"name_x": "name"}
    )
    merged_df = pd.merge(
        merged_df,
        drugbank_identifiers[["drugbank_id"]],
        on="drugbank_id",
        how="left",
    )
    merged_df = merged_df.drop_duplicates(subset=["name"], keep="first")
    merged_df = merged_df.drop(columns=["name", "synonyms"])

    return merged_df


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "name": pa.Column(nullable=False),
            "atc_name": pa.Column(nullable=True),
            "atc_synonym": pa.Column(nullable=True),
            "atc_main": pa.Column(nullable=True),
            "atc_level_1": pa.Column(nullable=True),
            "atc_level_2": pa.Column(nullable=True),
            "atc_level_3": pa.Column(nullable=True),
            "atc_level_4": pa.Column(nullable=True),
            "atc_level_5": pa.Column(nullable=True),
            "l1_label": pa.Column(nullable=True),
            "l2_label": pa.Column(nullable=True),
            "l3_label": pa.Column(nullable=True),
            "l4_label": pa.Column(nullable=True),
            "l5_label": pa.Column(nullable=True),
        },
        strict=True,
    )
)
def resolve_atc_codes(
    curated_drug_list: pd.DataFrame, atc_labels: pd.DataFrame, whocc_parallelism: int = 50
) -> pd.DataFrame:
    async def resolve_all_atc_codes(curated_drug_list: pd.DataFrame, parallelism: int) -> pd.DataFrame:
        semaphore = asyncio.Semaphore(parallelism)

        async def resolve_atc_code_with_semaphore(id: str, name: str, synonyms: list[str]) -> dict:
            async with semaphore:
                atc_codes = await get_drug_atc_codes(name, synonyms, session)
                atc_codes["id"] = id
                return atc_codes

        conn = aiohttp.TCPConnector(limit=parallelism, force_close=True)
        timeout = aiohttp.ClientTimeout(total=300, connect=60, sock_read=60)
        async with aiohttp.ClientSession(connector=conn, timeout=timeout) as session:
            tasks = [
                resolve_atc_code_with_semaphore(row["id"], row["name"], row["synonyms"])
                for _, row in curated_drug_list.iterrows()
            ]
            return await tqdm.gather(*tasks)

    def break_down_atc_code(atc_codes: pd.DataFrame):
        # Level 1: Anatomical main group (first character)
        atc_codes["atc_level_1"] = atc_codes.atc_main.apply(lambda x: x[0] if x is not None and len(x) >= 1 else None)
        # Level 2: Therapeutic subgroup (first 3 characters)
        atc_codes["atc_level_2"] = atc_codes.atc_main.apply(lambda x: x[:3] if x is not None and len(x) >= 3 else None)
        # Level 3: Pharmacological subgroup (first 4 characters)
        atc_codes["atc_level_3"] = atc_codes.atc_main.apply(lambda x: x[:4] if x is not None and len(x) >= 4 else None)
        # Level 4: Chemical subgroup (first 5 characters)
        atc_codes["atc_level_4"] = atc_codes.atc_main.apply(lambda x: x[:5] if x is not None and len(x) >= 5 else None)
        # Level 5: Chemical substance (all 7 characters)
        atc_codes["atc_level_5"] = atc_codes.atc_main.apply(lambda x: x if x is not None and len(x) == 7 else None)

        return atc_codes

    def add_atc_labels(atc_codes: pd.DataFrame, atc_labels: pd.DataFrame):
        return (
            atc_codes.merge(
                atc_labels.rename(columns={"atc_code": "atc_level_1", "atc_label": "l1_label"}),
                how="left",
                on="atc_level_1",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_2", "atc_label": "l2_label"}),
                how="left",
                on="atc_level_2",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_3", "atc_label": "l3_label"}),
                how="left",
                on="atc_level_3",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_4", "atc_label": "l4_label"}),
                how="left",
                on="atc_level_4",
            )
            .merge(
                atc_labels.rename(columns={"atc_code": "atc_level_5", "atc_label": "l5_label"}),
                how="left",
                on="atc_level_5",
            )
        )

    atc_codes = pd.DataFrame(asyncio.run(resolve_all_atc_codes(curated_drug_list, parallelism=whocc_parallelism)))
    atc_codes_broken_down = break_down_atc_code(atc_codes)

    atc_codes_and_labels = add_atc_labels(atc_codes_broken_down, atc_labels)

    result = pd.merge(curated_drug_list[["id", "name"]], atc_codes_and_labels, on="id", how="left")
    return result


def get_log_nan_check(column_name: str):
    return pa.Check(
        lambda col: col.apply(lambda x: pd.notna(x)),
        title=f"{column_name} must not be NaN",
        ignore_na=False,
        raise_warning=True,
    )


@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: len(col.unique()) == len(col),
                        title="id must be unique",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x.startswith("EC:") and len(x) == 8 and x[3:].isdigit()),
                        title="id does start with 'EC:', is 8 characters long and only contains numbers after EC",
                    ),
                ],
            ),
            "name": pa.Column(
                nullable=False,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="name must not be empty",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x[0].isupper()),
                        title="name must start with a capital letter",
                    ),
                    pa.Check(
                        lambda col: col.apply(lambda x: x == x.strip()),
                        title="name must be stripped (no leading/trailing whitespace)",
                    ),
                ],
            ),
            "translator_id": pa.Column(
                nullable=True,
                checks=pa.Check(
                    lambda col: len(col.unique()) == len(col),
                    title="translator_id must be unique",
                ),
            ),
            "drugbank_id": pa.Column(
                nullable=True,
                checks=[
                    get_log_nan_check("drugbank_id"),
                    pa.Check(
                        lambda col: len(col[col.notna()].unique()) == len(col[col.notna()]),
                        title="drugbank_id must not have duplicates",
                        raise_warning=True,
                    ),
                ],
            ),
            "synonyms": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.capitalize().strip() == item for item in x)),
                        title="aggregated_with must contain only stripped capitalized strings",
                    ),
                ],
            ),
            "aggregated_with": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: all(item.capitalize().strip() == item for item in x)),
                        title="aggregated_with must contain only stripped capitalized strings",
                    ),
                ],
            ),
            "drug_class": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_class must not be empty",
                    ),
                ],
            ),
            "therapeutic_area": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="therapeutic_area must not be empty",
                    ),
                ],
            ),
            "drug_function": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_function must not be empty",
                    ),
                ],
            ),
            "drug_target": pa.Column(
                nullable=True,
                checks=[
                    pa.Check(
                        lambda col: col.apply(lambda x: x.strip() != ""),
                        title="drug_target must not be empty",
                    ),
                ],
            ),
            "approved_usa": pa.Column(nullable=False),
            "is_antipsychotic": pa.Column(dtype=bool, nullable=False),
            "is_sedative": pa.Column(dtype=bool, nullable=False),
            "is_antimicrobial": pa.Column(dtype=bool, nullable=False),
            "is_glucose_regulator": pa.Column(dtype=bool, nullable=False),
            "is_chemotherapy": pa.Column(dtype=bool, nullable=False),
            "is_steroid": pa.Column(dtype=bool, nullable=False),
            "is_analgesic": pa.Column(dtype=bool, nullable=False),
            "is_cardiovascular": pa.Column(dtype=bool, nullable=False),
            "is_cell_therapy": pa.Column(dtype=bool, nullable=False),
            "atc_main": pa.Column(nullable=True),
            "atc_level_1": pa.Column(nullable=True),
            "atc_level_2": pa.Column(nullable=True),
            "atc_level_3": pa.Column(nullable=True),
            "atc_level_4": pa.Column(nullable=True),
            "atc_level_5": pa.Column(nullable=True),
            "l1_label": pa.Column(nullable=True),
            "l2_label": pa.Column(nullable=True),
            "l3_label": pa.Column(nullable=True),
            "l4_label": pa.Column(nullable=True),
            "l5_label": pa.Column(nullable=True),
            "deleted": pa.Column(dtype=bool, nullable=False),
            "deleted_reason": pa.Column(dtype=str, nullable=True),
            "new_id": pa.Column(dtype=str, nullable=True),
            "is_fda_generic_drug": pa.Column(dtype=bool, nullable=False),
        },
        unique=["id"],
        strict=True,
    ),
    obj_getter=0,
)
def merge_drug_lists(
    curated_drug_list: pd.DataFrame,
    normalized_drug_curies: pd.DataFrame,
    drug_list_with_atc_codes: pd.DataFrame,
    drug_list_with_drugbank_id: pd.DataFrame,
    drug_list_with_fda_generic_drug_info: pd.DataFrame,
    release_columns: list[str],
    drug_exception_list: list[str],
) -> pd.DataFrame:
    normalized_drug_curies = normalized_drug_curies.rename(columns={"curie": "translator_id"})

    df = (
        curated_drug_list.merge(normalized_drug_curies, on="id", how="left")
        .merge(drug_list_with_atc_codes, on="id", how="left")
        .merge(drug_list_with_drugbank_id, on="id", how="left")
        .merge(drug_list_with_fda_generic_drug_info, on="id", how="left")
    )
    # fill nan values in the is_fda_generic_drug column with False
    df["is_fda_generic_drug"] = df["is_fda_generic_drug"].fillna(False).astype(bool)
    # filter out drugs in the drug_exception_list (these are drugs that we want to keep in the curated drug list for completeness.
    df = df[~df["name"].isin(drug_exception_list)]
    df.loc[:, "name"] = df.loc[:, "name"].apply(lambda x: x.capitalize())
    df.loc[:, "synonyms"] = df.loc[:, "synonyms"].apply(
        lambda x: None if x is None else [xx.strip().capitalize() for xx in x]
    )
    df = df[release_columns]

    return df, df


def publish_drug_list(drug_list: pd.DataFrame) -> dict:
    return {
        "drug_list_parquet": drug_list,
        "drug_list_tsv": drug_list,
        "drug_list_bq": drug_list,
        "drug_list_bq_latest": drug_list,
    }


@pa.check_input(
    CURATED_DRUG_LIST_FOR_FDA_MATCH_SCHEMA,
    obj_getter="curated_drug_list",
)
@pa.check_input(
    FDA_DRUG_LIST_FOR_MATCHING_SCHEMA,
    obj_getter="fda_drug_list",
)
@pa.check_output(FDA_DRUG_LABELS_UNFILTERED_SCHEMA)
def resolve_fda_drugs_matches_to_drug_list_unfiltered(
    curated_drug_list: pd.DataFrame,
    fda_drug_list: pd.DataFrame,
    curated_drug_list_columns_to_use_for_matching: list[str],
    fda_drug_list_columns_to_use_for_matching: list[str],
    filter_curated_drug_list_params: dict[str, str],
) -> pd.DataFrame:
    """Match curated drugs to FDA drug applications.

    For single-ingredient drugs, any search term matching any FDA field counts.
    For only available with drugs (only_available_with is present), we also check whether the
    FDA product's active_ingredients cover the aggregated set.
    """

    # Apply filtering to the curated drug list before matching to reduce the number of comparisons and improve performance.
    # We filter out drugs that are not approved in the USA (as defined in the parameters file) etc.
    filtered_curated_drug_list = filter_dataframe_by_columns(curated_drug_list, filter_curated_drug_list_params)

    # Build work items for multiprocessing
    work_items = create_search_term_from_curated_drug_list(
        filtered_curated_drug_list, curated_drug_list_columns_to_use_for_matching
    )
    work_items_records = work_items.to_dict(orient="records")

    if len(work_items_records) == 0:
        logger.info("No curated drugs available for FDA matching after filtering")
        return pd.DataFrame(
            columns=[
                "fda_rows",
                "fda_match_count",
                "drug_name",
                "id",
                "search_terms",
                "available_in_combo_with",
            ]
        )

    worker_fn = partial(
        match_drug_to_fda_worker,
        fda_drug_list=fda_drug_list,
        fda_drug_list_columns_to_use_for_matching=fda_drug_list_columns_to_use_for_matching,
    )
    num_workers = min(multiprocessing.cpu_count(), len(work_items_records))
    chunksize = max(1, len(work_items_records) // (num_workers * 4))
    logger.info(f"Matching {len(work_items_records)} drugs to FDA rows using {num_workers} workers")

    with multiprocessing.Pool(processes=num_workers) as pool:
        fda_raw_matches = list(
            sync_tqdm(
                pool.imap(worker_fn, work_items_records, chunksize=chunksize),
                total=len(work_items_records),
                desc="Resolving FDA matches",
                unit="drug",
            )
        )

    fda_raw_matches_df = pd.DataFrame(fda_raw_matches)
    fda_raw_matches_df.loc[:, "search_terms"] = fda_raw_matches_df["search_terms"].apply(ensure_string_list)
    fda_raw_matches_df.loc[:, "available_in_combo_with"] = fda_raw_matches_df["available_in_combo_with"].apply(
        ensure_string_list
    )

    return fda_raw_matches_df


@pa.check_input(FDA_DRUG_LABELS_UNFILTERED_SCHEMA)
@pa.check_output(
    FDA_DRUG_LABELS_FILTERED_PARQUET_SCHEMA,
    obj_getter=0,
)
@pa.check_output(
    FDA_DRUG_LABELS_FILTERED_TSV_SCHEMA,
    obj_getter=1,
)
def resolve_fda_drugs_matches_to_drug_list_filtered(
    fda_drug_labels_unfiltered: pd.DataFrame,
) -> tuple[DataFrame, DataFrame]:
    """
    Filter the unfiltered FDA drug matches to determine which drugs are considered FDA generic drugs (according to the criteria), and to extract relevant information from the matched FDA rows for those drugs.
    """
    # we make sure that the filtered_fda_values column contains lists (even if empty) to avoid issues with downstream processing where we expect lists.
    filtered_fda_values = fda_drug_labels_unfiltered.apply(filter_fda_rows, axis=1).apply(ensure_python_list)
    # Count the number of matched FDA rows for each drug after filtering, which will be used to determine whether the drug is an FDA generic drug or not.
    filtered_fda_values_count = filtered_fda_values.str.len().fillna(0).astype(int)

    fda_drug_labels_filtered = fda_drug_labels_unfiltered.copy()
    fda_drug_labels_filtered.loc[:, "filtered_fda_values"] = filtered_fda_values
    fda_drug_labels_filtered.loc[:, "filtered_fda_values_count"] = filtered_fda_values_count
    fda_drug_labels_filtered.loc[:, "is_fda_generic_drug"] = filtered_fda_values_count.gt(0)
    # We consider a drug to be a biologic if any of the matched FDA rows have an application number that starts with "BLA" (Biologics License Application).
    fda_drug_labels_filtered["is_biologics"] = filtered_fda_values.apply(
        lambda rows: any(
            str(item.get("application_number", "")).lower().startswith("bla")
            for item in (rows or [])
            if isinstance(item, dict)
        )
    )
    fda_drug_labels_filtered.loc[:, "brand_name"] = filtered_fda_values.apply(
        lambda rows: extract_openfda_field(rows, "brand_name")
    )
    fda_drug_labels_filtered.loc[:, "generic_name"] = filtered_fda_values.apply(
        lambda rows: extract_openfda_field(rows, "generic_name")
    )
    fda_drug_labels_filtered.loc[:, "substance_name"] = filtered_fda_values.apply(
        lambda rows: extract_openfda_field(rows, "substance_name")
    )
    fda_drug_labels_filtered.loc[:, "active_ingredients"] = filtered_fda_values.apply(
        extract_product_active_ingredients
    )
    # We determine the marketing status of the drug based on the information in the matched FDA rows, such as the presence of specific fields or values that indicate whether the drug is currently marketed, discontinued, etc.
    fda_drug_labels_filtered.loc[:, "marketing_status"] = filtered_fda_values.apply(extract_product_marketing_status)
    # We determine whether the drug is an ANDA (Abbreviated New Drug Application) generic drug based on the presence of an application number that starts with "ANDA" in any of the matched FDA rows,
    # which indicates that the drug is a generic version of a previously approved drug.
    fda_drug_labels_filtered.loc[:, "is_anda"] = filtered_fda_values.apply(has_anda_application_number)

    fda_drug_labels_filtered_tsv = fda_drug_labels_filtered.drop(
        columns=["fda_rows", "filtered_fda_values"], inplace=False
    )

    return fda_drug_labels_filtered, fda_drug_labels_filtered_tsv


@pa.check_input(FDA_DRUG_LABELS_FOR_BIOSIMILAR_INPUT_SCHEMA)
@pa.check_output(
    FDA_DRUG_LABELS_BIOSIMILAR_PARQUET_SCHEMA,
    obj_getter=0,
)
@pa.check_output(
    FDA_DRUG_LABELS_BIOSIMILAR_TSV_SCHEMA,
    obj_getter=1,
)
def resolve_fda_drugs_that_are_biosimilar_and_are_generic(
    fda_drug_labels_filtered: pd.DataFrame,
    fda_purple_book_params: dict,
    fda_purple_book_data: DataFrame | None = None,
) -> tuple[DataFrame, DataFrame]:
    """
    From the filtered FDA drug matches, we want to determine which drugs are biosimilars (a specific type of generic drug that is similar to biologic drugs) and to extract relevant information about the biosimilar drugs.
    We determine whether a drug is a biosimilar based on the presence of specific information in the matched FDA rows, such as the BLA status as defined in the params.fda_purple_book_params and the application numbers.
    We also cross-reference with the FDA Purple Book data, if available, to enhance our identification of biosimilars.
    """

    if fda_drug_labels_filtered.empty:
        enriched_tsv = fda_drug_labels_filtered.drop(columns=["fda_rows", "filtered_fda_values"], errors="ignore")
        return fda_drug_labels_filtered, enriched_tsv
    # We first determine which BLA types are relevant for identifying biosimilars based on the provided parameters. We normalise the BLA types to ensure consistent comparison later on.
    raw_bla_types = fda_purple_book_params.get("generic_status_bla_type", None)
    if raw_bla_types is None:
        raw_bla_types = (fda_purple_book_params.get("fda_purple_book", {}) or {}).get("generic_status_bla_type", [])

    allowed_bla_types = {
        bla_type.lower().strip() for bla_type in ensure_string_list(raw_bla_types) if bla_type.strip() != ""
    }

    def normalize_bla_number(value: object) -> str | None:
        """
        Normalise BLA numbers by stripping whitespace, removing non-digit characters, and converting to a consistent string format. If the resulting string is empty or contains no digits, return None.
        """
        raw_value = str(value).strip()
        if raw_value == "":
            return None

        digits_only = re.sub(r"\D", "", raw_value)
        if digits_only == "":
            return None

        return str(int(digits_only))

    def extract_application_numbers_from_fda_values(fda_values: object) -> list[str]:
        """
        Extract application numbers from the FDA values. We look for application numbers in the main FDA rows, as well as in the openfda field and the products field, as the relevant information can be present in different places depending on the specific drug and how the data is structured.
        """
        rows = ensure_python_list(fda_values) or []
        application_numbers: list[str] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            application_numbers.extend(ensure_string_list(row.get("application_number")))

        return ensure_string_list(application_numbers)

    def extract_bla_types_from_fda_values(fda_values: object) -> list[str]:
        """
        Extract BLA types from the FDA values. We look for BLA types in the main FDA rows, as well as in the openfda field and the products field, as the relevant information can be present in different places depending on the specific drug and how the data is structured.
        """
        rows = ensure_python_list(fda_values) or []
        extracted: list[str] = []
        for row in rows:
            if not isinstance(row, dict):
                continue

            for key in ["bla_type", "BLA Type", "BLA_TYPE", "blaType"]:
                if key in row:
                    extracted.extend(ensure_string_list(row.get(key)))

            openfda = row.get("openfda")
            if isinstance(openfda, dict):
                for key in ["bla_type", "BLA Type", "BLA_TYPE", "blaType"]:
                    if key in openfda:
                        extracted.extend(ensure_string_list(openfda.get(key)))

            products = ensure_python_list(row.get("products")) or []
            for product in products:
                if not isinstance(product, dict):
                    continue
                for key in ["bla_type", "BLA Type", "BLA_TYPE", "blaType"]:
                    if key in product:
                        extracted.extend(ensure_string_list(product.get(key)))

        return ensure_string_list(extracted)

    # Depending on the structure of the FDA drug labels data, the relevant information for identifying biosimilars may be present in either the "filtered_fda_values" column (which contains the FDA rows after filtering) or the original "fda_rows" column.
    # We check which column is available and use it as the source for extracting BLA types and application numbers.
    fda_values_source = (
        "filtered_fda_values" if "filtered_fda_values" in fda_drug_labels_filtered.columns else "fda_rows"
    )
    # We extract the BLA types and application numbers from the FDA values using the defined functions.
    # This information will be used to determine which drugs are biosimilars and to enrich the data with relevant details about the biosimilar drugs.
    fda_drug_labels_filtered.loc[:, "biosimilar_bla_types"] = fda_drug_labels_filtered[fda_values_source].apply(
        extract_bla_types_from_fda_values
    )
    # We extract the application numbers from the FDA values,
    # as the presence of specific application numbers (e.g., those starting with "BLA") can be an important indicator for identifying biosimilars.
    fda_drug_labels_filtered.loc[:, "biosimilar_application_numbers"] = fda_drug_labels_filtered[
        fda_values_source
    ].apply(extract_application_numbers_from_fda_values)

    def resolve_column_name(df: pd.DataFrame, candidates: list[str]) -> str | None:
        """
        Resolve the column name in the given DataFrame that matches any of the candidate names (case-insensitive, ignoring leading/trailing whitespace). This is useful for handling variations in column naming conventions across different datasets. If a match is found, the actual column name from the DataFrame is returned; otherwise, None is returned.
        """
        normalized_name_to_column = {str(column).strip().lower(): column for column in df.columns}
        for candidate in candidates:
            matched = normalized_name_to_column.get(candidate.lower())
            if matched is not None:
                return matched
        return None

    def merge_unique_strings(values: list[object]) -> list[str]:
        """
        Merge a list of values into a list of unique strings, normalizing each string by stripping whitespace and converting to lowercase.
        """
        merged: list[str] = []
        seen: set[str] = set()

        for value in values:
            for item in ensure_string_list(value):
                normalized = item.lower().strip()
                if normalized in seen:
                    continue
                seen.add(normalized)
                merged.append(item)

        return merged

    # If the FDA Purple Book data is available and not empty, we use it to enhance our identification of biosimilars.
    # We look for the relevant columns in the Purple Book data that contain the BLA numbers and BLA types,
    # normalise the BLA numbers, and create a lookup dictionary to map normalised BLA numbers to their corresponding BLA types.
    # We then use this lookup to enrich our FDA drug labels data with biosimilar information based on the application numbers extracted from the FDA values.
    if fda_purple_book_data is not None and not fda_purple_book_data.empty:
        purple_book_df = fda_purple_book_data.copy()
        bla_number_column = resolve_column_name(purple_book_df, ["BLA Number", "bla number", "bla_number"])
        bla_type_column = resolve_column_name(purple_book_df, ["BLA Type", "bla type", "bla_type"])

        if bla_number_column is None or bla_type_column is None:
            logger.warning(
                "Could not find BLA Number/BLA Type columns in purple book data. Available columns: %s",
                list(purple_book_df.columns),
            )
        else:
            purple_book_df = purple_book_df[[bla_number_column, bla_type_column]].copy()
            # We normalise the BLA numbers and BLA types in the Purple Book data to ensure consistent comparison later on when we enrich our FDA drug labels data with biosimilar information.
            purple_book_df.loc[:, "normalized_bla_number"] = purple_book_df[bla_number_column].apply(
                normalize_bla_number
            )
            purple_book_df.loc[:, "normalized_bla_type"] = purple_book_df[bla_type_column].apply(
                lambda value: str(value).strip()
            )
            purple_book_df = purple_book_df[
                purple_book_df["normalized_bla_number"].notna() & (purple_book_df["normalized_bla_type"] != "")
            ]

            purple_book_lookup = (
                purple_book_df.groupby("normalized_bla_number")["normalized_bla_type"]
                .apply(lambda values: merge_unique_strings(values.tolist()))
                .to_dict()
            )
            # We enrich our FDA drug labels data with biosimilar information based on the application numbers extracted from the FDA values
            # and the lookup dictionary created from the Purple Book data. For each drug, we look at the application numbers extracted from the FDA values,
            # normalise them, and check if they match any BLA numbers in the Purple Book lookup. If a match is found,
            # we merge the corresponding BLA types from the Purple Book data into our FDA drug labels data for that drug.
            fda_drug_labels_filtered.loc[:, "biosimilar_bla_types"] = fda_drug_labels_filtered.apply(
                lambda row: merge_unique_strings(
                    [
                        row.get("biosimilar_bla_types", []),
                        [
                            bla_type
                            for application_number in ensure_string_list(row.get("biosimilar_application_numbers", []))
                            for bla_type in purple_book_lookup.get(normalize_bla_number(application_number), [])
                        ],
                    ]
                ),
                axis=1,
            )

    default_false = pd.Series(False, index=fda_drug_labels_filtered.index, dtype=bool)
    is_biologics = fda_drug_labels_filtered.get("is_biologics", default_false).fillna(False).astype(bool)
    # We determine whether a drug is a biosimilar generic drug based on the presence of relevant BLA types in the FDA values.
    # If any of the BLA types extracted from the FDA values for a drug match the allowed BLA types defined in the parameters,
    # We consider that drug to be a biosimilar and therefore a type of generic drug.
    biosimilar_is_generic = fda_drug_labels_filtered["biosimilar_bla_types"].apply(
        lambda types: any(bla_type.lower().strip() in allowed_bla_types for bla_type in ensure_string_list(types))
    )

    existing_generic = fda_drug_labels_filtered.get("is_fda_generic_drug", default_false).fillna(False).astype(bool)
    # We update the "is_fda_generic_drug" column to include drugs that are identified as biosimilars based on the BLA types,
    # while ensuring that we do not classify biologic drugs as generic drugs.
    fda_drug_labels_filtered.loc[:, "is_fda_generic_drug"] = existing_generic.where(
        ~is_biologics, biosimilar_is_generic
    )

    enriched_tsv = fda_drug_labels_filtered.drop(columns=["fda_rows", "filtered_fda_values"], errors="ignore")
    return fda_drug_labels_filtered, enriched_tsv


@pa.check_input(FDA_DRUG_LABELS_FOR_OTC_INPUT_SCHEMA)
@pa.check_output(
    FDA_DRUG_LABELS_OTC_PARQUET_SCHEMA,
    obj_getter=0,
)
@pa.check_output(
    FDA_DRUG_LABELS_OTC_TSV_SCHEMA,
    obj_getter=1,
)
@pa.check_output(
    pa.DataFrameSchema(
        columns={
            "id": pa.Column(nullable=False),
            "is_fda_generic_drug": pa.Column(dtype=bool, nullable=False),
        },
        unique=["id"],
        strict=True,
    ),
    obj_getter=2,
)
def resolve_fda_drugs_that_are_otc_monograph(
    fda_drug_labels_filtered: pd.DataFrame,
    fda_labels_params: dict,
) -> tuple[DataFrame, DataFrame, DataFrame]:
    if fda_drug_labels_filtered.empty:
        fda_drug_labels_filtered.loc[:, "otc_monograph_checked"] = False
        fda_drug_labels_filtered.loc[:, "otc_monograph_status"] = "NOT_CHECKED"
        fda_drug_labels_filtered.loc[:, "otc_monograph_application_numbers"] = [
            [] for _ in range(len(fda_drug_labels_filtered))
        ]
        fda_drug_labels_filtered.loc[:, "otc_monograph_total_matches"] = 0
        fda_drug_labels_filtered.loc[:, "otc_monograph_error_msg"] = ""
        fda_drug_labels_filtered.loc[:, "is_otc_monograph"] = False

        enriched_tsv = fda_drug_labels_filtered.drop(columns=["fda_rows", "filtered_fda_values"], errors="ignore")
        return (
            fda_drug_labels_filtered,
            enriched_tsv,
            fda_drug_labels_filtered.loc[
                fda_drug_labels_filtered["is_fda_generic_drug"], ["id", "is_fda_generic_drug"]
            ].copy(),
        )

    default_false = pd.Series(False, index=fda_drug_labels_filtered.index, dtype=bool)
    is_fda_generic = fda_drug_labels_filtered.get("is_fda_generic_drug", default_false).fillna(False).astype(bool)
    otc_candidates_mask = ~is_fda_generic
    # We consider drugs that are not already identified as FDA generic drugs as candidates for OTC monograph status,
    # as OTC monograph drugs are a specific category of drugs that can be marketed without a prescription and are not typically classified as generic drugs.
    otc_candidate_drug_names = (
        fda_drug_labels_filtered.loc[otc_candidates_mask, "drug_name"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )
    otc_candidate_drug_names = [name for name in otc_candidate_drug_names if name]
    otc_candidate_name_set = set(otc_candidate_drug_names)

    logger.info("Checking %d drugs for OTC monograph status via openFDA labels", len(otc_candidate_drug_names))

    otc_result_lookup: dict[str, dict] = {}
    if otc_candidate_drug_names:
        otc_results = resolve_otc_monograph_labels(
            drugs=otc_candidate_drug_names,
            config=fda_labels_params or {},
        )
        otc_result_lookup = {
            result.drug_name: {
                "otc_monograph_status": result.status,
                "otc_monograph_application_numbers": ensure_string_list(result.application_numbers),
                "otc_monograph_total_matches": int(result.total_matches),
                "otc_monograph_error_msg": result.error_msg,
            }
            for result in otc_results
        }

    def get_otc_result_for_row(drug_name: object) -> dict:
        normalized_name = str(drug_name).strip()
        return otc_result_lookup.get(normalized_name, {})

    fda_drug_labels_filtered.loc[:, "otc_monograph_checked"] = fda_drug_labels_filtered["drug_name"].apply(
        lambda name: str(name).strip() in otc_candidate_name_set
    )
    fda_drug_labels_filtered.loc[:, "otc_monograph_status"] = fda_drug_labels_filtered["drug_name"].apply(
        lambda name: get_otc_result_for_row(name).get("otc_monograph_status", "NOT_CHECKED")
    )
    fda_drug_labels_filtered.loc[:, "otc_monograph_application_numbers"] = fda_drug_labels_filtered["drug_name"].apply(
        lambda name: get_otc_result_for_row(name).get("otc_monograph_application_numbers", [])
    )
    fda_drug_labels_filtered.loc[:, "otc_monograph_total_matches"] = fda_drug_labels_filtered["drug_name"].apply(
        lambda name: get_otc_result_for_row(name).get("otc_monograph_total_matches", 0)
    )
    fda_drug_labels_filtered.loc[:, "otc_monograph_error_msg"] = fda_drug_labels_filtered["drug_name"].apply(
        lambda name: get_otc_result_for_row(name).get("otc_monograph_error_msg", "")
    )
    fda_drug_labels_filtered.loc[:, "is_otc_monograph"] = fda_drug_labels_filtered["otc_monograph_status"].eq(
        "OTC_MONOGRAPH"
    )

    fda_drug_labels_filtered.loc[:, "is_fda_generic_drug"] = (
        is_fda_generic | fda_drug_labels_filtered["is_otc_monograph"]
    )
    if "marketing_status" not in fda_drug_labels_filtered.columns:
        fda_drug_labels_filtered.loc[:, "marketing_status"] = [[] for _ in range(len(fda_drug_labels_filtered))]
    fda_drug_labels_filtered.loc[:, "marketing_status"] = fda_drug_labels_filtered.apply(
        lambda row: add_over_the_counter_status_if_needed(
            row["marketing_status"],
            row["is_otc_monograph"],
        ),
        axis=1,
    )

    fda_generic_drug_list_with_ec_ids = fda_drug_labels_filtered.loc[
        fda_drug_labels_filtered["is_fda_generic_drug"], ["id", "is_fda_generic_drug"]
    ].copy()

    enriched_tsv = fda_drug_labels_filtered.drop(columns=["fda_rows", "filtered_fda_values"], errors="ignore")
    return (fda_drug_labels_filtered, enriched_tsv, fda_generic_drug_list_with_ec_ids)
